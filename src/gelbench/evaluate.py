"""gelbench.evaluate — the leak-free nested-CV evaluator (spec §8).

Leak-freedom is *structural*: every data-dependent step (inf-cleaning,
imputation, variance filter, univariate selection, scaling, and the
hyperparameter choice) lives inside a scikit-learn ``Pipeline`` that is fit only
on the training rows of each fold, wrapped by ``GridSearchCV`` for the inner
hyperparameter search. Nothing is fit on data outside the training fold.

Public entry points:
  * ``build_pipeline`` — the leak-free preprocessing+model pipeline.
  * ``pooled_nested_cv`` — repeated stratified nested CV -> per-molecule OOF
    probabilities (each molecule averaged across repeats).
  * ``within_family_scores`` — pooled accuracy/AUC + bootstrap CI over molecules
    + majority-class baseline, for one (representation, family, model).
  * ``fit_full`` / ``predict_proba_full`` — train on all of one family (transfer).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# These fire deep inside worker processes and carry no information for us:
# SelectKBest returns all features when k exceeds the post-variance count, and
# the balanced logistic regression occasionally brushes its iteration cap.
warnings.filterwarnings("ignore", message="k=.* is greater than n_features")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRIMARY_MODEL = "logreg"
MODELS = ["logreg", "rf", "hgb"]


class CleanInf(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN so the imputer can handle them uniformly."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(np.isfinite(X), X, np.nan)


def _make_classifier(model_name: str, seed: int):
    if model_name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced",
                                  random_state=seed)
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=seed, n_jobs=1)
    if model_name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed,
                                              class_weight="balanced")
    raise ValueError(f"unknown model {model_name!r}")


# Inner hyperparameter grids (kept small: with a handful of families, a big grid
# just adds noise and runtime, spec §0/§8).
_GRIDS = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {"clf__max_depth": [None, 8]},
    "hgb": {"clf__learning_rate": [0.05, 0.1]},
}


def build_pipeline(model_name: str, n_features: int, k_select: int, seed: int) -> Pipeline:
    """Leak-free pipeline: clean -> impute -> variance -> select -> scale -> clf.

    ``k_select`` is clamped to the number of available features.
    """
    k = min(int(k_select), int(n_features))
    return Pipeline([
        ("clean", CleanInf()),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("clf", _make_classifier(model_name, seed)),
    ])


def _grid_search(model_name: str, n_features: int, cfg) -> GridSearchCV:
    inner = StratifiedKFold(n_splits=cfg["cv"]["inner_splits"], shuffle=True,
                            random_state=cfg["seed"])
    pipe = build_pipeline(model_name, n_features, cfg["cv"]["select_k"], cfg["seed"])
    return GridSearchCV(pipe, _GRIDS[model_name], cv=inner, scoring="roc_auc",
                        n_jobs=-1, refit=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def summarize(y: np.ndarray, prob: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
    }


def majority_baseline(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float(max(np.mean(y == 1), np.mean(y == 0)))


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, cfg, seed_offset: int = 0) -> dict:
    """Percentile bootstrap CI over MOLECULES (not folds), for acc and AUC."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    n = len(y)
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    B = cfg["stats"]["n_bootstrap"]
    accs, aucs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], prob[idx]
        accs.append(accuracy_score(yb, (pb >= 0.5).astype(int)))
        if len(np.unique(yb)) > 1:
            aucs.append(roc_auc_score(yb, pb))
    lo, hi = (1 - cfg["stats"]["ci"]) / 2 * 100, (1 + cfg["stats"]["ci"]) / 2 * 100
    out = {"acc_lo": float(np.percentile(accs, lo)),
           "acc_hi": float(np.percentile(accs, hi))}
    if aucs:
        out["auc_lo"] = float(np.percentile(aucs, lo))
        out["auc_hi"] = float(np.percentile(aucs, hi))
    else:
        out["auc_lo"] = out["auc_hi"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Nested CV (within-family)
# --------------------------------------------------------------------------- #
@dataclass
class OOFResult:
    prob: np.ndarray            # per-molecule averaged OOF probability
    y: np.ndarray
    best_params: list           # best params chosen in each outer fold


def pooled_nested_cv(X: pd.DataFrame, y: np.ndarray, model_name: str, cfg) -> OOFResult:
    """Repeated stratified nested CV; average each molecule's OOF prob over repeats."""
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n = len(y)
    outer = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["within_family"]["outer_splits"],
        n_repeats=cfg["cv"]["within_family"]["outer_repeats"],
        random_state=cfg["seed"],
    )
    prob_sum = np.zeros(n)
    prob_cnt = np.zeros(n)
    best_params = []
    for tr, te in outer.split(X, y):
        gs = _grid_search(model_name, X.shape[1], cfg)
        gs.fit(X.iloc[tr], y[tr])
        p = gs.predict_proba(X.iloc[te])[:, 1]
        prob_sum[te] += p
        prob_cnt[te] += 1
        best_params.append(gs.best_params_)
    prob = prob_sum / np.maximum(prob_cnt, 1)
    return OOFResult(prob=prob, y=y, best_params=best_params)


def within_family_scores(rep, family: str, model_name: str, cfg) -> dict:
    """Leak-free within-family baseline for one (representation, family, model)."""
    mask = rep.family == family
    X = rep.X.loc[mask]
    y = rep.y[mask]
    res = pooled_nested_cv(X, y, model_name, cfg)
    row = {
        "representation": rep.name,
        "family": family,
        "model": model_name,
        "n": int(mask.sum()),
        "n_features": int(X.shape[1]),
        "majority_baseline": majority_baseline(y),
        **summarize(y, res.prob),
        **bootstrap_ci(y, res.prob, cfg),
    }
    return row


# --------------------------------------------------------------------------- #
# Full-family fit (transfer): train on all of A, predict on B
# --------------------------------------------------------------------------- #
def fit_full(X: pd.DataFrame, y: np.ndarray, model_name: str, cfg):
    """Fit the leak-free pipeline (with inner-CV hyperparameter search) on all
    of a training family. The inner GridSearchCV still only sees training data.
    """
    gs = _grid_search(model_name, X.shape[1], cfg)
    gs.fit(X, np.asarray(y))
    return gs


def predict_proba_full(fitted, X: pd.DataFrame) -> np.ndarray:
    return fitted.predict_proba(X)[:, 1]
