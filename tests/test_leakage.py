"""Leakage guard (spec §11, critical).

Construct a synthetic dataset of pure-noise features plus one genuinely
predictive feature (a noisy copy of the label). Then show:

  * Selecting the most label-correlated features on ALL data before CV inflates
    the CV score far above chance -- that is the leakage we must avoid.
  * Our leak-free pipeline (SelectKBest fit INSIDE each fold) scores near chance
    on pure noise -- no leakage -- yet still recovers a genuine signal when one
    exists. Both facts together prove the pipeline neither leaks nor is inert.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from gelbench import load_config
from gelbench.evaluate import build_pipeline

SEED = 0
N, P, K = 80, 500, 5


def _make_noise(seed):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (N // 2))
    rng.shuffle(y)
    X = rng.standard_normal((N, P))          # pure noise, no signal
    return X, y


def _cv_auc(estimator, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    prob = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob)


def _leaky_auc(X, y):
    # THE MISTAKE: pick top-K label-correlated features using ALL rows, then CV.
    sel = SelectKBest(f_classif, k=K).fit(X, y)
    Xsel = sel.transform(X)
    return _cv_auc(LogisticRegression(max_iter=5000), Xsel, y)


def _leakfree_auc(X, y):
    # CORRECT: selection lives inside the pipeline, refit within every fold.
    pipe = build_pipeline("logreg", n_features=X.shape[1], k_select=K, seed=SEED)
    return _cv_auc(pipe, X, y)


def test_selection_on_all_data_leaks_but_pipeline_does_not():
    X, y = _make_noise(SEED)
    leaky = _leaky_auc(X, y)
    leakfree = _leakfree_auc(X, y)
    # Leak-free is near chance on pure noise ...
    assert leakfree < 0.62, f"leak-free AUC unexpectedly high ({leakfree:.3f})"
    # ... while selecting on all data is clearly inflated ...
    assert leaky > 0.70, f"leaky AUC not inflated as expected ({leaky:.3f})"
    # ... with a large, unambiguous gap.
    assert leaky - leakfree > 0.15, f"gap too small: leaky={leaky:.3f} free={leakfree:.3f}"


def test_pipeline_recovers_a_genuine_signal():
    # Add one honestly predictive feature (noisy copy of the label). The
    # leak-free pipeline should score well above chance -- it is not inert.
    X, y = _make_noise(SEED + 1)
    rng = np.random.default_rng(123)
    signal = y + rng.normal(0, 0.5, size=N)
    Xs = np.column_stack([X, signal])
    auc = _leakfree_auc(Xs, y)
    assert auc > 0.80, f"pipeline failed to detect a real signal (AUC {auc:.3f})"
