"""gelbench.transfer — cross-family transfer experiments (spec §8, RQ1).

Two designs, always split *by family* (never a shuffle-split across families):

  * Pairwise matrix: train on all of family A, test on all of family B, for every
    ordered pair. The diagonal (A -> A) is filled from the leak-free within-family
    CV score, not a train==test resubstitution.
  * Leave-one-family-out (LOFO): train on the union of every family but one, test
    on the held-out family. Also returns per-molecule predictions, which the
    applicability-domain analysis (RQ2) correlates with AD distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import fit_full, predict_proba_full, summarize, majority_baseline


def _slice(rep, families):
    mask = np.isin(rep.family, list(families))
    return rep.X.loc[mask], rep.y[mask], rep.ids[mask], rep.family[mask]


def transfer_pair(rep, train_fam: str, test_fam: str, model: str, cfg) -> dict:
    """Train on all of ``train_fam``; evaluate on all of ``test_fam``."""
    Xtr, ytr, _, _ = _slice(rep, [train_fam])
    Xte, yte, _, _ = _slice(rep, [test_fam])
    fitted = fit_full(Xtr, ytr, model, cfg)
    prob = predict_proba_full(fitted, Xte)
    s = summarize(yte, prob)
    return {
        "representation": rep.name, "model": model,
        "train_family": train_fam, "test_family": test_fam,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "test_majority_baseline": majority_baseline(yte),
        "accuracy": s["accuracy"], "auc": s["auc"],
        "is_diagonal": train_fam == test_fam,
    }


def transfer_matrix(rep, model: str, cfg, within_lookup: dict | None = None) -> pd.DataFrame:
    """All ordered (train, test) family pairs for one representation/model.

    ``within_lookup`` maps (representation, family, model) -> {"accuracy","auc"}
    to fill the diagonal with the honest within-family CV score.
    """
    families = sorted(set(rep.family))
    rows = []
    for a in families:
        for b in families:
            if a == b:
                key = (rep.name, a, model)
                wl = (within_lookup or {}).get(key)
                Xte, yte, _, _ = _slice(rep, [b])
                rows.append({
                    "representation": rep.name, "model": model,
                    "train_family": a, "test_family": b,
                    "n_train": int(len(yte)), "n_test": int(len(yte)),
                    "test_majority_baseline": majority_baseline(yte),
                    "accuracy": wl["accuracy"] if wl else np.nan,
                    "auc": wl["auc"] if wl else np.nan,
                    "is_diagonal": True,
                })
            else:
                rows.append(transfer_pair(rep, a, b, model, cfg))
    return pd.DataFrame(rows)


def leave_one_family_out(rep, model: str, cfg):
    """Train on the union of all families but one; test on the held-out family.

    Returns (summary_rows_df, per_molecule_predictions_df).
    """
    families = sorted(set(rep.family))
    summ, preds = [], []
    for held in families:
        train_fams = [f for f in families if f != held]
        Xtr, ytr, _, _ = _slice(rep, train_fams)
        Xte, yte, ids_te, fam_te = _slice(rep, [held])
        fitted = fit_full(Xtr, ytr, model, cfg)
        prob = predict_proba_full(fitted, Xte)
        s = summarize(yte, prob)
        summ.append({
            "representation": rep.name, "model": model,
            "held_out_family": held,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_majority_baseline": majority_baseline(yte),
            "accuracy": s["accuracy"], "auc": s["auc"],
        })
        pred = (prob >= 0.5).astype(int)
        preds.append(pd.DataFrame({
            "id": ids_te, "family": fam_te, "y": yte,
            "prob": prob, "pred": pred, "correct": (pred == yte).astype(int),
            "representation": rep.name, "model": model,
        }))
    return pd.DataFrame(summ), pd.concat(preds, ignore_index=True)
