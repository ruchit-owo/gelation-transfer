#!/usr/bin/env python
"""Phase 5 — cross-family transfer (RQ1).

Writes:
  results/transfer_matrix.csv        pairwise train->test AUC/accuracy (+ diagonal)
  results/leave_one_family_out.csv   each family held out once
  results/lofo_predictions.csv       per-molecule off-family predictions (for RQ2)

Resumable per (representation, model). The diagonal of the matrix is filled from
results/within_family.csv (leak-free CV), so run Phase 4 first.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", message="k=.* is greater than n_features")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gelbench import load_config
from gelbench.evaluate import MODELS
from gelbench.features import available_representations, load_representation
from gelbench.transfer import transfer_matrix, leave_one_family_out


def _within_lookup(cfg) -> dict:
    path = cfg["paths"]["results"] / "within_family.csv"
    if not path.exists():
        return {}
    wf = pd.read_csv(path)
    return {(r.representation, r.family, r.model): {"accuracy": r.accuracy, "auc": r.auc}
            for r in wf.itertuples()}


def _append(path: Path, df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=key_cols, keep="last")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    res = cfg["paths"]["results"]
    res.mkdir(parents=True, exist_ok=True)
    reps = args.reps or available_representations(cfg)
    models = args.models or MODELS
    wl = _within_lookup(cfg)

    mat_path = res / "transfer_matrix.csv"
    lofo_path = res / "leave_one_family_out.csv"
    pred_path = res / "lofo_predictions.csv"

    done_mat = set()
    if mat_path.exists():
        m = pd.read_csv(mat_path)
        done_mat = set(map(tuple, m[["representation", "model"]].drop_duplicates().values))

    for rep_name in reps:
        rep = load_representation(cfg, rep_name)
        for model in models:
            if (rep_name, model) in done_mat:
                continue
            print(f"[transfer] {rep_name:<9} {model:<7} matrix + LOFO ...", flush=True)
            mat = transfer_matrix(rep, model, cfg, wl)
            lofo, preds = leave_one_family_out(rep, model, cfg)

            _append(mat_path, mat,
                    ["representation", "model", "train_family", "test_family"]) \
                .sort_values(["representation", "model", "train_family", "test_family"],
                             kind="stable") \
                .to_csv(mat_path, index=False, lineterminator="\n", float_format="%.6f")
            _append(lofo_path, lofo, ["representation", "model", "held_out_family"]) \
                .sort_values(["representation", "model", "held_out_family"], kind="stable") \
                .to_csv(lofo_path, index=False, lineterminator="\n", float_format="%.6f")
            _append(pred_path, preds, ["representation", "model", "id"]) \
                .sort_values(["representation", "model", "family", "id"], kind="stable") \
                .to_csv(pred_path, index=False, lineterminator="\n", float_format="%.6f")

            offdiag = mat[~mat["is_diagonal"]]
            print(f"           off-diagonal AUC mean={offdiag['auc'].mean():.3f}  "
                  f"LOFO AUC={dict(zip(lofo['held_out_family'], lofo['auc'].round(3)))}",
                  flush=True)

    print(f"\nWrote {mat_path.name}, {lofo_path.name}, {pred_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
