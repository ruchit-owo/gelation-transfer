#!/usr/bin/env python
"""Phase 4 — leak-free within-family baselines (RQ baseline).

Repeated stratified 5x10 nested CV per (representation, family, model). Writes
results/within_family.csv with pooled accuracy/AUC, bootstrap CIs over molecules,
and the majority-class baseline. Resumable: existing rows are skipped.

    python experiments/03_within_family.py
    python experiments/03_within_family.py --reps rdkit2d --models logreg
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
from gelbench.evaluate import MODELS, within_family_scores
from gelbench.features import available_representations, load_representation

KEY = ["representation", "family", "model"]
COLS = KEY + ["n", "n_features", "majority_baseline", "accuracy",
             "acc_lo", "acc_hi", "auc", "auc_lo", "auc_hi"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=None,
                    help="subset of representations (default: all available)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of models (default: all)")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["paths"]["results"] / "within_family.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    reps = args.reps or available_representations(cfg)
    models = args.models or MODELS

    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLS)
    done_keys = set(map(tuple, done[KEY].values)) if len(done) else set()

    rows = done.to_dict("records")
    for rep_name in reps:
        rep = load_representation(cfg, rep_name)
        families = sorted(set(rep.family))
        for family in families:
            for model in models:
                if (rep_name, family, model) in done_keys:
                    continue
                print(f"[within] {rep_name:<9} {family:<13} {model:<7} ...",
                      flush=True, end="")
                row = within_family_scores(rep, family, model, cfg)
                rows.append({k: row.get(k) for k in COLS})
                print(f" acc={row['accuracy']:.3f} (base {row['majority_baseline']:.3f}) "
                      f"auc={row['auc']:.3f} [{row['auc_lo']:.3f},{row['auc_hi']:.3f}]",
                      flush=True)
                # Persist after every combo so the run is resumable.
                df = pd.DataFrame(rows)[COLS].sort_values(KEY, kind="stable")
                df.to_csv(out, index=False, lineterminator="\n",
                          float_format="%.6f")

    print(f"\nWrote {out.relative_to(cfg['_root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
