#!/usr/bin/env python
"""Phase 6 — applicability domain vs performance (RQ2).

Computes, for each molecule, its AD distance (leverage, ECFP4 novelty, kNN) from
the leave-one-family-out training union, then asks whether AD distance explains
the off-family performance drop.

Writes:
  results/ad_per_molecule.csv    id, family, AD distances, off-family correctness
  results/ad_vs_performance.csv  per (rep, model, family): within vs LOFO + median AD
  results/ad_correlation.json    Spearman/rank-sum summaries answering RQ2

Needs Phases 4-5 (within_family.csv, leave_one_family_out.csv, lofo_predictions.csv).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gelbench import load_config
from gelbench.domain import applicability_domain
from gelbench.features import load_representation


def _ad_table(cfg) -> pd.DataFrame:
    """AD distance of every molecule relative to its LOFO training union.

    AD is computed once in the rdkit2d descriptor space (leverage/kNN) and ECFP4
    space (novelty); it depends only on which family is held out, not on the ML
    representation or model.
    """
    rep = load_representation(cfg, "rdkit2d")
    families = sorted(set(rep.family))
    parts = []
    for held in families:
        train_mask = rep.family != held
        query_mask = rep.family == held
        ad = applicability_domain(rep.X, rep.smiles, train_mask, query_mask,
                                  rep.ids, cfg)
        ad["family"] = held
        parts.append(ad)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    cfg = load_config()
    res = cfg["paths"]["results"]

    preds_p = res / "lofo_predictions.csv"
    within_p = res / "within_family.csv"
    lofo_p = res / "leave_one_family_out.csv"
    for p in (preds_p, within_p, lofo_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run Phases 4 and 5 first.")

    ad = _ad_table(cfg)                       # id, family, leverage, novelty, ...
    preds = pd.read_csv(preds_p)              # id, family, y, prob, correct, rep, model
    within = pd.read_csv(within_p)
    lofo = pd.read_csv(lofo_p)

    # --- per-molecule table (RQ2, molecule level) --------------------------
    permol = preds.merge(ad.drop(columns="family"), on="id", how="left")
    permol = permol.sort_values(["representation", "model", "family", "id"],
                                kind="stable")
    permol.to_csv(res / "ad_per_molecule.csv", index=False,
                  lineterminator="\n", float_format="%.6f")

    # --- per-family table (RQ2, family level): within vs off-family + AD ----
    med = (ad.groupby("family")
             .agg(median_leverage=("leverage", "median"),
                  median_novelty=("novelty", "median"),
                  median_knn=("knn_distance", "median"),
                  frac_leverage_flagged=("leverage_flag", "mean"))
             .reset_index())
    within_key = within.rename(columns={"auc": "within_auc", "accuracy": "within_acc"})
    lofo_key = lofo.rename(columns={"held_out_family": "family",
                                    "auc": "lofo_auc", "accuracy": "lofo_acc"})
    perf = lofo_key.merge(
        within_key[["representation", "model", "family", "within_auc", "within_acc",
                    "majority_baseline"]],
        on=["representation", "model", "family"], how="left")
    perf = perf.merge(med, on="family", how="left")
    perf["auc_drop"] = perf["within_auc"] - perf["lofo_auc"]
    perf["acc_drop"] = perf["within_acc"] - perf["lofo_acc"]
    perf = perf.sort_values(["representation", "model", "family"], kind="stable")
    cols = ["representation", "model", "family", "n_test", "majority_baseline",
            "within_auc", "lofo_auc", "auc_drop", "within_acc", "lofo_acc",
            "acc_drop", "median_novelty", "median_leverage", "median_knn",
            "frac_leverage_flagged"]
    perf[cols].to_csv(res / "ad_vs_performance.csv", index=False,
                      lineterminator="\n", float_format="%.6f")

    # --- correlations answering RQ2 ----------------------------------------
    corr = {}
    # Family level: does a larger AD distance go with a larger AUC drop?
    for admetric in ("median_novelty", "median_leverage", "median_knn"):
        sub = perf[["auc_drop", admetric]].dropna()
        if len(sub) >= 3 and sub[admetric].nunique() > 1:
            rho, p = stats.spearmanr(sub[admetric], sub["auc_drop"])
            corr[f"family_spearman_{admetric}_vs_aucdrop"] = {
                "rho": float(rho), "p": float(p), "n": int(len(sub))}

    # Molecule level (primary model, pooled over representations): are wrong
    # off-family predictions more novel / higher leverage than correct ones?
    pm = permol[permol["model"] == "logreg"]
    for admetric in ("novelty", "leverage", "knn_distance"):
        wrong = pm.loc[pm["correct"] == 0, admetric].dropna()
        right = pm.loc[pm["correct"] == 1, admetric].dropna()
        if len(wrong) > 5 and len(right) > 5:
            u, p = stats.mannwhitneyu(wrong, right, alternative="greater")
            corr[f"molecule_{admetric}_wrong_gt_right"] = {
                "median_wrong": float(wrong.median()),
                "median_right": float(right.median()),
                "mannwhitney_U": float(u), "p": float(p),
                "n_wrong": int(len(wrong)), "n_right": int(len(right))}

    # Molecule-level TRUST test (the actionable RQ2 result): within the pooled
    # off-family predictions, is accuracy higher for molecules that sit INSIDE
    # the training applicability domain than outside it? This is well powered
    # (~1000s of molecules) and, unlike the 3-family drop-vs-AD correlation, is
    # not confounded by each family's within-family ceiling.
    trust_rows = []
    for model in sorted(permol["model"].unique()):
        pm_m = permol[permol["model"] == model].dropna(subset=["leverage"])
        inside = pm_m[pm_m["leverage"] <= pm_m["leverage_star"]]
        outside = pm_m[pm_m["leverage"] > pm_m["leverage_star"]]
        trust_rows += [
            {"model": model, "group": "inside_h*", "n": len(inside),
             "accuracy": float(inside["correct"].mean())},
            {"model": model, "group": "outside_h*", "n": len(outside),
             "accuracy": float(outside["correct"].mean())},
        ]
        q = pd.qcut(pm_m["leverage"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                    duplicates="drop")
        for name, g in pm_m.groupby(q, observed=True):
            trust_rows.append({"model": model, "group": f"leverage_{name}",
                               "n": int(len(g)), "accuracy": float(g["correct"].mean())})
    trust = pd.DataFrame(trust_rows)
    trust.to_csv(res / "ad_trust.csv", index=False, lineterminator="\n",
                 float_format="%.6f")
    lg = trust[trust["model"] == "logreg"].set_index("group")
    if "inside_h*" in lg.index and "outside_h*" in lg.index:
        corr["molecule_offfamily_accuracy_inside_vs_outside_domain"] = {
            "inside_h*_accuracy": float(lg.loc["inside_h*", "accuracy"]),
            "inside_h*_n": int(lg.loc["inside_h*", "n"]),
            "outside_h*_accuracy": float(lg.loc["outside_h*", "accuracy"]),
            "outside_h*_n": int(lg.loc["outside_h*", "n"]),
        }

    (res / "ad_correlation.json").write_text(json.dumps(corr, indent=2),
                                             encoding="utf-8")

    # --- console summary ---------------------------------------------------
    print("\n=== AD vs performance (per representation/model/family) ===")
    show = perf[perf["model"] == "logreg"][
        ["representation", "family", "within_auc", "lofo_auc", "auc_drop",
         "median_novelty", "frac_leverage_flagged"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== RQ2 correlations ===")
    print(json.dumps(corr, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
