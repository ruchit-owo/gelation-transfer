#!/usr/bin/env python
"""Phase 7 — figures and summary tables for the report.

Regenerates every figure and table from the tracked results CSVs. No numbers are
hand-entered; tables are written to tables/ and figures to figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gelbench import load_config
from gelbench import plots


def _md_table(df: pd.DataFrame, path: Path, floatfmt: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n",
              float_format=floatfmt)
    # Lightweight markdown (no external dep).
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = load_config()
    res, figs, tabs = cfg["paths"]["results"], cfg["paths"]["figures"], cfg["paths"]["tables"]

    within = pd.read_csv(res / "within_family.csv")
    matrix = pd.read_csv(res / "transfer_matrix.csv")
    lofo = pd.read_csv(res / "leave_one_family_out.csv")
    perf = pd.read_csv(res / "ad_vs_performance.csv")
    permol = pd.read_csv(res / "ad_per_molecule.csv")

    # --- figures -----------------------------------------------------------
    plots.fig_headline_drop_vs_ad(perf, figs / "fig_headline_drop_vs_ad",
                                  admetric="median_novelty")
    plots.fig_within_family(within, figs / "fig_within_family")
    plots.fig_molecule_ad(permol, figs / "fig_molecule_ad", admetric="novelty")
    plots.fig_descriptor_comparison(matrix, figs / "fig_descriptor_comparison")
    for rep in sorted(matrix["representation"].unique()):
        plots.fig_transfer_heatmap(matrix, figs / f"fig_transfer_{rep}",
                                   representation=rep)

    # --- tables ------------------------------------------------------------
    wf = within[within["model"] == "logreg"][
        ["representation", "family", "n", "majority_baseline", "accuracy",
         "auc", "auc_lo", "auc_hi"]]
    _md_table(wf, tabs / "table_within_family")

    off = matrix[~matrix["is_diagonal"]]
    tsum = (off.groupby(["representation", "model"])
               .agg(mean_transfer_auc=("auc", "mean"),
                    min_transfer_auc=("auc", "min"),
                    n_pairs=("auc", "size")).reset_index())
    _md_table(tsum, tabs / "table_transfer_summary")

    lo = lofo[lofo["model"] == "logreg"][
        ["representation", "held_out_family", "n_test", "test_majority_baseline",
         "accuracy", "auc"]]
    _md_table(lo, tabs / "table_lofo")

    ad = perf[perf["model"] == "logreg"][
        ["representation", "family", "within_auc", "lofo_auc", "auc_drop",
         "median_novelty", "median_leverage", "frac_leverage_flagged"]]
    _md_table(ad, tabs / "table_ad_vs_performance")

    print("Figures written to", figs.relative_to(cfg["_root"]))
    for p in sorted(figs.glob("*.png")):
        print("  ", p.name)
    print("Tables written to", tabs.relative_to(cfg["_root"]))
    for p in sorted(tabs.glob("*.md")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
