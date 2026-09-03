"""gelbench.plots — figures for the report.

Every figure is regenerated from the tracked results CSVs; nothing is hand-drawn.
The headline figure (spec §13) is performance-drop vs applicability-domain
distance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FAM_COLORS = {"nucleoside": "#1b9e77", "tripeptide": "#d95f02", "peptide_expt": "#7570b3"}
REP_MARKERS = {"rdkit2d": "o", "morgan": "s", "mordred": "^", "alvadesc": "D"}


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_headline_drop_vs_ad(perf: pd.DataFrame, out: Path,
                            admetric: str = "median_novelty") -> None:
    """Off-family AUC drop vs training-set AD distance (the lead figure)."""
    sub = perf.dropna(subset=["auc_drop", admetric]).copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for _, r in sub.iterrows():
        ax.scatter(r[admetric], r["auc_drop"],
                   color=FAM_COLORS.get(r["family"], "#555"),
                   marker=REP_MARKERS.get(r["representation"], "o"),
                   s=70, edgecolor="k", linewidth=0.5, alpha=0.9, zorder=3)
    if len(sub) >= 3 and sub[admetric].nunique() > 1:
        rho, p = stats.spearmanr(sub[admetric], sub["auc_drop"])
        b, a = np.polyfit(sub[admetric], sub["auc_drop"], 1)
        xs = np.linspace(sub[admetric].min(), sub[admetric].max(), 50)
        ax.plot(xs, a + b * xs, "--", color="k", lw=1, zorder=2)
        ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}  (p = {p:.2g})\nn = {len(sub)} points",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(f"training-set AD distance ({admetric.replace('median_', 'median ')})")
    ax.set_ylabel("off-family AUC drop\n(within-family CV − leave-one-family-out)")
    ax.set_title("Does applicability-domain distance explain the transfer drop?")
    _family_rep_legend(ax, sub)
    _save(fig, out)


def _family_rep_legend(ax, sub):
    from matplotlib.lines import Line2D
    fam_handles = [Line2D([0], [0], marker="o", color="w", label=f,
                          markerfacecolor=c, markeredgecolor="k", markersize=9)
                   for f, c in FAM_COLORS.items() if f in set(sub["family"])]
    rep_handles = [Line2D([0], [0], marker=REP_MARKERS.get(r, "o"), color="w",
                          label=r, markerfacecolor="0.6", markeredgecolor="k",
                          markersize=9)
                   for r in sub["representation"].unique()]
    leg1 = ax.legend(handles=fam_handles, title="held-out family",
                     loc="lower right", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=rep_handles, title="representation",
              loc="upper right", fontsize=8, title_fontsize=8)


# --------------------------------------------------------------------------- #
def fig_within_family(within: pd.DataFrame, out: Path, model: str = "logreg") -> None:
    sub = within[within["model"] == model].copy()
    reps = sorted(sub["representation"].unique())
    fams = sorted(sub["family"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.4))
    width = 0.8 / len(reps)
    x = np.arange(len(fams))
    for i, rep in enumerate(reps):
        rr = sub[sub["representation"] == rep].set_index("family").reindex(fams)
        ax.bar(x + i * width, rr["auc"], width, label=rep,
               yerr=[rr["auc"] - rr["auc_lo"], rr["auc_hi"] - rr["auc"]],
               capsize=3, edgecolor="k", linewidth=0.4)
    for j, fam in enumerate(fams):
        base = sub[sub["family"] == fam]["majority_baseline"].iloc[0]
        ax.hlines(0.5, x[j] - 0.1, x[j] + 0.8, color="0.4", lw=0.8)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(fams)
    ax.set_ylabel("within-family AUC (pooled OOF, 95% CI)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_title(f"Within-family baselines ({model}); dotted = AUC 0.5")
    ax.legend(title="representation", fontsize=8)
    _save(fig, out)


# --------------------------------------------------------------------------- #
def fig_transfer_heatmap(matrix: pd.DataFrame, out: Path,
                         representation: str, model: str = "logreg") -> None:
    sub = matrix[(matrix["representation"] == representation) &
                 (matrix["model"] == model)]
    fams = sorted(set(sub["train_family"]) | set(sub["test_family"]))
    M = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for _, r in sub.iterrows():
        M.loc[r["train_family"], r["test_family"]] = r["auc"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M.values.astype(float), vmin=0.4, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    for i in range(len(fams)):
        for j in range(len(fams)):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v < 0.75 else "k", fontsize=10)
    ax.set_xlabel("test family"); ax.set_ylabel("train family")
    ax.set_title(f"Transfer AUC — {representation} ({model})\ndiagonal = within-family CV")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC")
    _save(fig, out)


# --------------------------------------------------------------------------- #
def fig_molecule_ad(permol: pd.DataFrame, out: Path,
                    admetric: str = "novelty", model: str = "logreg") -> None:
    """Off-family correctness vs a per-molecule AD distance."""
    pm = permol[permol["model"] == model].dropna(subset=[admetric])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    groups = [pm.loc[pm["correct"] == 1, admetric], pm.loc[pm["correct"] == 0, admetric]]
    parts = ax.violinplot(groups, showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], ["#4daf4a", "#e41a1c"]):
        pc.set_facecolor(c); pc.set_alpha(0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["correct", "incorrect"])
    ax.set_ylabel(f"per-molecule {admetric}")
    if len(groups[0]) > 5 and len(groups[1]) > 5:
        u, p = stats.mannwhitneyu(groups[1], groups[0], alternative="greater")
        ax.set_title(f"Off-family {admetric} vs correctness ({model})\n"
                     f"Mann–Whitney (incorrect>correct) p = {p:.2g}")
    _save(fig, out)


# --------------------------------------------------------------------------- #
def fig_descriptor_comparison(matrix: pd.DataFrame, out: Path,
                              model: str = "logreg") -> None:
    """RQ3: off-diagonal (cross-family) transfer AUC by representation."""
    off = matrix[(~matrix["is_diagonal"]) & (matrix["model"] == model)]
    reps = sorted(off["representation"].unique())
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = [off[off["representation"] == r]["auc"].dropna().values for r in reps]
    ax.boxplot(data, tick_labels=reps, showmeans=True)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).normal(0, 0.04, len(d)),
                   d, s=18, color="0.3", alpha=0.6, zorder=3)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_ylabel("cross-family transfer AUC (off-diagonal)")
    ax.set_title(f"Descriptor comparison ({model}); dotted = AUC 0.5")
    _save(fig, out)
