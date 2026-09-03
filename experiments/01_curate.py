#!/usr/bin/env python
"""Phase 2 — curate the multi-family gelation dataset.

Downloads/copies every source, standardises and deduplicates, writes
data/processed/gelation_all.csv, data/raw/manifest.json, and fills PROVENANCE.md.
Prints a per-family count table and label balance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gelbench import load_config
from gelbench.data import build_dataset


def main() -> int:
    cfg = load_config()
    summary = build_dataset(cfg)

    print("\n=== gelation_all.csv provenance summary ===\n")
    header = f"{'family':<14}{'n':>6}{'gel':>6}{'non':>6}{'major.base':>12}"
    print(header)
    print("-" * len(header))
    for fam, info in summary["families"].items():
        print(f"{fam:<14}{info['n']:>6}{info['n_gel']:>6}{info['n_nongel']:>6}"
              f"{info['majority_baseline']:>12.3f}")
    print("-" * len(header))
    print(f"{'TOTAL':<14}{summary['n_total']:>6}")

    print("\nDrops (per family):")
    for fam, d in summary["drops"].items():
        print(f"  {fam:<14} parsed={d['n_parsed']:>5}  "
              f"unparseable={d['unparseable']:>3}  intra_dupes={d['intra_dupes']:>3}")

    coll = summary["cross_family_collisions"]
    print(f"\nCross-family SMILES collisions: {coll['n']} "
          f"(dropped by family: {coll['by_family_dropped'] or 'none'})")

    small = [f for f, i in summary["families"].items()
             if i["n"] < cfg["cv"]["min_family_size"]]
    if small:
        print(f"\n[flag] families below min_family_size="
              f"{cfg['cv']['min_family_size']} (transfer numbers unreliable): "
              f"{', '.join(small)}")

    print(f"\nWrote: {cfg['paths']['dataset'].relative_to(cfg['_root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
