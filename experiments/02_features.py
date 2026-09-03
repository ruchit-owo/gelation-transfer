#!/usr/bin/env python
"""Phase 3 — compute and cache molecular representations.

Computes the open representations (RDKit 2D, Morgan/ECFP4, optional Mordred) and
exports per-family `.smi` inputs for the commercial alvaDesc trial. RUN EARLY:
the alvaDesc licence is time-limited (spec §7).

    python experiments/02_features.py              # compute missing caches
    python experiments/02_features.py --recompute  # force recompute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gelbench import load_config
from gelbench.features import compute_and_cache, available_representations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    made = compute_and_cache(cfg, recompute=args.recompute)

    print("\n=== representations ===")
    for name, info in made.items():
        if info[0] == "cached":
            print(f"  {name:<10} cached ({info[1]})")
        else:
            print(f"  {name:<10} {info[0]} molecules x {info[1]} descriptors")

    smi_dir = cfg["paths"]["features"] / "alvadesc_inputs"
    print(f"\nalvaDesc inputs written to: {smi_dir.relative_to(cfg['_root'])}")
    for p in sorted(smi_dir.glob("*.smi")):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p.name:<20} {n} molecules")
    print("  -> follow ALVADESC_INSTRUCTIONS.md, then drop CSVs in data/features/")

    print(f"\nRepresentations available now: {available_representations(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
