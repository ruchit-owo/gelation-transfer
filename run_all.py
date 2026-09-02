#!/usr/bin/env python
"""Reproduce the entire benchmark end to end.

    python run_all.py            # run every stage in order
    python run_all.py --from 3   # resume from stage 3 (features already cached)

Each stage is an idempotent script under experiments/. Stages read cached
outputs from earlier stages, so re-running is cheap once features exist. No
stage writes wall-clock time into a tracked results file (spec §12).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

# (stage number, script, one-line description)
STAGES = [
    (1, "01_curate.py", "Download, standardise, dedupe -> gelation_all.csv"),
    (2, "02_features.py", "RDKit 2D, Morgan/ECFP4, Mordred, alvaDesc (RUN EARLY)"),
    (3, "03_within_family.py", "Leak-free within-family baselines"),
    (4, "04_transfer.py", "Pairwise transfer matrix + leave-one-family-out"),
    (5, "05_domain_analysis.py", "Applicability domain vs performance drop"),
    (6, "06_figures.py", "Figures + tables for the report"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="first stage to run (default 1)")
    ap.add_argument("--only", type=int, default=None,
                    help="run only this stage")
    args = ap.parse_args()

    for num, script, desc in STAGES:
        if args.only is not None and num != args.only:
            continue
        if num < args.start:
            continue
        print(f"\n{'=' * 78}\n[stage {num}] {script} — {desc}\n{'=' * 78}", flush=True)
        rc = subprocess.run([sys.executable, str(EXPERIMENTS / script)]).returncode
        if rc != 0:
            print(f"\n[stage {num}] FAILED (exit {rc}). Stopping.", file=sys.stderr)
            return rc
    print("\nAll requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
