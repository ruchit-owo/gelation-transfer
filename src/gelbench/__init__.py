"""gelbench — cross-chemistry gelation transfer benchmark.

An undergraduate research benchmark that measures how well machine-learning
gelation predictors for low-molecular-weight gelators transfer *across* chemical
families, and whether the off-family performance drop is explained by an
applicability-domain distance.

Standard, well-established methods only (scikit-learn, RDKit, nested CV,
published applicability-domain measures). The contribution is an honest,
reproducible benchmark and a clear finding — not a new algorithm.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml

# Repository root = two levels up from this file (src/gelbench/__init__.py).
ROOT = Path(__file__).resolve().parents[2]

__version__ = "0.1.0"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml once and resolve every path to an absolute Path.

    Paths in the returned ``config['paths']`` are absolute ``Path`` objects
    rooted at the repository, so scripts can be run from any working directory.
    """
    cfg_path = Path(path) if path is not None else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (ROOT / v) for k, v in cfg["paths"].items()}
    cfg["_root"] = ROOT
    return cfg
