"""Feature-loader tests: alignment and fail-loud on missing molecules (spec §7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gelbench import load_config
from gelbench.features import (
    load_dataset, load_representation, compute_morgan, _cache_path,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _require_cache(cfg, name):
    if not _cache_path(cfg, name).exists():
        pytest.skip(f"{name} cache absent; run experiments/02_features.py")


def test_loader_aligns_to_dataset_order(cfg):
    _require_cache(cfg, "rdkit2d")
    ds = load_dataset(cfg)
    rep = load_representation(cfg, "rdkit2d")
    assert list(rep.X.index) == list(ds["id"])          # exact order
    assert np.array_equal(rep.y, ds["label"].to_numpy())
    assert len(rep.X) == len(ds)


def test_morgan_is_binary_and_right_width(cfg):
    ds = load_dataset(cfg).head(20)
    X = compute_morgan(ds, cfg["features"]["morgan_radius"],
                       cfg["features"]["morgan_nbits"])
    assert X.shape == (20, cfg["features"]["morgan_nbits"])
    assert set(np.unique(X.to_numpy())) <= {0, 1}


def test_loader_fails_loudly_on_missing_molecule(cfg, tmp_path, monkeypatch):
    _require_cache(cfg, "rdkit2d")
    # Build a truncated cache missing some ids, point the loader at it.
    full = pd.read_csv(_cache_path(cfg, "rdkit2d"), index_col="id")
    truncated = full.iloc[5:].copy()          # drop first 5 molecules
    bad = tmp_path / "rdkit2d.csv"
    truncated.to_csv(bad)

    import gelbench.features as F
    monkeypatch.setattr(F, "_cache_path", lambda c, n: bad)
    with pytest.raises(ValueError, match="missing from the feature table"):
        F.load_representation(cfg, "rdkit2d")


def test_alvadesc_requested_but_absent_fails_loudly(cfg):
    with pytest.raises(FileNotFoundError, match="alvaDesc CSV"):
        load_representation(cfg, "alvadesc")
