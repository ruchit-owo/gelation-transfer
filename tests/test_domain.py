"""Applicability-domain math tests (spec §11)."""
from __future__ import annotations

import numpy as np
import pytest

from gelbench.domain import (
    hat_leverage, leverage_threshold, novelty_ecfp4, knn_distance,
)


def test_leverage_threshold_is_3p_over_n():
    assert leverage_threshold(5, 50) == pytest.approx(3 * 5 / 50)
    assert leverage_threshold(10, 40, multiplier=3) == pytest.approx(0.75)


def test_training_mean_leverage_equals_p_over_n():
    rng = np.random.default_rng(0)
    n, p = 60, 5
    X = rng.standard_normal((n, p))
    h_train, _ = hat_leverage(X, X[:1])
    # Sum of hat-matrix diagonal = p, so the mean is exactly p/n.
    assert h_train.mean() == pytest.approx(p / n, rel=1e-6)
    assert h_train.sum() == pytest.approx(p, rel=1e-6)


def test_novelty_in_unit_interval_and_zero_for_identical():
    train = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    query = ["CCO",              # identical to a training molecule -> novelty 0
             "c1ccccc1",         # identical -> 0
             "C1CCCCC1CCCCN"]    # different -> > 0
    nov = novelty_ecfp4(train, query)
    assert np.all(nov >= 0.0) and np.all(nov <= 1.0)
    assert nov[0] == pytest.approx(0.0, abs=1e-9)
    assert nov[1] == pytest.approx(0.0, abs=1e-9)
    assert nov[2] > 0.0


def test_novelty_symmetric_bounds_on_disjoint_sets():
    nov = novelty_ecfp4(["CCO"], ["c1ccc2ccccc2c1"])   # ethanol vs naphthalene
    assert 0.0 < nov[0] <= 1.0


def test_knn_distance_zero_when_query_in_training():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    d = knn_distance(X, X[:3], k=1)          # nearest neighbour is itself
    assert np.allclose(d, 0.0, atol=1e-9)
