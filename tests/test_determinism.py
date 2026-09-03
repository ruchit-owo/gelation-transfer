"""Determinism (spec §11): the same seed reproduces identical result tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gelbench import load_config
from gelbench.data import _stratified_subsample
from gelbench.evaluate import within_family_scores, pooled_nested_cv
from gelbench.features import Representation


def _toy_rep(seed=0, n=60, p=20):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    # a couple of informative features + noise, so AUC is non-degenerate
    X = rng.standard_normal((n, p))
    X[:, 0] += y * 1.5
    cols = [f"f{i}" for i in range(p)]
    ids = np.array([f"m{i}" for i in range(n)])
    return Representation(
        name="toy",
        X=pd.DataFrame(X, index=ids, columns=cols),
        y=y, family=np.array(["toy"] * n), ids=ids,
        smiles=np.array(["C"] * n),
    )


def test_pooled_nested_cv_is_deterministic():
    cfg = load_config()
    rep = _toy_rep()
    a = pooled_nested_cv(rep.X, rep.y, "logreg", cfg)
    b = pooled_nested_cv(rep.X, rep.y, "logreg", cfg)
    assert np.array_equal(a.prob, b.prob)


def test_within_family_scores_row_is_identical():
    cfg = load_config()
    rep = _toy_rep()
    r1 = within_family_scores(rep, "toy", "logreg", cfg)
    r2 = within_family_scores(rep, "toy", "logreg", cfg)
    assert r1 == r2                       # includes accuracy, auc, and both CIs


def test_stratified_subsample_is_deterministic():
    labels = np.array([0] * 80 + [1] * 20)
    i1 = _stratified_subsample(labels, 30, np.random.default_rng(20260903))
    i2 = _stratified_subsample(labels, 30, np.random.default_rng(20260903))
    assert np.array_equal(i1, i2)
