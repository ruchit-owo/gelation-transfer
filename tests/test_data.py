"""Data-invariant tests (spec §11).

These read the tracked, committed data/processed/gelation_all.csv -- no network.
"""
from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from gelbench import load_config
from gelbench.data import DATASET_COLUMNS


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    cfg = load_config()
    path = cfg["paths"]["dataset"]
    if not path.exists():
        pytest.skip(f"{path} not built yet; run experiments/01_curate.py")
    return pd.read_csv(path)


def test_columns_exact(dataset):
    assert list(dataset.columns) == DATASET_COLUMNS


def test_every_smiles_parses(dataset):
    bad = [s for s in dataset["smiles"] if Chem.MolFromSmiles(s) is None]
    assert not bad, f"{len(bad)} SMILES failed to parse, e.g. {bad[:3]}"


def test_no_duplicate_canonical_smiles(dataset):
    dupes = dataset["smiles"][dataset["smiles"].duplicated()].tolist()
    assert not dupes, f"duplicate canonical SMILES present: {dupes[:3]}"


def test_labels_are_binary(dataset):
    assert set(dataset["label"].unique()) <= {0, 1}


def test_each_family_has_both_classes(dataset):
    for fam, g in dataset.groupby("family"):
        classes = set(g["label"].unique())
        assert classes == {0, 1}, f"family {fam!r} missing a class: {classes}"


def test_families_meet_minimum_size(dataset):
    cfg = load_config()
    counts = dataset["family"].value_counts()
    # Not a hard failure below the flag threshold, but there must be >=3 families
    # and none absurdly tiny (spec requires >=3 chemical families).
    assert len(counts) >= 3, f"need >=3 families, found {list(counts.index)}"
    assert counts.min() >= 10, f"a family is implausibly small: {counts.to_dict()}"


def test_ids_unique(dataset):
    assert dataset["id"].is_unique, "molecule ids must be unique"
