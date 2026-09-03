"""gelbench.domain — applicability-domain distances (spec §8, RQ2).

Two published AD measures, plus an optional kNN distance:

  * **Leverage** (hat value) in the standardised descriptor space. For a query
    row x and training design matrix X, h = x (XᵀX)⁻¹ xᵀ. The Williams warning
    threshold is h* = 3p/n (p descriptors, n training molecules); the mean
    training leverage equals p/n exactly (trace of the hat matrix = p).
  * **Structural novelty** = 1 − max ECFP4 Tanimoto to the training set, in
    [0, 1]; a molecule identical to a training molecule has novelty 0.
  * **kNN distance** = mean Euclidean distance to the k nearest training points
    in the standardised descriptor space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as _fpg
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Leverage
# --------------------------------------------------------------------------- #
def hat_leverage(X_train: np.ndarray, X_query: np.ndarray):
    """Return (train_leverage, query_leverage) hat values.

    Assumes columns are standardised and p < n (full column rank). Uses the
    Moore-Penrose pseudo-inverse, which equals the true inverse when full rank,
    so the training leverages still sum to p.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    core = np.linalg.pinv(X_train.T @ X_train)          # (p, p)
    h_train = np.einsum("ij,jk,ik->i", X_train, core, X_train)
    h_query = np.einsum("ij,jk,ik->i", X_query, core, X_query)
    return h_train, h_query


def leverage_threshold(p: int, n: int, multiplier: int = 3) -> float:
    """Williams leverage warning threshold h* = multiplier * p / n."""
    return float(multiplier) * p / n


# --------------------------------------------------------------------------- #
# ECFP4 structural novelty
# --------------------------------------------------------------------------- #
def _ecfp4(smiles, radius: int, nbits: int):
    gen = _fpg.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"novelty: unparseable SMILES {smi!r}")
        fps.append(gen.GetFingerprint(mol))
    return fps


def novelty_ecfp4(train_smiles, query_smiles, radius: int = 2, nbits: int = 2048):
    """1 - max ECFP4 Tanimoto of each query to the training set (in [0, 1])."""
    train_fps = _ecfp4(train_smiles, radius, nbits)
    query_fps = _ecfp4(query_smiles, radius, nbits)
    out = np.empty(len(query_fps))
    for i, q in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(q, train_fps)
        out[i] = 1.0 - (max(sims) if sims else 0.0)
    return out


# --------------------------------------------------------------------------- #
# kNN distance
# --------------------------------------------------------------------------- #
def knn_distance(X_train: np.ndarray, X_query: np.ndarray, k: int = 5):
    """Mean Euclidean distance from each query to its k nearest training points."""
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    k = min(k, len(X_train))
    out = np.empty(len(X_query))
    for i, x in enumerate(X_query):
        d = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        out[i] = np.sort(d)[:k].mean()
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _prep_descriptor_space(X_train_desc: pd.DataFrame, X_query_desc: pd.DataFrame):
    """Standardise on train, drop zero-variance, PCA-reduce if p >= n_train.

    Returns (X_train_std, X_query_std, p_used).
    """
    vt = VarianceThreshold(0.0)
    Xtr = vt.fit_transform(np.nan_to_num(X_train_desc.to_numpy(dtype=float),
                                         nan=0.0, posinf=0.0, neginf=0.0))
    Xte = vt.transform(np.nan_to_num(X_query_desc.to_numpy(dtype=float),
                                     nan=0.0, posinf=0.0, neginf=0.0))
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    n, p = Xtr.shape
    if p >= n:  # keep the design matrix full-rank so leverage is well-defined
        n_comp = max(1, min(p, n - 1))
        pca = PCA(n_components=n_comp, random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    return Xtr, Xte, Xtr.shape[1]


def applicability_domain(desc_df: pd.DataFrame, smiles: np.ndarray,
                         train_mask: np.ndarray, query_mask: np.ndarray,
                         ids: np.ndarray, cfg) -> pd.DataFrame:
    """Per-query-molecule AD distances relative to a training set.

    ``desc_df`` is a descriptor table (e.g. rdkit2d) indexed like the dataset;
    ``train_mask``/``query_mask`` select rows. Returns one row per query molecule
    with leverage (+ flag), novelty, and kNN distance.
    """
    Xtr_desc = desc_df.loc[train_mask]
    Xq_desc = desc_df.loc[query_mask]
    Xtr, Xq, p_used = _prep_descriptor_space(Xtr_desc, Xq_desc)
    n_train = len(Xtr)

    h_train, h_query = hat_leverage(Xtr, Xq)
    h_star = leverage_threshold(p_used, n_train, cfg["domain"]["leverage_multiplier"])

    nov = novelty_ecfp4(smiles[train_mask], smiles[query_mask],
                        radius=cfg["domain"]["ecfp_radius"],
                        nbits=cfg["domain"]["ecfp_nbits"])
    knn = knn_distance(Xtr, Xq, k=5)

    return pd.DataFrame({
        "id": ids[query_mask],
        "leverage": h_query,
        "leverage_star": h_star,
        "leverage_flag": (h_query > h_star).astype(int),
        "novelty": nov,
        "knn_distance": knn,
        "p_used": p_used,
        "n_train": n_train,
    })
