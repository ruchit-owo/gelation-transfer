"""gelbench.features — molecular representations and an aligned loader.

Three representations (spec §7):
  * rdkit2d  — the full rdkit.Chem.Descriptors set (open).
  * morgan   — Morgan/ECFP4 fingerprints, 2048 bits (open).
  * mordred  — Mordred 2D descriptors (open, optional).
  * alvadesc — commercial 2D descriptors, computed by the user's licensed
               alvaDesc trial. This module EXPORTS per-family .smi inputs and
               LOADS the resulting CSVs (it cannot run alvaDesc itself).

``load_representation`` returns an aligned ``(X, y, family)`` for any
representation and *fails loudly* if a molecule is missing from a feature table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as _fpg

RDLogger.DisableLog("rdApp.*")

OPEN_REPRESENTATIONS = ["rdkit2d", "morgan", "mordred"]


@dataclass
class Representation:
    """Aligned feature block for one representation over the whole dataset."""
    name: str
    X: pd.DataFrame          # index = id, columns = descriptors (dataset order)
    y: np.ndarray            # 0/1 labels
    family: np.ndarray       # family string per row
    ids: np.ndarray          # molecule ids
    smiles: np.ndarray       # canonical SMILES per row


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
def load_dataset(cfg) -> pd.DataFrame:
    path = cfg["paths"]["dataset"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run experiments/01_curate.py first.")
    return pd.read_csv(path)


def _mols(dataset: pd.DataFrame):
    return [(i, s, Chem.MolFromSmiles(s))
            for i, s in zip(dataset["id"], dataset["smiles"])]


# --------------------------------------------------------------------------- #
# Open representations
# --------------------------------------------------------------------------- #
def compute_rdkit2d(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for mid, smi, mol in _mols(dataset):
        if mol is None:
            raise ValueError(f"rdkit2d: unparseable SMILES for id={mid}: {smi}")
        rows[mid] = Descriptors.CalcMolDescriptors(mol)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "id"
    return out


def compute_morgan(dataset: pd.DataFrame, radius: int, nbits: int) -> pd.DataFrame:
    gen = _fpg.GetMorganGenerator(radius=radius, fpSize=nbits)
    cols = [f"m{radius}_{i}" for i in range(nbits)]
    data, index = [], []
    for mid, smi, mol in _mols(dataset):
        if mol is None:
            raise ValueError(f"morgan: unparseable SMILES for id={mid}: {smi}")
        data.append(gen.GetFingerprintAsNumPy(mol))
        index.append(mid)
    out = pd.DataFrame(np.vstack(data), index=index, columns=cols, dtype=np.uint8)
    out.index.name = "id"
    return out


def compute_mordred(dataset: pd.DataFrame):
    """Mordred 2D descriptors, or None if mordred is unavailable."""
    try:
        from mordred import Calculator, descriptors
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"[features] mordred unavailable ({exc}); skipping.")
        return None
    calc = Calculator(descriptors, ignore_3D=True)
    mols, ids = [], []
    for mid, smi, mol in _mols(dataset):
        if mol is None:
            raise ValueError(f"mordred: unparseable SMILES for id={mid}: {smi}")
        mols.append(mol)
        ids.append(mid)
    df = calc.pandas(mols, nproc=1, quiet=True)
    df.index = ids
    df.index.name = "id"
    # Mordred returns error objects for failed descriptors -> coerce to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# alvaDesc — export inputs, load outputs
# --------------------------------------------------------------------------- #
def export_alvadesc_inputs(cfg, dataset: pd.DataFrame) -> list:
    """Write one `.smi` per family (SMILES<TAB>id) for the alvaDesc trial.

    Also writes ALVADESC_INSTRUCTIONS.md with the exact settings to use.
    Returns the list of written .smi paths.
    """
    outdir = cfg["paths"]["features"] / "alvadesc_inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fam, g in dataset.groupby("family"):
        p = outdir / f"{fam}.smi"
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for smi, mid in zip(g["smiles"], g["id"]):
                fh.write(f"{smi}\t{mid}\n")
        written.append(p)

    (outdir / "ALVADESC_INSTRUCTIONS.md").write_text(
        _ALVADESC_INSTRUCTIONS, encoding="utf-8")
    return written


_ALVADESC_INSTRUCTIONS = """\
# Running alvaDesc for this benchmark

For **each** `.smi` file in this folder (`nucleoside.smi`, `tripeptide.smi`,
`peptide_expt.smi`), produce one descriptor CSV.

Each `.smi` line is `SMILES<TAB>id`. Keep the `id` as the molecule **Name** so
the loader can align rows.

## Settings (must match, spec §7)
1. **Import** the `.smi` file (SMILES with the second column as Name/ID).
2. Enable **all 2D descriptor blocks**.
3. **Disable every 3D block** (no conformers are generated).
4. Do **not** filter or scale inside alvaDesc — export raw descriptor values.
5. **Export to CSV** with the molecule **Name (id)** as the first column.
   `;` or tab delimiters are both fine — the loader auto-detects.

## Save the outputs here, named exactly:
    data/features/alvadesc_nucleoside.csv
    data/features/alvadesc_tripeptide.csv
    data/features/alvadesc_peptide_expt.csv

The nucleoside export from the sibling NHGPM reproduction can be reused if it was
produced with the same all-2D / no-3D settings; otherwise re-run it here.

Once the three CSVs are in place, `python run_all.py --from 3` will pick up the
`alvadesc` representation automatically.
"""


def _read_alvadesc_csv(path: Path) -> pd.DataFrame:
    """Read one alvaDesc CSV: auto-detect `;`/tab, index by NAME, numeric only."""
    # Sniff the delimiter from the header line.
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    sep = "\t" if header.count("\t") >= header.count(";") else ";"
    df = pd.read_csv(path, sep=sep)
    # Identify the name/id column and drop the running index column.
    name_col = next((c for c in df.columns if c.strip().upper() in ("NAME", "ID")), None)
    if name_col is None:
        raise ValueError(f"alvaDesc CSV {path.name} has no NAME/ID column "
                         f"(columns start: {list(df.columns)[:4]})")
    df = df.set_index(name_col)
    df = df.drop(columns=[c for c in df.columns if c.strip().upper() in ("NO.", "NO")],
                 errors="ignore")
    df.index.name = "id"
    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_alvadesc(cfg, families: list) -> pd.DataFrame:
    """Concatenate per-family alvaDesc CSVs on their shared descriptor columns."""
    feat_dir = cfg["paths"]["features"]
    parts, missing = [], []
    for fam in families:
        p = feat_dir / f"alvadesc_{fam}.csv"
        if not p.exists():
            missing.append(fam)
            continue
        parts.append(_read_alvadesc_csv(p))
    if missing:
        raise FileNotFoundError(
            "alvaDesc CSV(s) missing for families: " + ", ".join(missing) +
            f"\nExport them with alvaDesc (see {feat_dir}/alvadesc_inputs/"
            "ALVADESC_INSTRUCTIONS.md), then re-run.")
    common = sorted(set.intersection(*[set(p.columns) for p in parts]))
    return pd.concat([p[common] for p in parts], axis=0)


# --------------------------------------------------------------------------- #
# Cache + aligned loader
# --------------------------------------------------------------------------- #
def _cache_path(cfg, name: str) -> Path:
    return cfg["paths"]["features"] / f"{name}.csv"


def compute_and_cache(cfg, *, recompute: bool = False) -> dict:
    """Compute and cache the open representations; export alvaDesc inputs.

    Returns a dict {name: (n_rows, n_cols)} for the representations produced.
    """
    dataset = load_dataset(cfg)
    feat_dir = cfg["paths"]["features"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    made = {}

    builders = {
        "rdkit2d": lambda: compute_rdkit2d(dataset),
        "morgan": lambda: compute_morgan(dataset, cfg["features"]["morgan_radius"],
                                         cfg["features"]["morgan_nbits"]),
    }
    if cfg["features"].get("mordred", True):
        builders["mordred"] = lambda: compute_mordred(dataset)

    for name, fn in builders.items():
        path = _cache_path(cfg, name)
        if path.exists() and not recompute:
            made[name] = ("cached", path.name)
            continue
        tbl = fn()
        if tbl is None:      # mordred unavailable
            continue
        tbl.to_csv(path, lineterminator="\n")
        made[name] = (tbl.shape[0], tbl.shape[1])

    export_alvadesc_inputs(cfg, dataset)
    return made


def load_representation(cfg, name: str) -> Representation:
    """Load one representation aligned to gelation_all.csv order.

    Raises loudly if any molecule in the dataset is missing from the feature
    table (spec §7).
    """
    dataset = load_dataset(cfg)
    ids = dataset["id"].to_numpy()

    if name == "alvadesc":
        tbl = load_alvadesc(cfg, sorted(dataset["family"].unique()))
    else:
        path = _cache_path(cfg, name)
        if not path.exists():
            raise FileNotFoundError(
                f"feature cache {path} missing — run experiments/02_features.py.")
        tbl = pd.read_csv(path, index_col="id")
    tbl.index = tbl.index.astype(str)

    missing = set(ids.astype(str)) - set(tbl.index)
    if missing:
        raise ValueError(
            f"[{name}] {len(missing)} molecules missing from the feature table, "
            f"e.g. {sorted(missing)[:3]}. Feature matrices must cover every "
            f"molecule in gelation_all.csv.")

    X = tbl.loc[ids.astype(str)].copy()
    X.index = ids
    X.index.name = "id"
    return Representation(
        name=name, X=X,
        y=dataset["label"].to_numpy(),
        family=dataset["family"].to_numpy(),
        ids=ids,
        smiles=dataset["smiles"].to_numpy(),
    )


def available_representations(cfg) -> list:
    """Representations whose feature tables are present on disk."""
    out = []
    for name in OPEN_REPRESENTATIONS:
        if _cache_path(cfg, name).exists():
            out.append(name)
    fams = sorted(load_dataset(cfg)["family"].unique())
    if all((cfg["paths"]["features"] / f"alvadesc_{f}.csv").exists() for f in fams):
        out.append("alvadesc")
    return out
