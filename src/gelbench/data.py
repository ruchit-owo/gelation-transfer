"""gelbench.data — download, standardise, deduplicate; build gelation_all.csv.

Every data-provenance decision lives here and is echoed to PROVENANCE.md and
data/raw/manifest.json so the dataset is regenerable and auditable.

Curation protocol (identical for every family, spec §6):
  1. Parse SMILES with RDKit; drop and log anything unparseable.
  2. Keep the largest organic fragment (strip counter-ions / salts).
  3. Neutralise charges where chemically sensible.
  4. Canonicalise to isomeric SMILES -- this string is the dedup key.
  5. Deduplicate within and across families; cross-family collisions are
     assigned to ONE family and counted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # silence expected parse warnings; we log our own

# Final dataset columns (spec §5.1).
DATASET_COLUMNS = ["id", "smiles", "label", "family", "source_doi"]


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def _standardize_mol(mol, cfg_std: dict):
    """Apply the shared standardisation pipeline to an RDKit mol.

    Returns a canonical isomeric SMILES string, or ``None`` if anything fails.
    """
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        if cfg_std.get("largest_fragment", True):
            mol = _LARGEST_FRAGMENT.choose(mol)
        if cfg_std.get("neutralize", True):
            mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(
        mol,
        isomericSmiles=cfg_std.get("isomeric", True),
        canonical=cfg_std.get("canonical", True),
    )


def standardize_smiles(smiles: str, cfg_std: dict):
    """Standardise a SMILES string -> canonical isomeric SMILES, or None."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return _standardize_mol(Chem.MolFromSmiles(smiles), cfg_std)


def sequence_to_smiles(seq: str, cfg_std: dict):
    """Build a linear peptide from a one-letter sequence, then standardise it.

    Uses RDKit ``MolFromSequence`` (standard L-amino acids, free N- and
    C-termini). Returns canonical isomeric SMILES, or None for unknown residues.
    """
    if not isinstance(seq, str) or not seq.strip():
        return None
    return _standardize_mol(Chem.MolFromSequence(seq.strip()), cfg_std)


# --------------------------------------------------------------------------- #
# Download / cache with a tracked manifest
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "gelbench/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out)


def acquire(cfg, family: str, url_or_path: str, dest_name: str,
            *, is_url: bool, manifest: list) -> Path:
    """Fetch a URL (cached) or copy a local file into data/raw/<family>/.

    Records source, sha256, bytes, and retrieval time in ``manifest``. If the
    cached file already exists, it is reused (no network / re-copy).
    """
    raw = cfg["paths"]["raw"] / family
    dest = raw / dest_name
    if not dest.exists():
        if is_url:
            _fetch(url_or_path, dest)
        else:
            src = Path(url_or_path)
            if not src.exists():
                raise FileNotFoundError(
                    f"[{family}] expected local source not found: {src}\n"
                    "Adjust sources.<family>.repo_path in config.yaml."
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    manifest.append({
        "family": family,
        "source": url_or_path,
        "kind": "url" if is_url else "local_copy",
        "path": str(dest.relative_to(cfg["_root"])),
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "retrieved_utc": _now(),
    })
    return dest


# --------------------------------------------------------------------------- #
# Per-family loaders -> raw frames (id, smiles_raw|sequence, label, family, doi)
# --------------------------------------------------------------------------- #
def load_nucleoside(cfg, src: dict, manifest: list) -> pd.DataFrame:
    repo = (cfg["_root"] / src["repo_path"]).resolve()
    csv = acquire(cfg, "nucleoside", str(repo / src["smiles_csv"]),
                  "Smiles_data.csv", is_url=False, manifest=manifest)
    df = pd.read_csv(csv)
    out = pd.DataFrame({
        "id": df[src["id_col"]].astype(str),
        "smiles_raw": df[src["smiles_col"]].astype(str),
        "label": df[src["label_col"]].astype(int),
        "family": "nucleoside",
        "source_doi": src["doi"],
        "kind": "smiles",
    })
    return out


def load_tripeptide(cfg, src: dict, manifest: list, rng: np.random.Generator) -> pd.DataFrame:
    npy = acquire(cfg, "tripeptide", src["url"], "tripeptides_AP.npy",
                  is_url=True, manifest=manifest)
    ap = np.load(npy, allow_pickle=True).item()          # {seq: AP}
    seqs = np.array(list(ap.keys()))
    scores = np.array([ap[s] for s in seqs], dtype=float)
    thr = float(src["ap_threshold"])
    labels = (scores >= thr).astype(int)

    # Seeded, label-stratified subsample to keep families comparable in size
    # and the alvaDesc/Mordred workload tractable (full set stays in data/raw).
    n_sub = int(src.get("subsample_n", len(seqs)))
    if n_sub < len(seqs):
        idx = _stratified_subsample(labels, n_sub, rng)
    else:
        idx = np.arange(len(seqs))
    idx = np.sort(idx)

    out = pd.DataFrame({
        "id": ["TRI_" + seqs[i] for i in idx],
        "sequence": seqs[idx],
        "ap_score": scores[idx],
        "label": labels[idx],
        "family": "tripeptide",
        "source_doi": src["doi_source"],
        "kind": "sequence",
    })
    return out


def load_peptide_expt(cfg, src: dict, manifest: list) -> pd.DataFrame:
    csv = acquire(cfg, "peptide_expt", src["url"], "data_SA.csv",
                  is_url=True, manifest=manifest)
    df = pd.read_csv(csv, sep=src.get("sep", ","))
    out = pd.DataFrame({
        "id": ["PEP_" + s for s in df[src["seq_col"]].astype(str)],
        "sequence": df[src["seq_col"]].astype(str),
        "label": df[src["label_col"]].astype(int),
        "family": "peptide_expt",
        "source_doi": src.get("doi_data", ""),
        "kind": "sequence",
    })
    return out


_LOADERS = {
    "nucleoside": load_nucleoside,
    "tripeptide": load_tripeptide,
    "peptide_expt": load_peptide_expt,
}


def _stratified_subsample(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices of a size-n subsample preserving the class ratio."""
    idx_all = []
    classes, counts = np.unique(labels, return_counts=True)
    frac = n / len(labels)
    for c in classes:
        ci = np.where(labels == c)[0]
        take = int(round(len(ci) * frac))
        take = max(1, min(take, len(ci)))
        idx_all.append(rng.choice(ci, size=take, replace=False))
    return np.concatenate(idx_all)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_dataset(cfg) -> dict:
    """Run the full curation and write gelation_all.csv, manifest, PROVENANCE.

    Returns a summary dict (per-family counts, drops, collisions) so the driver
    script and tests can assert on it.
    """
    cfg_std = cfg["standardize"]
    rng = np.random.default_rng(cfg["seed"])
    manifest: list = []
    frames = []
    drops = {}

    for family, src in cfg["sources"].items():
        loader = _LOADERS[family]
        raw = (loader(cfg, src, manifest, rng)
               if family == "tripeptide" else loader(cfg, src, manifest))

        # Standardise -> canonical SMILES (drop + log failures).
        if raw["kind"].iloc[0] == "sequence":
            smi = raw["sequence"].map(lambda s: sequence_to_smiles(s, cfg_std))
        else:
            smi = raw["smiles_raw"].map(lambda s: standardize_smiles(s, cfg_std))
        raw = raw.assign(smiles=smi)
        n_before = len(raw)
        raw = raw[raw["smiles"].notna()].copy()
        drops[family] = {"unparseable": n_before - len(raw), "n_parsed": len(raw)}

        # Deduplicate WITHIN family by canonical SMILES (keep first).
        n_pre = len(raw)
        raw = raw.drop_duplicates(subset="smiles", keep="first").copy()
        drops[family]["intra_dupes"] = n_pre - len(raw)
        frames.append(raw)

    alldf = pd.concat(frames, ignore_index=True)

    # Cross-family collisions: same canonical SMILES in >1 family. Assign to the
    # family that appears first in config order (nucleoside > tripeptide > ...),
    # a stable, documented rule. Count how many rows were dropped as collisions.
    family_order = list(cfg["sources"].keys())
    alldf["_forder"] = alldf["family"].map(family_order.index)
    alldf = alldf.sort_values(["smiles", "_forder"], kind="stable")
    dup_mask = alldf.duplicated(subset="smiles", keep="first")
    collisions = (alldf[dup_mask]
                  .groupby("family").size().to_dict())
    n_collisions = int(dup_mask.sum())
    alldf = alldf[~dup_mask].drop(columns="_forder")

    # Final table, stable order: family (config order) then id.
    alldf["_forder"] = alldf["family"].map(family_order.index)
    alldf = (alldf.sort_values(["_forder", "id"], kind="stable")
             .reset_index(drop=True))
    final = alldf[DATASET_COLUMNS].copy()

    # --- write outputs -----------------------------------------------------
    out_csv = cfg["paths"]["dataset"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False, lineterminator="\n")

    # Keep the tripeptide AP scores alongside (for threshold-sensitivity reruns).
    ap_side = alldf.loc[alldf["family"] == "tripeptide",
                        ["id", "ap_score", "sequence"]]
    ap_side.to_csv(cfg["paths"]["processed"] / "tripeptide_ap_scores.csv",
                   index=False, lineterminator="\n")

    (cfg["paths"]["raw"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "n_total": len(final),
        "families": {},
        "drops": drops,
        "cross_family_collisions": {"n": n_collisions, "by_family_dropped": collisions},
    }
    for fam, g in final.groupby("family"):
        summary["families"][fam] = {
            "n": int(len(g)),
            "n_gel": int((g["label"] == 1).sum()),
            "n_nongel": int((g["label"] == 0).sum()),
            "majority_baseline": float(max((g["label"] == 1).mean(),
                                           (g["label"] == 0).mean())),
        }
    _write_provenance(cfg, summary)
    return summary


def _write_provenance(cfg, summary: dict) -> None:
    """Fill the Families and collisions sections of PROVENANCE.md."""
    src = cfg["sources"]
    lines = []
    for fam, info in summary["families"].items():
        s = src[fam]
        d = summary["drops"][fam]
        lines.append(f"### {fam} — {info['n']} molecules "
                     f"({info['n_gel']} gel / {info['n_nongel']} non-gel)\n")
        lines.append(f"- **Source:** {s.get('citation', '')}")
        for k in ("doi", "doi_source", "doi_data", "deposit"):
            if s.get(k):
                lines.append(f"- **{k}:** {s[k]}")
        lines.append(f"- **Licence:** {s.get('licence', '')}")
        if "ap_threshold" in s:
            lines.append(f"- **Labels:** continuous aggregation propensity (AP) "
                         f"binarised at AP >= {s['ap_threshold']} "
                         f"(sensitivity: {s['ap_threshold_sensitivity']}); "
                         f"seeded stratified subsample n={s.get('subsample_n')}.")
        else:
            lines.append("- **Labels:** experimental gel/non-gel (binary as deposited).")
        lines.append(f"- **Dropped:** {d['unparseable']} unparseable, "
                     f"{d['intra_dupes']} within-family duplicates.")
        lines.append("")
    fam_block = "\n".join(lines)

    coll = summary["cross_family_collisions"]
    coll_block = (f"{coll['n']} molecule(s) appeared in more than one family and "
                  f"were assigned to the earliest family in config order; rows "
                  f"dropped per family: {coll['by_family_dropped'] or 'none'}.")

    text = (cfg["paths"]["provenance"]).read_text(encoding="utf-8")
    text = text.replace("_Populated during curation (Phase 2)._", fam_block)
    text = text.replace("_Populated during curation._", coll_block)
    (cfg["paths"]["provenance"]).write_text(text, encoding="utf-8")
