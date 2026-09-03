# Data provenance

Every data source used in this benchmark is recorded here: where it came from,
its licence, what was downloaded, how it was standardised, and (for continuous
scores) the binarisation threshold. Raw downloads live under `data/raw/`
(git-ignored); this file plus `data/raw/manifest.json` are the tracked record.

Each molecule in `data/processed/gelation_all.csv` carries a `source_doi` that
points back to one of the sources below.

---

## Families

<!-- One subsection per family is filled in by experiments/01_curate.py.
     Template:

### <family> — <n> molecules (<n_gel> gel / <n_nongel> non-gel)

- **Source:** <citation>
- **DOI:** <doi>
- **Deposit:** <url / repo>
- **Licence:** <licence + verified date>
- **Downloaded:** <file(s), sha256, date>
- **Labels:** <how gel/non-gel was defined; threshold if continuous>
- **Standardisation:** RDKit largest-fragment + neutralise + canonical isomeric SMILES
- **Dropped:** <n unparseable, n duplicates>
-->

### nucleoside — 71 molecules (38 gel / 33 non-gel)

- **Source:** Li W. et al., Nat. Commun. 15, 2603 (2024)
- **doi:** 10.1038/s41467-024-46866-9
- **deposit:** github.com/leescu/NHGPM
- **Licence:** CC BY 4.0 (article); Apache-2.0 (deposited code/data)
- **Labels:** experimental gel/non-gel (binary as deposited).
- **Dropped:** 0 unparseable, 0 within-family duplicates.

### peptide_expt — 367 molecules (248 gel / 119 non-gel)

- **Source:** Njirjak M. et al., Nat. Mach. Intell. 6, 1487 (2024), experimental set compiled from the self-assembly literature
- **doi_data:** 10.5281/zenodo.13847868
- **Licence:** MIT (data deposit)
- **Labels:** experimental gel/non-gel (binary as deposited).
- **Dropped:** 0 unparseable, 0 within-family duplicates.

### tripeptide — 600 molecules (179 gel / 421 non-gel)

- **Source:** Frederix P.W.J.M. et al., Nat. Chem. 7, 30-37 (2015); AP values redeposited by Njirjak M. et al., Nat. Mach. Intell. 6, 1487 (2024)
- **doi_source:** 10.1038/nchem.2122
- **doi_data:** 10.5281/zenodo.13847868
- **Licence:** MIT (data deposit)
- **Labels:** continuous aggregation propensity (AP) binarised at AP >= 1.5 (sensitivity: [1.4, 1.6]); seeded stratified subsample n=600.
- **Dropped:** 0 unparseable, 0 within-family duplicates.


---

## Standardisation protocol (all families)

1. Parse SMILES with RDKit; drop and log anything unparseable.
2. Keep the largest organic fragment (strip counterions / salts).
3. Neutralise charges where chemically sensible.
4. Canonicalise to isomeric SMILES — this canonical string is the dedup key.
5. Deduplicate within and across families. Cross-family collisions are assigned
   to a single family and the collision count is recorded.

## Cross-family SMILES collisions

1 molecule(s) appeared in more than one family and were assigned to the earliest family in config order; rows dropped per family: {'peptide_expt': 1}.

---

## alvaDesc feature exports (commercial; NOT tracked in git)

The alvaDesc descriptor matrices are large (~34 MB total) and require a
commercial licence, so they are **git-ignored** rather than committed. They are
**not regenerable by the open pipeline** — to recreate them, run alvaDesc on the
tracked inputs `data/features/alvadesc_inputs/*.smi` following
`data/features/alvadesc_inputs/ALVADESC_INSTRUCTIONS.md` (all 2D descriptor
blocks, no 3D, no variable reduction), and save each export into `data/features/`
under the name below. The loader accepts `.txt` (alvaDesc's Tabbed-text export)
or `.csv`. Everything downstream (within-family, transfer, AD, figures) then
regenerates from `run_all.py --from 3`.

Version used: **alvaDesc 3.0.16**; 4215 2D descriptors; molecule Name = the `id`.

| file | molecules | size | sha256 |
|---|---|---|---|
| `alvadesc_nucleoside.csv` | 71 | 2.3 MB | `0cff2117e8971dc5ac0ac6cfc155ba0cdaf7d2d069b4b9d5207869c75685e54e` |
| `alvadesc_tripeptide.txt` | 600 | 19.1 MB | `f158afe794825bdf65707091734d13565e3ad39bf87fc93090e516ce51d43797` |
| `alvadesc_peptide_expt.txt` | 367 | 11.8 MB | `6e823f549c9567595776b68df03693647217d59b8ffcdf20a60b5ae3b8c0fa40` |

(alvaDesc output is not guaranteed bit-identical across versions/machines; these
checksums record the exact files used to produce the committed results.)
