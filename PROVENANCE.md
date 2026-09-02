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

_Populated during curation (Phase 2)._

---

## Standardisation protocol (all families)

1. Parse SMILES with RDKit; drop and log anything unparseable.
2. Keep the largest organic fragment (strip counterions / salts).
3. Neutralise charges where chemically sensible.
4. Canonicalise to isomeric SMILES — this canonical string is the dedup key.
5. Deduplicate within and across families. Cross-family collisions are assigned
   to a single family and the collision count is recorded.

## Cross-family SMILES collisions

_Populated during curation._
