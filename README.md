# gelation-transfer

Cross-chemistry gelation transfer benchmark for low-molecular-weight gelators
(LMWGs).

> **Status:** under construction. The headline finding will go here once the
> transfer and applicability-domain experiments have run. See
> [`report/REPORT.md`](report/REPORT.md) for the full write-up.

## What this is

Most published gelation predictors are trained and tested inside a single
chemical family. This benchmark measures how far such a model generalises to a
family it never saw, and whether the off-family accuracy drop is explained by an
**applicability-domain distance** (leverage + structural novelty).

Standard methods only — scikit-learn, RDKit, nested cross-validation, published
applicability-domain measures. The contribution is an honest, reproducible
benchmark, not a new algorithm.

## Reproduce

```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
python run_all.py             # full pipeline, or: python run_all.py --from 3
pytest                        # data invariants, leakage guard, domain math, determinism
```

Raw downloads land in `data/raw/` (git-ignored); everything downstream —
`data/processed/gelation_all.csv`, cached features, results tables, and figures —
is tracked and regenerable. Data provenance and licences are recorded in
[`PROVENANCE.md`](PROVENANCE.md).

## Layout

```
src/gelbench/     data · features · evaluate · domain · transfer · plots
experiments/      01_curate → 06_figures  (run by run_all.py)
data/processed/   gelation_all.csv  (id, smiles, label, family, source_doi)
data/features/    cached descriptor matrices
results/ tables/ figures/   regenerable outputs
tests/            pytest suite incl. the leakage guard
report/REPORT.md  3–5 page write-up
```
