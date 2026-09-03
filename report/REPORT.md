<!--
  DRAFT REPORT. Numbers in the results sections are filled from the tracked
  results CSVs after run_all.py completes. Sections written before the full run
  (motivation, data, method) are stable; results/limitations get final numbers.
-->
# Cross-chemistry gelation transfer benchmark

**Headline (to be finalised):** Machine-learning gelation predictors that look
strong *within* a chemical family lose essentially all of their skill when
applied to a family they were not trained on, and the molecules they get wrong
off-family are the ones that sit outside the training set's applicability domain.

---

## 1. Motivation

Low-molecular-weight gelators (LMWGs) self-assemble into gels that are useful for
drug delivery, tissue engineering, and sensing. Predicting whether a candidate
will gel — from its 2D structure alone — would save a great deal of trial-and-
error synthesis. Several published models do this well, but almost always
*within a single chemical family* (peptides, or nucleosides, or one capped-
dipeptide series). For a materials scientist screening a genuinely new scaffold,
the question that actually matters is different: **how far does such a model
generalise to a family it has never seen, and can it tell me when not to trust
its own prediction?** This benchmark measures that transfer gap and tests whether
a standard applicability-domain (AD) distance explains it.

## 2. Data

Three families with deposited structures and binary gel / non-gel (or
self-assemble / not) labels, standardised identically (RDKit: largest fragment,
neutralise, canonical isomeric SMILES) and deduplicated within and across
families. Full provenance, licences, and the AP threshold are in `PROVENANCE.md`.

| family | n | gel / non-gel | label origin | chemistry |
|---|---|---|---|---|
| nucleoside | 71 | 38 / 33 | experimental hydrogelation | ribose + nucleobase |
| tripeptide | 600 | 179 / 421 | coarse-grained MD aggregation propensity, AP ≥ 1.5 | bare tripeptides |
| peptide_expt | 367 | 248 / 119 | experimental self-assembly | variable-length peptides |

Sources: nucleosides — Li et al., *Nat. Commun.* **15**, 2603 (2024); tripeptide
AP — Frederix et al., *Nat. Chem.* **7**, 30 (2015), values redeposited by
Njirjak et al., *Nat. Mach. Intell.* **6**, 1487 (2024); experimental peptides —
Njirjak et al. (same deposit).

**Honest scope note.** These three datasets span two core chemistries
(nucleosides vs peptides); the two peptide sets differ in label origin
(computational AP vs experimental) and length. The strongest *cross-chemistry*
contrast is therefore nucleoside ↔ peptide, and the tripeptide ↔ peptide_expt
pair is a within-chemistry / cross-dataset contrast. This is stated plainly
rather than dressed up as four unrelated chemistries, and it is in fact useful:
it gives a *range* of AD distances against which to test RQ2. The tripeptide
family is a seeded, label-stratified subsample of the full 8000 (kept comparable
in size to the others and to keep the alvaDesc/Mordred workload tractable).

## 3. Method — the leak-free protocol

Every data-dependent step (inf-cleaning, median imputation, variance filter,
univariate `SelectKBest`, standardisation, and hyperparameter choice) lives
inside a scikit-learn `Pipeline` wrapped by `GridSearchCV`, so it is fit **only**
on the training rows of each fold. This makes leak-freedom structural rather than
a matter of discipline; a dedicated leakage-guard test (`tests/test_leakage.py`)
demonstrates that selecting features on all data inflates the score while the
in-fold pipeline stays at chance on pure noise.

- **Models:** logistic regression (primary), random forest, HistGradientBoosting.
- **Within-family baseline:** repeated stratified 5×10 nested CV; pooled
  out-of-fold accuracy and AUC with a bootstrap CI over *molecules*; the
  majority-class baseline is reported next to every score.
- **Transfer (RQ1):** train on all of family A, test on all of family B, every
  ordered pair; plus leave-one-family-out (train on the union of all but one
  family, test on the held-out family). Splits are always *by family*.
- **Applicability domain (RQ2):** per test molecule, leverage (Williams
  h\* = 3p/n) and kNN distance in standardised RDKit-2D space, and structural
  novelty = 1 − max ECFP4 Tanimoto to the training set. Correlated with
  off-family correctness (molecule level) and with the family-level AUC drop.
- **Descriptors (RQ3):** the whole pipeline is run for RDKit-2D, Morgan/ECFP4,
  Mordred, and (when available) alvaDesc.

## 4. Within-family results (RQ baseline)

<!-- FILL: table_within_family, fig_within_family -->
_Pending full run. rdkit2d / logreg (final): nucleoside AUC 0.79 (base 0.54),
peptide_expt 0.91 (base 0.68), tripeptide 0.97 (base 0.70)._

## 5. Transfer results (RQ1)

<!-- FILL: transfer_matrix, leave_one_family_out, fig_transfer_* -->
_Pending full run. rdkit2d / logreg (final): mean off-diagonal AUC ≈ 0.47;
leave-one-family-out AUC nucleoside 0.52, peptide_expt 0.37, tripeptide 0.66 —
i.e. off-family performance falls to chance or below._

## 6. Does applicability domain explain the drop? (RQ2)

<!-- FILL: ad_vs_performance, ad_correlation.json, fig_headline, fig_molecule_ad -->
_Pending full run. Molecule level (rdkit2d / logreg, final): off-family molecules
predicted **wrong** have far higher leverage (median 2.54 vs 0.83,
Mann–Whitney p ≈ 1e-11) and kNN distance (10.5 vs 7.3, p ≈ 4e-11) than those
predicted correctly; ECFP4 novelty separates them only weakly (p ≈ 0.07).
Family-level Spearman is underpowered at three families._

## 7. Descriptor comparison (RQ3)

<!-- FILL: table_transfer_summary, fig_descriptor_comparison -->
_Pending full run (RDKit-2D, Morgan, Mordred; alvaDesc when the trial CSVs land)._

## 8. Limitations

- **Three families, two chemistries.** Transfer numbers across so few families
  are noisy; confidence intervals are wide and small differences are not
  interpreted. No claim of a descriptor ranking is made beyond what the
  intervals support.
- **One computational family.** The tripeptide labels are simulated aggregation
  propensities, not wet-lab gelation; the AP threshold (1.5) is checked for
  sensitivity but remains a modelling choice.
- **Subsampling.** The tripeptide family is a subsample; the full-set behaviour
  is expected to be similar but is not exhaustively rerun here.
- **2D only.** No conformer-dependent descriptors, by design.

## 9. What a materials scientist should take away

<!-- FILL after results -->
_Pending: a within-family AUC is not evidence that a gelation model will work on
your new scaffold. Before trusting an off-family prediction, check the AD
distance (leverage / nearest-neighbour distance): a molecule far outside the
training domain is where the model is most likely wrong._
