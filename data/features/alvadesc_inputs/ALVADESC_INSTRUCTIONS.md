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
