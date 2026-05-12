# Mock data generation

The repository includes a lightweight mock-data workflow for exercising the dark-sirens pipeline end to end.
It is intended for testing data plumbing and sampler smoke tests, not for production-grade observing-scenario studies.

## Generator

Run:

```bash
python scripts/mock_data/generate_mock_data.py --outdir data/mock_dark_sirens
```

By default the generator uses a realistic local galaxy-density normalization,
`--n0 1e-3` Mpc^-3, and a low-redshift generation range, `--zmax 0.08`, so the
fixture remains lightweight.  The generated survey hyperparameters (`log10n0`,
`z50`, `w`, and `delta`) are validated against the default inference prior
bounds before any files are written.

Mock GW posterior widths can be controlled with fractional PE-uncertainty
arguments, for example:

```bash
python scripts/mock_data/generate_mock_data.py \
  --outdir data/mock_dark_sirens \
  --dL-fractional-uncertainty 0.20 \
  --m1det-fractional-uncertainty 0.08 \
  --m2det-fractional-uncertainty 0.10 \
  --sky-uncertainty-deg 5.0
```

If `--dL-fractional-uncertainty` or `--sky-uncertainty-deg` is omitted, that
width falls back to the SNR-scaled heuristic.

The generator writes files that can be consumed directly by `darksirens_inference`:

| File | Purpose |
| --- | --- |
| `mock_galaxy_catalog_complete.h5` | Complete galaxy catalog before EM incompleteness. |
| `mock_survey_raw.h5` | Raw survey-table format accepted by `darksirens_pixelate`. |
| `catalog_pixelated_nside_<nside>.h5` | Pixelated survey catalog accepted by `--survey_path`. |
| `mock_gw_events.h5` | Mock per-event GW posterior samples accepted by `--gw_path`. |
| `mock_gw_selection.h5` | Mock detected injection/selection samples accepted by `--gwselection_path`. |

The simulation is intentionally simple:

1. Draw galaxies isotropically on the sky and uniformly in comoving volume.
2. Apply an EM survey footprint, an apparent-magnitude cut, a redshift limit, and a smooth redshift-completeness curve.
3. Draw GW hosts from the complete pre-selection catalog.
4. Draw binary masses/spins from `powerlaw+peak_shared_beta_spin`: a power-law plus Gaussian peak mass model with one shared mass-ratio beta and one shared truncated-Gaussian `chi_eff` model.
5. Apply a semi-analytic network-SNR threshold to decide GW detectability.
6. Write posterior samples around detected truth values and detected injection samples with `p_draw` weights.

## Smoke-test shell script

Run the default ingestibility smoke test with:

```bash
bash scripts/mock_data/run_mock_data_test.sh
```

The script creates a small data set under `data/mock_dark_sirens_test` using `N0=1e-3` Mpc^-3 and calls `darksirens.inference.data.load_all_data` to verify that the generated HDF5 products are readable by the inference pipeline.

To also launch a tiny optional sampler run, use:

```bash
RUN_INFERENCE=1 bash scripts/mock_data/run_mock_data_test.sh
```

You can override the mock size without editing the script, for example:

```bash
NOBS=5 NSAMP=256 NDRAW=50000 NSIDE=16 bash scripts/mock_data/run_mock_data_test.sh
```

The optional inference run fixes the survey hyperparameters to the generated
scenario via `--fixed_parameter_values`, including `log10n0 = -3`.
