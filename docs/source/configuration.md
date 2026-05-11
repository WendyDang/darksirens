# Configuration and parameters

## Boolean values

Boolean command-line options accept common true/false strings such as `true`, `false`, `1`, `0`, `yes`, and `no`.

## JSON options

`--prior_overrides` and `--fixed_parameter_values` must be JSON objects.

Prior override example:

```bash
--prior_overrides '{"H0": [60.0, 80.0], "Om0": [0.2, 0.4]}'
```

Fixed-parameter example:

```bash
--fixed_parameter_values '{"H0": 67.74, "Om0": 0.3075}'
```

Parameter labels must match the labels produced by the selected cosmology, population, and survey blocks. The inference command prints a parameter table at startup showing sampled, fixed, and overridden parameters.


## Bright-siren counterparts

For `--universe_model bright_sirens`, pass the electromagnetic counterpart as event metadata rather than as `--survey_path`:

```bash
--universe_model bright_sirens --counterpart RA DEC Z
```

`RA` and `DEC` are in radians. The inference loader turns the counterpart into a fixed one-object catalog at `--counterpart_nside` with redshift width `--counterpart_dz`, so the survey/completion parameter block is fixed automatically for this model. Selection samples are still loaded from `--gwselection_path` in the standard way.

## Cosmology block

The standard cosmology block includes:

- `H0`: Hubble constant.
- `Om0`: matter density fraction.

## Survey block

The dark-siren incompleteness model uses survey/completion parameters such as:

- `log10n0`: number-density normalization.
- `z50`: redshift where the survey rolloff reaches 50%.
- `w`: rolloff width.
- `delta`: galaxy number-density evolution.
- `b_miss`: missing-galaxy bias parameter.
- `alpha`: completion-mixture parameter.

## Population block

Population parameters depend on `--pop_model`. Use a small dry run to print the parameter table before committing compute time to a production job.

## Normalization-grid tuning

GW-population mass, mass-ratio, and spin components are normalized on cached trapezoid grids. The defaults (`--norm_nmass 500 --norm_nq 200 --norm_nchi 200`) are intended for development and moderate analyses. You can change individual dimensions from the command line or with the environment variables `DARKSIRENS_GW_N_MASS`, `DARKSIRENS_GW_N_Q`, and `DARKSIRENS_GW_N_CHI`; the active values are printed at startup and saved in `settings.json` as `normalization_grid`.

For production 500-event analyses, especially when priors allow minimum smoothing widths such as `\delta m_{\min}=0.01` or `\sigma_\chi=0.01`, use at least:

- `--norm_nmass 2000` for power-law and broken-power-law mass edges.
- `--norm_nq 1000` for mass-ratio normalizations with low-mass systems near the secondary-mass cutoff.
- `--norm_nchi 1000` for narrow effective-spin components.

If only one distribution has narrow features, increase only the corresponding grid rather than all three dimensions. For final evidence runs or sensitivity checks, compare against a higher-resolution rerun such as `--norm_nmass 5000 --norm_nq 3000 --norm_nchi 3000` and confirm posterior and evidence changes are negligible for the science target.

## Performance tuning

- Increase `--nlive` for more reliable nested-sampling evidences in high dimensions.
- Set `--sel_batch_size` if the injection file is too large to process at once.
- Reduce posterior-predictive grid sizes (`--nm`, `--nq`, `--nz`, `--nchi`) during analyzer smoke tests.
