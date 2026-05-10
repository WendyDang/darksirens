# Command-line interface

Installing the package exposes three console scripts.

## `darksirens_pixelate`

Convert a raw galaxy survey HDF5 file into the dense HEALPix layout used by dark-siren inference.

```bash
darksirens_pixelate --survey_path SURVEY.h5 [--save_path OUTDIR] [--nside 64] [--add_plots]
```

Options:

- `--survey_path`: required path to the raw HDF5 survey file.
- `--save_path`: output directory; defaults to the current directory.
- `--nside`: HEALPix NSIDE; defaults to `64`.
- `--add_plots`: create diagnostic skymap, redshift, and occupancy plots.

## `darksirens_inference`

Run hierarchical inference.

```bash
darksirens_inference \
  --gw_path GW.h5 \
  --gwselection_path INJECTIONS.h5 \
  --sampler dynesty \
  [options]
```

### Data options

- `--gw_path`: required GW posterior-sample HDF5 file.
- `--gwselection_path`: required injection/selection HDF5 file.
- `--survey_path`: pixelated survey HDF5 file; required for dark-siren models.
- `--save_path`: directory for settings, samples, plots, and summaries.

### Physical-model options

- `--universe_model`: one of `spectral_sirens`, `dark_sirens`, or `dark_sirens_complete`.
- `--pop_model`: population model name, for example `powerlaw+peak`.
- `--fix_population`: fix all population parameters to fiducial values.
- `--fix_cosmology`: fix cosmological parameters to fiducial values.
- `--fix_survey`: fix survey-completion parameters to fiducial values.
- `--prior_overrides`: JSON object mapping parameter names to `[lower, upper]` prior bounds.
- `--fixed_parameter_values`: JSON object mapping parameter names to fixed scalar values.

### Catalog options

- `--sigma_kernel`: smoothing kernel width used by catalog-related calculations.
- `--use_LSS`: include large-scale-structure overdensity where supported.

### Sampler options

- `--sampler`: required; one of `jaxns`, `dynesty`, or `emcee`.
- `--nlive`: live points for nested samplers.
- `--dlogz`: evidence stopping threshold where supported.
- `--max_samples`: maximum samples for samplers that expose this limit.
- `--nwalkers`: number of walkers for `emcee`.
- `--nsteps`: number of steps for `emcee`.
- `--seed`: random seed.
- `--show_progress`: enable or disable progress bars.

### Performance options

- `--sel_batch_size`: optional injection-selection batch size.

## `darksirens_analyze`

Analyze saved inference products and compute posterior-predictive summaries.

```bash
darksirens_analyze RUN_DIR [--mmin 1] [--mmax 100] [--nm 300]
```

Important options:

- positional `RUN_DIR`: directory produced by `darksirens_inference`.
- `--mmin`, `--mmax`, `--nm`: primary-mass grid bounds and size.
- `--nq`: mass-ratio grid size.
- `--nz`: redshift grid size.
- `--nchi`, `--chimin`, `--chimax`: spin grid configuration.
- `--batch_size`: posterior-predictive evaluation batch size.
- `--cred_lo`, `--cred_hi`: lower and upper credible intervals.
