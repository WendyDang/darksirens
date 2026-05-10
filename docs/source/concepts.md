# Concepts

## Spectral sirens

A spectral-siren analysis uses GW data alone. The redshift prior is the cosmological comoving-volume element, while the population model captures the mass, mass-ratio, spin, and redshift distribution of compact-binary mergers.

Use this mode with:

```bash
--universe_model spectral_sirens
```

## Dark sirens with a complete catalog

A complete-catalog dark-siren analysis assumes the electromagnetic catalog traces all possible hosts in the survey volume. The redshift prior is driven by the catalog density in each sky pixel.

Use this mode with:

```bash
--universe_model dark_sirens_complete
```

## Dark sirens with an incomplete catalog

The default dark-siren model combines catalog galaxies with a missing-galaxy completion term. The completeness curve changes with redshift and is controlled by survey parameters such as `z50`, `w`, and density/evolution parameters.

Use this mode with:

```bash
--universe_model dark_sirens
```

## Population models

Population models are selected by name with `--pop_model`. The code uses a registry internally, so documented names map to callable model implementations and prior blocks. Common examples include:

- `powerlaw+peak`
- `brokenpowerlaw+2peaks`

## Selection effects

Selection effects are handled with the GW injection file supplied by `--gwselection_path`. The inference code computes a selection correction for each proposed parameter point, optionally batching the selection calculation with `--sel_batch_size` for memory control.
