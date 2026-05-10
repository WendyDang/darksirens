# Data formats

The command-line tools exchange HDF5 files. This page documents the expected top-level structure used by the loaders and writers.

## GW posterior samples

`--gw_path` should point to an HDF5 file containing per-event posterior samples. Each event must provide samples for the quantities consumed by the likelihood, including luminosity distance, sky position/pixel information, component masses, spin variables, and event-level prior weights.

Because different GW pipelines name datasets differently, inspect `darksirens.gw.utils.load_gw_samples` before adapting a new posterior format. Keep units consistent with the cosmology and population model assumptions used in the run.

## GW selection samples

`--gwselection_path` should point to an HDF5 file of injections/selection samples. The selection loader reads injection parameters and their weights, then the likelihood computes an expected-detection correction.

For large injection sets, use:

```bash
--sel_batch_size 200000
```

Tune the value to fit your memory budget.

## Raw survey file for pixelation

`darksirens_pixelate` expects the input survey HDF5 file to contain these datasets:

| Dataset | Meaning | Units |
| --- | --- | --- |
| `TARGET_RA` | Right ascension | degrees |
| `TARGET_DEC` | Declination | degrees |
| `Z` | Redshift | dimensionless |
| `ZERR` | Redshift uncertainty | dimensionless |
| `WEIGHT` | Galaxy weight | arbitrary/non-negative |

## Pixelated survey output

The pixelation command writes `catalog_pixelated_nside_<nside>.h5` with:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `zgals` | `(npix, max_galaxies_per_pixel)` | Galaxy redshifts per HEALPix pixel, padded with `100.0` |
| `dzgals` | `(npix, max_galaxies_per_pixel)` | Redshift uncertainties, padded with `1.0` |
| `wgals` | `(npix, max_galaxies_per_pixel)` | Galaxy weights, padded with `0.0` |
| `ngals` | `(npix,)` | Number of real galaxies in each pixel |

The file also stores `nside` as an HDF5 attribute.
