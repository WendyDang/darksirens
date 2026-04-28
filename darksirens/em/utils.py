"""
utils.py
--------
Module-level utilities and redshift grid shared across all submodules.

The grid is defined once here so that JAX can trace through it at
compile time (via `jit`) without recompilation when it is imported by
multiple submodules.  Using a log-spaced grid gives finer resolution
at low redshift where the catalog is densest, and coarser resolution
at high redshift where the prior is smooth.
"""

import jax.numpy as jnp
import h5py

# Log-spaced from z~0 to zMax, giving 1000 points.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
zMax: float = 5.0
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), 1000))


def load_survey(survey_path):
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = jnp.asarray(f['zgals'])
        ngals = jnp.asarray(f['ngals'])
        dzgals = jnp.asarray(f['dzgals'])
        wgals = jnp.asarray(f['wgals'])
    return nside, ngals, zgals, dzgals, wgals