"""
utils.py
--------
Shared utility functions for the hierarchical inference likelihood.

The key design principle here is that ``log_sample_weight`` is a *pure
function* of its arguments — no closures over data or parameters.  This
makes it:

  1. Independently unit-testable against known analytic cases.
  2. Profil-able in isolation (JAX's profiler can attribute cost here).
  3. Reusable from both the PE term and the selection term without
     code duplication.

The Jacobian
------------
Changing integration variables from (dL, m1_det, m2_det) to
(z, m1_src, m2_src) introduces a factor:

    |∂(dL, m1_det, m2_det) / ∂(z, m1_src, m2_src)|
        = d(dL)/dz * (1+z) * (1+z)
        = ddL_of_z(z, dL, H0, Om0) * (1+z)^2

In log space: log ddL_of_z + 2 log(1+z).

This is the *only* place in the codebase where this Jacobian is
computed.  Do not inline it elsewhere.
"""

from __future__ import annotations

import jax.numpy as jnp

from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog


def log_jacobian_dL_to_z(
    z: jnp.ndarray,
    dL: jnp.ndarray,
    H0: jnp.ndarray,
    Om0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log-Jacobian for the variable change (dL, m1_det, m2_det) → (z, m1_src, m2_src).

    Parameters
    ----------
    z : redshift at the sample point
    dL : luminosity distance [Mpc] at the sample point
    H0, Om0 : cosmological parameters

    Returns
    -------
    log |J| = log d(dL)/dz + 2 log(1+z)
    """
    return jnp.log(ddL_of_z(z, dL, H0, Om0)) + 2.0 * jnp.log1p(z)


def log_sample_weight(
    m1det: jnp.ndarray,
    q: jnp.ndarray,
    dL: jnp.ndarray,
    chieff: jnp.ndarray,
    pix: jnp.ndarray,
    prior_wt: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    log_p_pop_fn,
    log_prior_z_fn,
) -> jnp.ndarray:
    """
    Per-sample log importance weight, shared by the PE and selection terms.

    The importance weight reweights samples drawn from ``prior_wt``
    (the PE proposal or the injection draw distribution) to the
    population-plus-cosmology model:

        log w = log p_pop(m1_src, q, z, chi_eff | λ)
              + log p_z(z | pix, Θ)
              - log |J(dL → z)|        ← change of variables
              - log p_draw(sample)     ← proposal density

    Parameters
    ----------
    m1det : detector-frame primary mass [M_sun]
    q : mass ratio m2/m1, pre-computed at event construction
    dL : luminosity distance [Mpc]
    chieff : effective inspiral spin
    pix : HEALPix pixel index
    prior_wt : PE prior weight / injection draw probability at this sample
    cosmo : CosmoParams
    survey : SurveyParams
    pop_params : flat parameter vector for the population model
    catalog : EMCatalog (PE catalog or selection catalog)
    log_p_pop_fn : callable(m1_src, q, z, chieff, pop_params) → log probability
    log_prior_z_fn : callable(z, pix, catalog) → log probability
        Should already incorporate the finite-value guard (replace -inf → -1e6).

    Returns
    -------
    log w : scalar or array matching the shape of the inputs
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    z     = z_of_dL(dL, H0, Om0)
    m1src = m1det / (1.0 + z)

    return (
        log_p_pop_fn(m1src, q, z, chieff, pop_params)
        + log_prior_z_fn(z, pix, catalog)
        - log_jacobian_dL_to_z(z, dL, H0, Om0)
        - jnp.log(prior_wt)
    )
