"""
prior.py
--------
Redshift prior assembly for dark-siren and spectral-siren cosmological
inference with gravitational waves.

Physical picture
~~~~~~~~~~~~~~~~
We want p(z | pix, Θ) — the probability that a GW source at sky
position ``pix`` has redshift z, given cosmological parameters Θ.

Four regimes are supported:

1. ``"spectral_sirens"``
   GW data only.  Prior is the comoving volume element dV_c/dz.

2. ``"bright_sirens"``
   Counterpart-informed inference using a synthetic one-object catalog.
   The prior is the same catalog-density model as ``dark_sirens_complete``
   evaluated on the fixed counterpart redshift and sky position.

3. ``"dark_sirens_complete"``
   EM catalog assumed 100 % complete.

        p(z | pix) = p_cat(z | pix)

   **Empty-pixel fallback**: at nside ≥ 64 many pixels contain zero
   catalog galaxies, making log_p_cat = -inf.  For these pixels we
   fall back to the volume prior — the maximally agnostic choice
   consistent with the complete-catalog assumption (no information
   from an empty pixel).  Without the fallback the sampler sees
   -inf for every proposal, finds no valid live points, and fails
   silently.

4. ``"dark_sirens"``  (default)
   Incomplete catalog; prior is a mixture weighted by C_eff(z):

        p(z | pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                   + (1 - C_eff(z|pix)) * p_miss(z|pix)

   Empty-pixel handling: C_eff → 0 automatically when the pixel has
   no observed galaxies, routing the prior entirely to p_miss.  No
   explicit fallback needed here.

Performance: single-vmap fusion in ``_log_prior_dark_sirens``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The previous implementation called ``catalog_completion_vmap`` and
``log_catalog_prior_vmap`` as two separate vmaps over the same (z, pix)
pairs.  The new implementation fuses them into a single inner function
that is vmapped once:

  1. ``_precompute_grids`` runs once (pixel-independent grids).
  2. A single vmap over (z, pix) evaluates completion + catalog prior
     in one sweep, halving the vmap invocation overhead.
  3. The per-pixel KDE lookup (cached) happens once per (z, pix) pair
     instead of once per vmap call.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .volume import log_volume_prior, log_volume_prior_vmap, _precompute_volume_grid
from .catalog import log_catalog_prior
from .completion import _precompute_grids, _catalog_completion_inner

from .utils import zgrid


# ------------------------------------------------------------
# Individual prior implementations
# ------------------------------------------------------------

@jit
def _log_prior_spectral_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    GW-only prior: normalised comoving volume element.

    ``pix`` and ``em_catalog`` are accepted for API uniformity; ignored.
    """
    return log_volume_prior_vmap(z, cosmo, survey)


@jit
def _log_prior_complete_catalog(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior under the complete-catalog assumption.

        p(z | pix) = p_cat(z | pix)

    Empty-pixel fallback
    --------------------
    At nside ≥ 64 many pixels contain no catalog galaxies, giving
    log_p_cat = -inf.  Those pixels carry no redshift information
    under the complete-catalog assumption, which is equivalent to
    using the agnostic volume prior.

    We apply the fallback per-sample: if log_p_cat[i] is not finite,
    substitute log_volume_prior at z[i].  This is correct and safe
    inside JIT because ``jnp.where`` is evaluated element-wise on the
    output arrays without branching in the computational graph.

    Note: ``dark_sirens`` does not need this because C_eff → 0 for
    empty pixels automatically routes the mixture to p_miss.
    """
    from .catalog import log_catalog_prior_vmap  # local import avoids circular
    log_p_cat = log_catalog_prior_vmap(z, pix, cosmo, survey, em_catalog)

    # Precompute volume grid once (shared across all samples via CSE).
    pvol_norm  = _precompute_volume_grid(cosmo)
    log_p_vol  = vmap(lambda z_i: jnp.interp(z_i, zgrid, jnp.log(pvol_norm)))(z)

    return jnp.where(jnp.isfinite(log_p_cat), log_p_cat, log_p_vol)


@jit
def _log_prior_dark_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior with catalog completion (the general case).

    Mixes catalog and missing-galaxy densities using C_eff(z):

        p(z|pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                 + (1 - C_eff(z|pix)) * p_miss(z|pix)

    Single-vmap implementation
    --------------------------
    ``_precompute_grids`` (pixel-independent) runs once.  A single inner
    function computes both the completion term and the catalog prior for
    each (z_i, pix_i) pair.  This function is vmapped once, replacing
    the previous two-vmap implementation (one for completion, one for
    catalog prior).

    The per-pixel KDE lookup uses ``em_catalog.dN_obs_kde`` (precomputed
    at startup by ``build_pixel_kde_cache``) — an O(1) index rather than
    an O(N_grid × N_max_gals) recomputation.
    """
    # Pixel-independent grids — hoisted out of the vmap by JAX CSE.
    grids = _precompute_grids(cosmo, survey, em_catalog)

    def _inner(z_i, pix_i):
        # --- Completion ---
        _, p_miss, C_z = _catalog_completion_inner(z_i, pix_i, grids, survey, em_catalog)

        # --- Catalog prior ---
        log_p_cat = log_catalog_prior(z_i, pix_i, cosmo, survey, em_catalog)

        # --- Numerically safe log-space mixture ---
        log_C      = jnp.where(C_z   >  0.0, jnp.log(C_z),       -jnp.inf)
        log_1mC    = jnp.where(C_z   <  1.0, jnp.log1p(-C_z),    -jnp.inf)
        log_p_miss = jnp.where(p_miss > 0.0, jnp.log(p_miss),    -jnp.inf)
        log_p_cat  = jnp.nan_to_num(log_p_cat, neginf=-jnp.inf)
        log_p_miss = jnp.nan_to_num(log_p_miss, neginf=-jnp.inf)

        return logsumexp(jnp.stack([
            log_C   + log_p_cat,
            log_1mC + log_p_miss,
        ]))

    return vmap(_inner)(z, pix)


# ------------------------------------------------------------
# Registry and factory
# ------------------------------------------------------------

#: Maps model name → compiled prior function.
#: Signature: f(z, pix, cosmo, survey, em_catalog) → log_prior (array).
PRIOR_REGISTRY: dict = {
    "spectral_sirens":      _log_prior_spectral_sirens,
    "bright_sirens":        _log_prior_complete_catalog,
    "dark_sirens_complete": _log_prior_complete_catalog,
    "dark_sirens":          _log_prior_dark_sirens,
}


def get_redshift_prior(model: str):
    """
    Return the compiled log-prior function for the requested model.

    Parameters
    ----------
    model : str
        One of ``"spectral_sirens"``, ``"bright_sirens"``,
        ``"dark_sirens_complete"``, or ``"dark_sirens"``.

    Returns
    -------
    callable with signature::

        log_prior(z, pix, cosmo, survey, em_catalog) -> jnp.ndarray

    Raises
    ------
    ValueError if model is not in ``PRIOR_REGISTRY``.
    """
    if model not in PRIOR_REGISTRY:
        available = ", ".join(f'"{k}"' for k in PRIOR_REGISTRY)
        raise ValueError(
            f"Unknown redshift prior model '{model}'. "
            f"Available: {available}."
        )
    return PRIOR_REGISTRY[model]