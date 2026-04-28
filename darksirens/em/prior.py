"""
prior.py
--------
Redshift prior assembly for dark-siren and spectral-siren cosmological
inference with gravitational waves.

Physical picture
~~~~~~~~~~~~~~~~
We want p(z | pix, Θ) — the probability that a GW source at sky
position `pix` has redshift z, given cosmological parameters Θ.

Three regimes are supported, ordered by how much EM information enters:

1. ``"spectral_sirens"``
   GW data only.  No EM catalog.  Prior is the comoving volume element,
   which is the maximally agnostic choice when no galaxy information is
   available.  Galaxy number-density evolution (delta) and merger rate
   evolution do not enter here; merger rate is handled elsewhere.

        p(z | pix) ∝ dV_c/dz

2. ``"dark_sirens_complete"``
   EM catalog available and assumed 100 % complete to the GW horizon.
   The prior is just the galaxy density in the catalog pixel:

        p(z | pix) = p_cat(z | pix)

   Galaxy weights carry the number-density factor (1+z)^delta.
   This is the idealised limit of the general model.

3. ``"dark_sirens"``  (default / general case)
   EM catalog available but *incomplete*: the survey misses some
   fraction of galaxies.  The prior is a mixture of the catalog term
   and a missing-galaxy term:

        p(z | pix) ∝ f * p_cat(z | pix) + (1 - f) * p_miss(z | pix)

   where f is the pixel-level completeness fraction returned by
   ``catalog_completion``.

Usage
-----
    from redshift_prior import get_redshift_prior

    log_prior = get_redshift_prior("dark_sirens")
    lp = log_prior(z_samples, pix_samples, cosmo, survey, em_catalog)
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .volume import log_volume_prior
from .catalog import log_catalog_prior_vmap
from .completion import catalog_completion_vmap


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

    No EM catalog information is used.  `pix` and `em_catalog` are
    accepted for API uniformity but ignored.  Galaxy number-density
    evolution (delta) and merger rate evolution are both handled
    outside this module.

    Suitable for spectral-siren analyses where the mass spectrum of
    compact binaries provides the redshift anchor.
    """
    return vmap(log_volume_prior, in_axes=(0, None, None))(z, cosmo, survey)


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

    The EM survey is taken to be 100 % complete to the GW detection
    horizon, so the catalog fully specifies the galaxy distribution:

        p(z | pix) = p_cat(z | pix)

    Galaxy mixture weights carry the number-density evolution factor
    (1+z)^delta.  This is an upper bound on the information available
    from the catalog; use ``"dark_sirens"`` for the realistic incomplete
    case.
    """
    return log_catalog_prior_vmap(z, pix, cosmo, survey, em_catalog)


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

    Mixes the within-catalog galaxy density and the missing-galaxy
    density according to the pixel completeness fraction f:

        p(z | pix) ∝ f * p_cat(z | pix)  +  (1 - f) * p_miss(z | pix)

    f and p_miss are computed by ``catalog_completion_vmap``, which
    supports both isotropic completion (alpha=0) and LSS-modulated
    completion (alpha>0) transparently.  Both p_cat and p_miss use
    the same (1+z)^delta number-density weighting; merger rate
    evolution is handled elsewhere.
    """
    f, p_miss, _ = catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)

    log_f   = jnp.where(f > 0.0,       jnp.log(f),        -jnp.inf)
    log_1mf = jnp.where(f < 1.0,       jnp.log1p(-f),     -jnp.inf)
    log_p_miss = jnp.where(p_miss > 0.0, jnp.log(p_miss), -jnp.inf)
    log_p_cat  = log_catalog_prior_vmap(z, pix, cosmo, survey, em_catalog)

    return logsumexp(
        jnp.stack([log_f + log_p_cat,
                   log_1mf + log_p_miss]),
        axis=0,
    )


# ------------------------------------------------------------
# Registry and factory
# ------------------------------------------------------------

#: Maps model name → compiled prior function.
#: All functions share the signature::
#:
#:     f(z, pix, cosmo, survey, em_catalog) -> log_prior  (array)
#:
#: Add new physical assumptions by inserting an entry here.
PRIOR_REGISTRY: dict = {
    "spectral_sirens":       _log_prior_spectral_sirens,
    "dark_sirens_complete":  _log_prior_complete_catalog,
    "dark_sirens":           _log_prior_dark_sirens,
}


def get_redshift_prior(model: str):
    """
    Return the compiled log-prior function for the requested model.

    Parameters
    ----------
    model : str
        One of:

        ``"spectral_sirens"``
            GW-only comoving volume prior.  Use when no EM catalog
            is available or when testing the spectral-siren method.

        ``"dark_sirens_complete"``
            Catalog-only prior assuming 100 % survey completeness.
            Use as an optimistic / upper-bound scenario.

        ``"dark_sirens"``
            General incomplete-catalog prior (catalog + missing-galaxy
            mixture).  The recommended default for realistic surveys.

    Returns
    -------
    callable
        A JAX-jitted function with signature::

            log_prior(z, pix, cosmo, survey, em_catalog) -> jnp.ndarray

        where ``z`` and ``pix`` are 1-D arrays of the same length.

    Raises
    ------
    ValueError
        If `model` is not in ``PRIOR_REGISTRY``.

    Examples
    --------
    >>> log_prior = get_redshift_prior("dark_sirens")
    >>> lp = log_prior(z_samples, pix_samples, cosmo, survey, em_catalog)
    """
    if model not in PRIOR_REGISTRY:
        available = ", ".join(f'"{k}"' for k in PRIOR_REGISTRY)
        raise ValueError(
            f"Unknown redshift prior model '{model}'. "
            f"Available models: {available}."
        )
    return PRIOR_REGISTRY[model]
