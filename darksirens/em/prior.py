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
   and a missing-galaxy term, with a *redshift-dependent* mixing weight:

        p(z | pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                   + (1 - C_eff(z|pix)) * p_miss(z|pix)

   C_eff(z|pix) is the completeness curve evaluated at the specific
   redshift z rather than a scalar pixel-level average f.  This means
   the catalog term is trusted where the survey is deep and the missing
   term where it is shallow, at every redshift independently.

   The scalar f returned by ``catalog_completion_vmap`` is still used
   for diagnostics and selection-correction bookkeeping in the likelihood
   but does not appear in the prior mixture.

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
    density using the *redshift-dependent* completeness curve C_eff(z):

        p(z | pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                   + (1 - C_eff(z|pix)) * p_miss(z|pix)

    C_eff(z) is the third return value of ``catalog_completion_vmap``.
    It is evaluated at the specific redshift z of each sample, so the
    mixing weight tracks the actual survey depth at that redshift rather
    than using a volume-averaged scalar f.

    C_eff is derived from the differential completeness (dN_obs/dN_exp
    per shell), which responds locally to survey depth without smearing
    over-densities to higher redshifts.

    The scalar f (first return value) is computed but not used here; it
    remains available for diagnostics and selection-correction bookkeeping
    in the likelihood.
    """
    # f: scalar completeness fraction (not used as mixing weight here)
    # p_miss: normalised missing-galaxy PDF at z
    # C_z: redshift-dependent completeness curve at z  ← mixing weight
    _, p_miss, C_z = catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)

    log_C   = jnp.where(C_z > 0.0,   jnp.log(C_z),       -jnp.inf)
    log_1mC = jnp.where(C_z < 1.0,   jnp.log1p(-C_z),    -jnp.inf)
    log_p_miss = jnp.where(p_miss > 0.0, jnp.log(p_miss), -jnp.inf)
    log_p_cat  = log_catalog_prior_vmap(z, pix, cosmo, survey, em_catalog)

    return logsumexp(
        jnp.stack([log_C   + log_p_cat,
                   log_1mC + log_p_miss]),
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
            General incomplete-catalog prior with redshift-dependent
            mixing weight C_eff(z).  The recommended default for
            realistic surveys.

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