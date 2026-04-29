"""
catalog.py
----------
EM catalog redshift prior: p_cat(z | pix).

Each galaxy in the pixel is treated as a Gaussian in redshift space
(centre = photo-z, width = photo-z uncertainty).  The mixture is
weighted by:

    w_i ∝ w_gal,i * dV_c(z_i) * (1+z_i)^delta

where the volume weight reflects the galaxy number density model

    n(z) = n0 (1+z)^delta

Note: the (1+z)^delta factor here is purely the number-density
evolution of the galaxy population.  Merger rate evolution is handled
elsewhere in the pipeline and must not appear in this weight.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog


@jit
def log_catalog_prior(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> float:
    """
    Log of the EM-catalog redshift prior at redshift z for pixel pix.

    Models the within-pixel galaxy distribution as a weighted Gaussian
    mixture:

        p_cat(z | pix) ∝ Σ_i w_i N(z; z_i, σ_i)

    where w_i absorbs the volume weight dV_c(z_i) * (1+z_i)^delta and
    any per-galaxy weights from the catalog (e.g. photo-z quality,
    luminosity).  The exponent delta parametrises the number-density
    evolution n(z) = n0 (1+z)^delta.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate the prior.
    pix : int
        HEALPix pixel index.
    cosmo : CosmoParams
        Cosmological parameters (H0, Om0).
    survey : SurveyParams
        Survey parameters; only `delta` (number-density evolution
        index) is used.
    em_catalog : EMCatalog
        EM galaxy catalog arrays (zgals, dzgals, wgals, …).

    Returns
    -------
    float
        log p_cat(z | pix).
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    delta = survey.delta
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals

    zs = zgals[pix]        # (Ngal,)
    sig = dzgals[pix]      # (Ngal,)

    # Volume-weighted mixture coefficients.
    # (1+z)^delta is the number-density evolution; merger rate is elsewhere.
    vol_weight = dV_of_z(zs, H0, Om0) * (1.0 + zs) ** delta
    w = wgals[pix] * vol_weight
    # Guard: if all galaxies in the pixel have zero weight (e.g. empty pixel
    # or all photo-z outside the survey range), division by zero would produce
    # NaN which propagates silently into logsumexp and corrupts the likelihood.
    w_sum = jnp.sum(w)
    w = w / jnp.where(w_sum > 0.0, w_sum, 1.0)

    return logsumexp(jnp.log(w) + norm.logpdf(z, zs, sig))


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.
log_catalog_prior_vmap = jit(
    vmap(log_catalog_prior, in_axes=(0, 0, None, None, None), out_axes=0)
)