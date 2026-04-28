"""
volume.py
---------
Comoving volume prior: the agnostic redshift prior used in GW-only
(spectral sirens) inference.

    p(z) ∝ dV_c/dz

No galaxy number-density evolution enters here — `delta` describes
n(z) = n0 (1+z)^delta and is only relevant where galaxies are counted
(catalog.py, completion.py).  Merger rate evolution is accounted for
elsewhere in the pipeline and must not be included here.
"""

import jax.numpy as jnp
from jax import jit

from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams

from .utils import zgrid


@jit
def log_volume_prior(z: float, cosmo: CosmoParams, survey: SurveyParams) -> float:
    """
    Log of the normalised comoving-volume redshift prior evaluated at z.

    The prior is proportional to the comoving volume element dV_c/dz.
    Galaxy number-density evolution (delta) and merger rate evolution
    are both handled outside this module.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate the prior.
    cosmo : CosmoParams
        Cosmological parameters (H0, Om0).
    survey : SurveyParams
        Accepted for API uniformity; not used here.

    Returns
    -------
    float
        log p(z), normalised over zgrid.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0

    pvol = dV_of_z(zgrid, H0, Om0)
    pvol = pvol / jnp.trapezoid(pvol, zgrid)

    return jnp.interp(z, zgrid, jnp.log(pvol))
