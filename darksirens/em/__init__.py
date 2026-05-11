"""
redshift_prior
==============
Redshift prior models for gravitational-wave cosmological inference.

This package builds p(z | pix, Θ) — the probability that a GW source
lies at redshift z given its sky localisation pixel and cosmological
parameters Θ — under three physical assumptions about EM survey
completeness:

    spectral_sirens
        GW data only; prior = comoving volume element dV_c/dz.
        No galaxy evolution, no merger rate correction (both handled
        elsewhere in the pipeline).

    dark_sirens_complete
        EM catalog assumed 100 % complete; prior = catalog density.
        Galaxy weights carry the number-density evolution (1+z)^delta,
        where delta parametrises n(z) = n0 (1+z)^delta.

    dark_sirens  (default)
        Realistic incomplete catalog; prior = z-dependent mixture of
        catalog density and a missing-galaxy completion term, weighted
        by C_eff(z) — the completeness curve at that specific redshift.

Note on `delta`
---------------
Throughout this package `delta` is the galaxy *number-density*
evolution index: n(z) = n0 (1+z)^delta.  It is distinct from merger
rate evolution, which is accounted for elsewhere in the pipeline and
does not appear here.

Quickstart
----------
    from redshift_prior import get_redshift_prior

    log_prior = get_redshift_prior("dark_sirens")
    lp = log_prior(z_samples, pix_samples, cosmo, survey, em_catalog)

Correctness checks
------------------
Run normalisation checks once at startup to catch modelling errors early:

    from redshift_prior.checks import run_all_checks

    run_all_checks(
        cosmo, survey, em_catalog,
        test_pixels=jnp.array([0, 100, 500]),
        raise_on_failure=True,
    )

Lower-level building blocks are also exported for custom workflows:

    log_volume_prior          – comoving volume prior (scalar)
    log_catalog_prior         – EM catalog density (scalar)
    log_catalog_prior_vmap    – EM catalog density (vectorised)
    catalog_completion        – completion fraction + p_miss (scalar)
    catalog_completion_vmap   – completion fraction + p_miss (vectorised)
    compute_lss_overdensity   – pre-compute δ_g(pix, z) at startup
    PRIOR_REGISTRY            – dict of all registered prior functions
"""

from .prior import get_redshift_prior, PRIOR_REGISTRY
from .volume import log_volume_prior
from .catalog import log_catalog_prior, log_catalog_prior_vmap
from .completion import (
    catalog_completion,
    catalog_completion_vmap,
    completion_clip_diagnostics,
    compute_lss_overdensity,
)
from .utils import zgrid, zMax, load_survey
from . import checks

__all__ = [
    # Factory
    "get_redshift_prior",
    "PRIOR_REGISTRY",
    # Volume
    "log_volume_prior",
    # Catalog
    "log_catalog_prior",
    "log_catalog_prior_vmap",
    # Completion
    "catalog_completion",
    "catalog_completion_vmap",
    "completion_clip_diagnostics",
    "compute_lss_overdensity",
    # Grid
    "zgrid",
    "zMax",
    # Checks
    "checks",
]