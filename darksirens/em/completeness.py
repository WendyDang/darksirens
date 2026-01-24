import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from jax.scipy.special import expit

from darksirens.utils.cosmology import dV_of_z

# ------------------------------------------------------------
# Global redshift grid
# ------------------------------------------------------------
zgrid = jnp.linspace(0.0, 5.0, 1024)

# ------------------------------------------------------------
# Galaxy counts
# ------------------------------------------------------------
@jit
def Ngals_lessthanz(z, pix, zgals):
    zs = zgals[pix]
    return jnp.where(zs < z, 1.0, 0.0).sum()

Ngals_lessthanz_vmap = jit(
    vmap(Ngals_lessthanz, in_axes=(0, None, None), out_axes=0)
)

@partial(jit, static_argnames=['apix'])
def Ngals_expected_lessthanz(z, H0, Om0, n0, delta, apix):
    dN = n0 * apix * dV_of_z(zgrid, H0, Om0) * (1 + zgrid)**(delta - 1)
    dN = jnp.where(zgrid < z, dN, 0.0)
    return jnp.trapezoid(dN, zgrid)

Ngals_expected_lessthanz_vmap = jit(
    vmap(Ngals_expected_lessthanz,
         in_axes=(0, None, None, None, None, None),
         out_axes=0)
)

# ------------------------------------------------------------
# Optional logistic high‑z rolloff
# ------------------------------------------------------------
@jit
def Pcomplete0(z, z1, z50):
    return expit(-z1 * (z / z50) + z1)

# ------------------------------------------------------------
# LSS completeness (unified: alpha=0 → isotropic)
# ------------------------------------------------------------
@partial(jit, static_argnames=['apix'])
def completeness_fraction(H0, Om0, n0, z1, z50, delta,
                          z, pix, apix, zgals,
                          delta_g_pix_z, b_miss, alpha):
    """
    Unified completeness model:
      - alpha = 0 → isotropic
      - alpha > 0 → LSS-aware blended missing density
    """

    # 1. Isotropic completeness curve C_iso(z)
    Nexp = 1.0 + Ngals_expected_lessthanz_vmap(zgrid, H0, Om0, n0, delta, apix)
    Nobs = Ngals_lessthanz_vmap(zgrid, pix, zgals)

    C_iso = jnp.clip(Nobs / Nexp, 0.0, 1.0)
    C_iso = C_iso * Pcomplete0(zgrid, z1, z50)

    # 2. Volume term
    pvol = dV_of_z(zgrid, H0, Om0) * (1 + zgrid)**(delta - 1)

    # 3. Physical isotropic missing density
    pmiss_iso = (1.0 - C_iso) * pvol

    # 4. LSS modulation (physical space)
    delta_g_z = delta_g_pix_z[pix]
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)

    pmiss_LSS = pmiss_iso * (1.0 + b_miss * delta_g_z)

    # 5. Blended missing density
    pmiss_eff = (1 - alpha) * pmiss_iso + alpha * pmiss_LSS
    pmiss_eff = jnp.clip(pmiss_eff, 0.0, jnp.inf)

    # 6. Completeness curve from physical missing density
    C_eff = 1.0 - pmiss_eff / jnp.where(pvol > 0, pvol, 1.0)
    C_eff = jnp.clip(C_eff, 0.0, 1.0)

    # 7. Pixel-level completeness fraction
    Vmiss = jnp.trapezoid(pmiss_eff, zgrid)
    Vmax = jnp.trapezoid(pvol, zgrid)
    f = 1.0 - Vmiss / Vmax

    # 8. Normalized missing PDF for priors
    pmiss_norm = pmiss_eff / jnp.trapezoid(pmiss_eff, zgrid)
    pmiss_z = jnp.interp(z, zgrid, pmiss_norm)

    C_z = jnp.interp(z, zgrid, C_eff)

    return f, pmiss_z, C_z

# vmap version
completeness_fraction_vmap = jit(
    vmap(completeness_fraction,
         in_axes=(None, None, None, None, None, None,
                  0, 0, None, None,
                  None, None, None),
         out_axes=(0, 0, 0))
)

# ------------------------------------------------------------
# Catalog prior
# ------------------------------------------------------------
@jit
def logpcatalog(z, pix, H0, Om0, delta, zgals, dzgals, wgals):
    zs = zgals[pix]
    sig = dzgals[pix]
    w = wgals[pix] * dV_of_z(zs, H0, Om0) * (1 + zs)**(delta - 1)
    w = w / jnp.sum(w)
    return logsumexp(jnp.log(w) + norm.logpdf(z, zs, sig))

logpcatalog_vmap = jit(
    vmap(logpcatalog,
         in_axes=(0, 0, None, None, None, None, None, None),
         out_axes=0)
)

# ------------------------------------------------------------
# Universe prior (unified)
# ------------------------------------------------------------
@partial(jit, static_argnames=['apix'])
def logPriorUniverse(z, pix, H0, Om0, n0, z1, z50,
                     delta, gamma, apix,
                     zgals, dzgals, wgals,
                     delta_g_pix_z, b_miss, alpha):
    """
    Unified dark-siren prior:
      p(z|pix) ∝ [ f p_cat + (1-f) p_miss ] (1+z)^{gamma-1}
    """

    f, pmiss, _ = completeness_fraction_vmap(
        H0, Om0, n0, z1, z50, delta,
        z, pix, apix, zgals,
        delta_g_pix_z, b_miss, alpha
    )

    logpmiss = jnp.nan_to_num(jnp.log(pmiss), neginf=-jnp.inf)
    logpcat = jnp.nan_to_num(
        logpcatalog_vmap(z, pix, H0, Om0, delta, zgals, dzgals, wgals),
        neginf=-jnp.inf
    )

    log_mix = logsumexp(
        jnp.stack([jnp.log(f) + logpcat,
                   jnp.log1p(-f) + logpmiss]),
        axis=0
    )

    return log_mix + (gamma - 1) * jnp.log1p(z)


@partial(jit, static_argnames=['apix'])
def logPriorUniverse_spectralsirens(z, pix, H0, Om0, n0, z1, z50,
                                    delta, gamma, apix, zgals, dzgals, wgals,
                                    delta_g_pix_z, b_miss, alpha):
    """
    Same structure, but with f=0 (no catalog term in the mixture).
    """
    f, pmiss, _ = completeness_fraction_vmap(
        H0, Om0, n0, z1, z50, delta,
        z, pix, apix, zgals,
        delta_g_pix_z, b_miss, alpha
    )

    f = 0.0 * f  # force f=0

    logpmiss = jnp.nan_to_num(jnp.log(pmiss), neginf=-jnp.inf)
    logpcat = jnp.nan_to_num(
        logpcatalog_vmap(z, pix, H0, Om0, delta, zgals, dzgals, wgals),
        neginf=-jnp.inf
    )

    log_mix = logsumexp(
        jnp.stack([jnp.log(f + 1e-300) + logpcat,
                   jnp.log1p(-f) + logpmiss],
                  axis=0),
        axis=0
    )

    return log_mix + (gamma - 1.0) * jnp.log1p(z)


@partial(jit, static_argnames=['apix'])
def logPriorUniverse_spectralsirens_fast(z, pix, H0, Om0, n0, z1, z50,
                                         delta, gamma, apix, zgals, dzgals, wgals,
                                         delta_g_pix_z, b_miss, alpha):
    """
    Pure volume + evolution prior (no catalog, no completeness).
    """
    pvol = dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** (gamma - 1.0)
    pvol = pvol / jnp.trapezoid(pvol, zgrid)
    logpvol = jnp.log(pvol)
    return jnp.interp(z, zgrid, logpvol)


def universe_model_parser(universe_model='dark_sirens'):
    if universe_model == 'dark_sirens':
        logp = logPriorUniverse
    elif universe_model == 'spectral_sirens':
        logp = logPriorUniverse_spectralsirens
    elif universe_model == 'spectral_sirens_fast':
        logp = logPriorUniverse_spectralsirens_fast
    else:
        raise ValueError(f"Unknown universe_model: {universe_model}")
    return logp
