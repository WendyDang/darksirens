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

import jax
import jax.numpy as jnp
from functools import partial

def build_binned_catalog(zgals, dzgals, wgals, nbins=100):
    # zgals, dzgals, wgals: (npix, maxgals)

    npix, maxgals = zgals.shape

    # Global z-range from non-padded galaxies (wgals > 0)
    mask = wgals > 0
    zmin = jnp.min(jnp.where(mask, zgals, jnp.inf))
    zmax = jnp.max(jnp.where(mask, zgals, -jnp.inf))

    edges = jnp.linspace(zmin, zmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def per_pixel(z_row, dz_row, w_row):
        # z_row, dz_row, w_row: (maxgals,)

        # Ignore padded galaxies via weights (w=0)
        # Assign each galaxy to a bin
        bin_idx = jnp.digitize(z_row, edges) - 1
        bin_idx = jnp.clip(bin_idx, 0, nbins - 1)

        # Weighted sums per bin
        W = jax.ops.segment_sum(w_row, bin_idx, nbins)
        Z = jax.ops.segment_sum(w_row * z_row, bin_idx, nbins)
        S = jax.ops.segment_sum(w_row * dz_row, bin_idx, nbins)

        z_bin = jnp.where(W > 0, Z / W, centers)
        sig_bin = jnp.where(W > 0, S / W, 0.03 * (1 + centers))

        return z_bin, sig_bin, W

    z_bin, sig_bin, W = jax.vmap(per_pixel, in_axes=(0, 0, 0))(zgals, dzgals, wgals)
    return z_bin, sig_bin, W, centers


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
LOG_SQRT_2PI = 0.5 * jnp.log(2 * jnp.pi)

@jit
def logpcatalog(z, pix, H0, Om0, delta, zgals, dzgals, wgals):
    # Extract galaxy data for this pixel
    zs = zgals[pix]          # (Ng,)
    sig = dzgals[pix]        # (Ng,)
    w   = wgals[pix]         # (Ng,)

    # Compute log-weights including cosmology + LSS factor
    log_w = (
        jnp.log(w + 1e-300)
        + jnp.log(dV_of_z(zs, H0, Om0))
        + (delta - 1.0) * jnp.log1p(zs)
    )

    # Gaussian log-kernel: log N(z | zs, sig)
    log_norm = -jnp.log(sig) - LOG_SQRT_2PI
    log_kernel = log_norm - 0.5 * ((z - zs) / sig)**2

    # Mixture log-probability
    return logsumexp(log_w + log_kernel)

logpcatalog_vmap = jit(
    vmap(
        logpcatalog,
        in_axes=(0, 0, None, None, None, None, None, None),
        out_axes=0
    )
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

# ============================================================
#  Experimental LF-based completeness model (drop-in)
#  Names are prefixed with lf_ to avoid collisions
# ============================================================

# ------------------------------------------------------------
# 1. Schechter LF parameters (you can tune these)
# ------------------------------------------------------------
def lf_phi_star(z, phi0, beta):
    """Redshift evolution of phi*."""
    return phi0 * (1.0 + z)**beta

def lf_M_star(z, M0, Q):
    """Redshift evolution of M*."""
    return M0 - Q * z

def lf_schechter_integral(Mlim, z, phi0, M0, alpha, Q, beta):
    """
    Integral of Schechter LF from Lmin to infinity.
    Returns number density n(z | Mlim).
    """
    Mstar = lf_M_star(z, M0, Q)
    phi_star = lf_phi_star(z, phi0, beta)

    # Convert magnitude threshold to L/L*
    x = 10.0**(0.4 * (Mstar - Mlim))

    # Upper incomplete gamma: Γ(α+1, x)
    return phi_star * gammaincc(alpha + 1.0, x)


# ------------------------------------------------------------
# 2. Magnitude limit per pixel
# ------------------------------------------------------------
def lf_mlim_from_counts(zgals_pix):
    """
    Experimental: infer an effective magnitude limit from the
    observed galaxy distribution in a pixel.
    For now: use the 90th percentile of observed magnitudes.
    Replace with real depth maps if available.
    """
    # Placeholder: assume zgals_pix has magnitudes stored somewhere
    # You can replace this with real magnitude data.
    return 23.0  # constant depth for now


# ------------------------------------------------------------
# 3. Expected cumulative counts using LF
# ------------------------------------------------------------
@jit
def lf_Nexpected_lessthanz(z, pix, H0, Om0, apix,
                           phi0, M0, alpha, Q, beta,
                           mlim_pix):
    """
    Expected cumulative number of galaxies up to z in pixel pix
    using a Schechter LF and magnitude limit mlim_pix.
    """
    # Distance modulus μ(zgrid)
    dL_grid = luminosity_distance(zgrid, H0, Om0)
    mu_grid = 5.0 * jnp.log10(dL_grid * 1e6) - 5.0  # Mpc → pc

    # Absolute magnitude limit at each z'
    Mlim_grid = mlim_pix - mu_grid  # ignoring K-correction for now

    # LF integral at each z'
    n_grid = lf_schechter_integral(Mlim_grid, zgrid,
                                   phi0, M0, alpha, Q, beta)

    # Expected counts: ∫ n(z') dV/dz' dz'
    dVdz = dV_of_z(zgrid, H0, Om0)
    dN = n_grid * dVdz * apix

    # Only integrate up to z
    dN = jnp.where(zgrid < z, dN, 0.0)

    return jnp.trapezoid(dN, zgrid)


lf_Nexpected_lessthanz_vmap = jit(
    vmap(lf_Nexpected_lessthanz,
         in_axes=(0, None, None, None, None,
                  None, None, None, None, None,
                  0),
         out_axes=0)
)


# ------------------------------------------------------------
# 4. Observed cumulative counts (same as your current version)
# ------------------------------------------------------------
@jit
def lf_Ngals_lessthanz(z, pix, zgals):
    zs = zgals[pix]
    return jnp.where(zs < z, 1.0, 0.0).sum()

lf_Ngals_lessthanz_vmap = jit(
    vmap(lf_Ngals_lessthanz, in_axes=(0, None, None), out_axes=0)
)


# ------------------------------------------------------------
# 5. LF-based completeness function
# ------------------------------------------------------------
@jit
def lf_Pcomplete0(z, z1, z50):
    """Optional parametric modulation."""
    return expit(-z1 * (z / z50) + z1)


@partial(jit, static_argnames=['apix'])
def lf_completeness_fraction(H0, Om0,
                             phi0, M0, alpha, Q, beta,
                             z1, z50,
                             z, pix, apix, zgals, mlim_pix):
    """
    LF-based completeness:
      - continuous
      - per-pixel
      - redshift-dependent
    """
    # Expected cumulative counts
    Nexp = 1.0 + lf_Nexpected_lessthanz_vmap(
        zgrid, pix, H0, Om0, apix,
        phi0, M0, alpha, Q, beta,
        mlim_pix
    )

    # Observed cumulative counts
    Nobs = lf_Ngals_lessthanz_vmap(zgrid, pix, zgals)

    # Continuous completeness
    ratio = Nobs / Nexp
    ratio = jnp.clip(ratio, 0.0, 1.0)

    # Optional parametric modulation
    ratio = ratio * lf_Pcomplete0(zgrid, z1, z50)

    # Volume prior
    pvol = dV_of_z(zgrid, H0, Om0)

    # Completeness fraction f(p)
    V = jnp.trapezoid(ratio * pvol, zgrid)
    Vmax = jnp.trapezoid(pvol, zgrid)
    f = V / Vmax

    # Missing-galaxy PDF
    pmiss = (1.0 - ratio) * pvol
    pmiss_normed = pmiss / jnp.trapezoid(pmiss, zgrid)

    pmiss_z = jnp.interp(z, zgrid, pmiss_normed)

    return f, pmiss_z, ratio


lf_completeness_fraction_vmap = jit(
    vmap(lf_completeness_fraction,
         in_axes=(None, None,
                  None, None, None, None, None,
                  None, None,
                  0, 0, None, None, 0),
         out_axes=(0, 0, 0))
)

