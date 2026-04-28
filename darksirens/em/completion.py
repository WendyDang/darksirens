"""
completion.py
-------------
Catalog completion model: characterises the missing-galaxy distribution
for pixels / redshifts where the EM survey is incomplete.

Galaxy number density follows n(z) = n0 (1+z)^delta, so the expected
count per redshift shell scales as:

    dN_exp/dz ∝ n0 * apix * dV_c/dz * (1+z)^delta

The exponent is delta, not (delta-1).  Merger rate evolution is handled
elsewhere in the pipeline and must not appear here.

The model has two regimes controlled by `alpha`:

    alpha = 0  →  isotropic completion
                   Missing galaxies are distributed like the mean
                   expected galaxy density, irrespective of local
                   large-scale structure.

    alpha = 1  →  LSS-aware completion
                   The missing density is modulated by the local
                   galaxy overdensity δ_g(pix, z), weighted by the
                   galaxy bias of the missing population b_miss.

    0 < alpha < 1 → Linear blend of the two.

Public API
----------
catalog_completion(z, pix, cosmo, survey, em_catalog)
    Returns (f, p_miss, C) for a single (z, pix) pair.

catalog_completion_vmap
    JAX-vmapped version over arrays of (z, pix) pairs.

compute_lss_overdensity(zgals, nside)
    Pre-computes δ_g(pix, z) on the global zgrid for all HEALPix pixels.
    Call once at startup and store the result in EMCatalog.delta_g_pix_z.
"""

import jax.numpy as jnp
from jax import jit, vmap
import healpy as hp
from jax.nn import sigmoid

from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .utils import zgrid


# ------------------------------------------------------------
# Internal helpers: observed and expected cumulative counts
# ------------------------------------------------------------

@jit
def _Ngals_lessthanz_grid(pix: int, zgals) -> jnp.ndarray:
    """
    Cumulative observed galaxy count N(<z) for pixel `pix` on zgrid.

    Returns
    -------
    jnp.ndarray of shape (len(zgrid),)
    """
    zs = zgals[pix]                          # (Ngal,)
    mask = zs[None, :] < zgrid[:, None]      # (Nz, Ngal)
    return mask.sum(axis=1).astype(float)


_Ngals_lessthanz_grid_vmap = jit(
    vmap(_Ngals_lessthanz_grid, in_axes=(0, None), out_axes=0)
)


# ------------------------------------------------------------
# LSS overdensity pre-computation (run once at startup)
# ------------------------------------------------------------

def compute_lss_overdensity(zgals, nside: int) -> jnp.ndarray:
    """
    Compute the galaxy overdensity δ_g(pix, z) on the global zgrid for
    every HEALPix pixel at the given nside.

    This should be called once during catalog initialisation and stored
    in ``EMCatalog.delta_g_pix_z``.

    Parameters
    ----------
    zgals : array-like, shape (npix, Ngal_max)
        Per-pixel galaxy redshifts (padded with NaN or sentinel values).
    nside : int
        HEALPix nside parameter.

    Returns
    -------
    delta_g_pix_z : jnp.ndarray, shape (npix, len(zgrid))
        Fractional overdensity relative to the pixel-mean at each z.
    """
    npix = hp.nside2npix(nside)
    pix_indices = jnp.arange(npix)

    # Cumulative counts per pixel per z-bin: (npix, Nz)
    Ncum_pix_z = _Ngals_lessthanz_grid_vmap(pix_indices, zgals)

    # Mean across pixels at each z; guard against empty z-shells
    Nmean_z = jnp.mean(Ncum_pix_z, axis=0)
    Nmean_z = jnp.where(Nmean_z > 0.0, Nmean_z, 1.0)

    delta_g_pix_z = (Ncum_pix_z - Nmean_z[None, :]) / Nmean_z[None, :]
    return delta_g_pix_z


# ------------------------------------------------------------
# High-z logistic rolloff
# ------------------------------------------------------------

@jit
def _survey_rolloff(z: float, z50: float, w: float) -> float:
    """
    Logistic function giving P(complete) → 1 at z ≪ z50 and → 0 at z ≫ z50.

        P(z) = σ((z50 - z) / w)

    Parameters
    ----------
    z : float or array
        Redshift(s).
    z50 : float
        Redshift of 50 % completeness.
    w : float
        Transition width (clipped to ≥ 1e-6 to avoid singularity).
    """
    w = jnp.clip(w, 1e-6, None)
    return sigmoid((z50 - z) / w)


# ------------------------------------------------------------
# Main catalog completion function
# ------------------------------------------------------------

@jit
def catalog_completion(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Characterise catalog incompleteness at redshift z in pixel pix.

    Computes:
      1. The pixel-level completeness fraction f ∈ [0, 1].
      2. The normalised missing-galaxy PDF p_miss(z | pix).
      3. The redshift-dependent completeness curve C(z | pix).

    Physical model
    ~~~~~~~~~~~~~~
    The galaxy number density evolves as n(z) = n0 (1+z)^delta, giving
    an expected count per redshift shell:

        dN_exp/dz ∝ n0 * apix * dV_c/dz * (1+z)^delta

    Merger rate evolution is handled elsewhere and does not enter here.

    The isotropic completeness curve is:

        C_iso(z) = clip( N_obs(<z) / N_exp(<z) , 0, 1 ) * P_rolloff(z)

    where P_rolloff is the logistic high-z survey rolloff.

    The physical volume element (used for the missing-galaxy density) is:

        pvol(z) = dV_c/dz * (1+z)^delta

    The isotropic missing physical density is:

        rho_miss_iso(z) = (1 - C_iso(z)) * pvol(z)

    LSS modulation introduces anisotropy via the galaxy bias b_miss:

        rho_miss_LSS(z) = rho_miss_iso(z) * (1 + b_miss * δ_g(pix, z))

    with δ_g mean-subtracted within the pixel.  The effective missing
    density is the alpha-blend:

        rho_miss_eff = (1 - alpha) * rho_miss_iso + alpha * rho_miss_LSS

    Parameters
    ----------
    z : float
        Redshift at which to evaluate p_miss and C.
    pix : int
        HEALPix pixel index.
    cosmo : CosmoParams
        Cosmological parameters (H0, Om0).
    survey : SurveyParams
        Survey parameters: n0, z50, w, delta, b_miss, alpha.
        `delta` is the number-density evolution index in n(z)=n0(1+z)^delta.
    em_catalog : EMCatalog
        EM catalog: apix, zgals, delta_g_pix_z.

    Returns
    -------
    f : float
        Pixel-level completeness fraction ∈ [0, 1].
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Completeness curve C(z | pix) evaluated at z.
    """
    # --- Unpack parameters ---
    H0, Om0 = cosmo.H0, cosmo.Om0
    n0, z50, w, delta, b_miss, alpha = (
        survey.n0, survey.z50, survey.w,
        survey.delta, survey.b_miss, survey.alpha,
    )
    apix, zgals, delta_g_pix_z = (
        em_catalog.apix, em_catalog.zgals, em_catalog.delta_g_pix_z
    )

    # --- Step 1: Observed cumulative counts on zgrid ---
    N_obs = jnp.cumsum(
        jnp.where(
            zgals[pix][None, :] < zgrid[:, None],
            1.0, 0.0
        ).sum(axis=1)
    )

    # --- Step 2: Expected cumulative counts on zgrid ---
    # n(z) = n0 (1+z)^delta; merger rate evolution is handled elsewhere.
    dN_exp = n0 * apix * dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta
    dz = jnp.diff(zgrid, prepend=zgrid[0])
    N_exp = 1.0 + jnp.cumsum(dN_exp * dz)

    # --- Step 3: Isotropic completeness curve ---
    C_iso = jnp.clip(N_obs / N_exp, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    # --- Step 4: Physical volume element ---
    # Carries the same (1+z)^delta galaxy-density weight as the counts.
    pvol = dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta

    # --- Step 5: Isotropic missing physical density ---
    rho_miss_iso = (1.0 - C_iso) * pvol

    # --- Step 6: LSS-modulated missing density ---
    delta_g_z = delta_g_pix_z[pix]                          # (Nz,)
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)             # mean-subtract
    rho_miss_lss = rho_miss_iso * (1.0 + b_miss * delta_g_z)

    # --- Step 7: Effective missing density (alpha blend) ---
    rho_miss_eff = (1.0 - alpha) * rho_miss_iso + alpha * rho_miss_lss
    rho_miss_eff = jnp.clip(rho_miss_eff, 0.0, jnp.inf)

    # --- Step 8: Completeness curve from effective missing density ---
    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff = jnp.clip(1.0 - rho_miss_eff / pvol_safe, 0.0, 1.0)

    # --- Step 9: Pixel-level completeness fraction ---
    V_miss = jnp.trapezoid(rho_miss_eff, zgrid)
    V_max = jnp.trapezoid(pvol, zgrid)
    f = 1.0 - V_miss / V_max

    # --- Step 10: Normalised missing PDF ---
    norm_factor = jnp.trapezoid(rho_miss_eff, zgrid)
    norm_factor = jnp.where(norm_factor > 0.0, norm_factor, 1.0)
    p_miss_grid = rho_miss_eff / norm_factor

    # Evaluate at the requested z by interpolation
    p_miss_z = jnp.interp(z, zgrid, p_miss_grid)
    C_z = jnp.interp(z, zgrid, C_eff)

    return f, p_miss_z, C_z


# Vectorised over arrays of (z, pix) pairs
catalog_completion_vmap = jit(
    vmap(catalog_completion, in_axes=(0, 0, None, None, None), out_axes=(0, 0, 0))
)