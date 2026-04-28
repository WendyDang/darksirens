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

Performance notes
-----------------
``_cumulative_N_exp`` depends only on (H0, Om0, n0, delta, apix) — not on
pixel or redshift — and must be called *once per likelihood evaluation*,
before any vmap.  Calling it inside the vmap would recompute the same
1000-point integral ~250 000 times per likelihood call (256 PE samples ×
~1000 selection samples).

The cumulative integral itself is computed via ``jnp.cumsum`` over the
trapezoidal increments, which produces a single O(N) scan.  The previous
implementation used a Python for-loop over zgrid that unrolled into 1000
separate JAX ops at trace time, making compilation extremely slow and the
XLA graph enormous.

Public API
----------
catalog_completion(z, pix, cosmo, survey, em_catalog)
    Returns (f, p_miss, C) for a single (z, pix) pair.

catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)
    Same signature, vectorised over arrays of (z, pix) pairs.
    Computes N_exp once internally, then vmaps the inner function.

compute_lss_overdensity(zgals, nside)
    Pre-computes delta_g(pix, z) on the global zgrid for all HEALPix pixels.
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
# Internal helpers: observed cumulative counts
# ------------------------------------------------------------

@jit
def _Ngals_lessthanz_grid(pix: int, zgals) -> jnp.ndarray:
    """
    Cumulative observed galaxy count N_obs(<z) for pixel `pix` on zgrid.

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
# Expected cumulative counts — lifted out of the vmap
# ------------------------------------------------------------

@jit
def _cumulative_N_exp(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Expected cumulative galaxy count N_exp(<z) on zgrid.

    Depends only on (H0, Om0, n0, delta, apix) — independent of pixel
    and redshift — and must be called *once per likelihood evaluation*
    before any vmap over (z, pix) pairs.

    Uses a cumulative trapezoid sum (jnp.cumsum) in a single O(N) pass.
    This replaces the previous Python for-loop which unrolled into 1000
    separate JAX ops at trace time.

    Returns
    -------
    jnp.ndarray of shape (len(zgrid),)
        N_exp(<zgrid[i]) for each grid point, offset by +1 to guard
        against division by zero at z -> 0.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    n0, delta, apix = survey.n0, survey.delta, em_catalog.apix

    # Integrand: expected galaxy density per unit redshift
    dN_dz = n0 * apix * dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta

    # Cumulative trapezoid: (f[i] + f[i+1])/2 * dz[i], accumulated left to right
    dz    = jnp.diff(zgrid)                                          # (N-1,)
    trap  = 0.5 * (dN_dz[:-1] + dN_dz[1:]) * dz                    # (N-1,)
    N_cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trap)])   # (N,)

    return 1.0 + N_cum   # +1 guards against zero denominator at z->0


# ------------------------------------------------------------
# LSS overdensity pre-computation (run once at startup)
# ------------------------------------------------------------

def compute_lss_overdensity(zgals, nside: int) -> jnp.ndarray:
    """
    Compute the galaxy overdensity delta_g(pix, z) on the global zgrid
    for every HEALPix pixel at the given nside.

    Call once during catalog initialisation and store the result in
    EMCatalog.delta_g_pix_z.

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

    Ncum_pix_z = _Ngals_lessthanz_grid_vmap(pix_indices, zgals)

    Nmean_z = jnp.mean(Ncum_pix_z, axis=0)
    Nmean_z = jnp.where(Nmean_z > 0.0, Nmean_z, 1.0)

    return (Ncum_pix_z - Nmean_z[None, :]) / Nmean_z[None, :]


# ------------------------------------------------------------
# High-z logistic rolloff
# ------------------------------------------------------------

@jit
def _survey_rolloff(z, z50: float, w: float) -> jnp.ndarray:
    """
    Logistic rolloff: P(complete) -> 1 at z << z50, -> 0 at z >> z50.

        P(z) = sigmoid((z50 - z) / w)
    """
    w = jnp.clip(w, 1e-6, None)
    return sigmoid((z50 - z) / w)


# ------------------------------------------------------------
# Inner completion kernel — takes precomputed N_exp_grid
# ------------------------------------------------------------

@jit
def _catalog_completion_inner(
    z: float,
    pix: int,
    N_exp_grid: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Core completion computation for a single (z, pix) pair.

    Accepts ``N_exp_grid`` as a pre-computed argument so it is not
    recomputed for every element of the vmap.  Use the public functions
    ``catalog_completion`` or ``catalog_completion_vmap`` instead of
    calling this directly.

    Returns
    -------
    f : float
        Pixel-level completeness fraction in [0, 1].
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Completeness curve C(z | pix) evaluated at z.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    z50, w, delta, b_miss, alpha = (
        survey.z50, survey.w, survey.delta, survey.b_miss, survey.alpha,
    )
    zgals, delta_g_pix_z = em_catalog.zgals, em_catalog.delta_g_pix_z

    # --- Step 1: Observed cumulative counts on zgrid ---
    N_obs = _Ngals_lessthanz_grid(pix, zgals)           # (Nz,)

    # --- Step 2: Isotropic completeness curve ---
    # N_exp_grid was computed once outside the vmap.
    C_iso = jnp.clip(N_obs / N_exp_grid, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    # --- Step 3: Physical volume element ---
    # (1+z)^delta is galaxy number-density evolution; merger rate is elsewhere.
    pvol = dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta

    # --- Step 4: Isotropic missing physical density ---
    rho_miss_iso = (1.0 - C_iso) * pvol

    # --- Step 5: LSS-modulated missing density ---
    delta_g_z = delta_g_pix_z[pix]
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)        # mean-subtract within pixel
    rho_miss_lss = rho_miss_iso * (1.0 + b_miss * delta_g_z)

    # --- Step 6: Effective missing density (alpha blend) ---
    rho_miss_eff = (1.0 - alpha) * rho_miss_iso + alpha * rho_miss_lss
    rho_miss_eff = jnp.clip(rho_miss_eff, 0.0, jnp.inf)

    # --- Step 7: Completeness curve ---
    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff = jnp.clip(1.0 - rho_miss_eff / pvol_safe, 0.0, 1.0)

    # --- Step 8: Pixel-level completeness fraction ---
    V_miss = jnp.trapezoid(rho_miss_eff, zgrid)
    V_max  = jnp.trapezoid(pvol, zgrid)
    f = 1.0 - V_miss / V_max

    # --- Step 9: Normalised missing PDF ---
    norm_factor = jnp.trapezoid(rho_miss_eff, zgrid)
    norm_factor = jnp.where(norm_factor > 0.0, norm_factor, 1.0)
    p_miss_grid = rho_miss_eff / norm_factor

    p_miss_z = jnp.interp(z, zgrid, p_miss_grid)
    C_z      = jnp.interp(z, zgrid, C_eff)

    return f, p_miss_z, C_z


# vmap over (z, pix); N_exp_grid, cosmo, survey, em_catalog are held constant.
_catalog_completion_inner_vmap = vmap(
    _catalog_completion_inner,
    in_axes=(0, 0, None, None, None, None),
    out_axes=(0, 0, 0),
)


# ------------------------------------------------------------
# Public API — same external signature as before
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

    Scalar entry point.  For vectorised use over many (z, pix) pairs
    prefer ``catalog_completion_vmap``, which computes N_exp only once.

    Returns
    -------
    f : float
        Pixel-level completeness fraction in [0, 1].
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Completeness curve C(z | pix) evaluated at z.
    """
    N_exp_grid = _cumulative_N_exp(cosmo, survey, em_catalog)
    return _catalog_completion_inner(z, pix, N_exp_grid, cosmo, survey, em_catalog)


@jit
def catalog_completion_vmap(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Vectorised catalog completion over arrays of (z, pix) pairs.

    Computes ``N_exp_grid`` once before the vmap so it is shared across
    all (z, pix) evaluations rather than recomputed for each one.

    Parameters
    ----------
    z : jnp.ndarray, shape (N,)
    pix : jnp.ndarray, shape (N,)
    cosmo, survey, em_catalog : as in ``catalog_completion``

    Returns
    -------
    f, p_miss, C : jnp.ndarray, each shape (N,)
    """
    N_exp_grid = _cumulative_N_exp(cosmo, survey, em_catalog)
    return _catalog_completion_inner_vmap(z, pix, N_exp_grid, cosmo, survey, em_catalog)