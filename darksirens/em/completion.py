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

Completeness model
------------------
Completeness is computed as a *differential* ratio per redshift shell:

    C_iso(z) = clip( (dN_obs/dz)(z) / (dN_exp/dz)(z) , 0, 1 ) * P_rolloff(z)

Both numerator and denominator are smoothed with a Gaussian kernel of
width ``_SIGMA_SMOOTH`` before the ratio is formed.  This removes the
smearing artifact of the previous cumulative approach: with cumulative
counts, a galaxy over-density at z=0.3 would inflate C_iso for all z>0.3
even if the survey is incomplete at z=0.5.  The differential ratio
responds locally and faithfully tracks the actual survey depth.

The kernel-smoothed expected dN/dz is independent of pixel and is lifted
out of the vmap alongside N_exp (see Performance notes below).

Per-pixel observed dN/dz is estimated via a Gaussian KDE over the
galaxy positions in the pixel, using the same kernel width, so the ratio
is a proper density estimator throughout.

Mixing weight
-------------
The redshift prior mixes the catalog and missing-galaxy terms using
C_eff(z) — the full redshift-dependent completeness curve — rather than a
scalar pixel-level fraction f.  This is physically more accurate: the
catalog term is trusted where the survey is deep and the missing term
where it is shallow, at every redshift independently.

The scalar f is still returned for diagnostics and for selection-correction
bookkeeping in the likelihood.

Performance notes
-----------------
``_smooth_expected_dndz`` depends only on (H0, Om0, n0, delta, apix) and
must be called *once per likelihood evaluation* before any vmap.

The Gaussian kernel matrix for smoothing is O(N_grid^2) = O(10^6) but is
applied once as a single matrix multiply.  The previous Python for-loop
unrolled into 1000 separate JAX ops at trace time; this is replaced by
``jnp.cumsum`` for N_exp (used in the scalar entry point) and by a KDE
projection for the differential completeness.

Public API
----------
catalog_completion(z, pix, cosmo, survey, em_catalog)
    Returns (f, p_miss, C) for a single (z, pix) pair.

catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)
    Same signature, vectorised over arrays of (z, pix) pairs.
    Computes dN_exp_smooth once internally, then vmaps the inner function.

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


# Gaussian kernel width for smoothing both dN_obs/dz and dN_exp/dz.
# Should be comparable to the typical photo-z uncertainty of the survey.
# Larger values give smoother completeness curves; smaller values preserve
# finer redshift structure at the cost of noisier per-pixel estimates.
_SIGMA_SMOOTH: float = 0.05


# ------------------------------------------------------------
# Internal helpers: observed dN/dz via Gaussian KDE
# ------------------------------------------------------------

def _kde_dndz_obs(pix: int, zgals) -> jnp.ndarray:
    """
    Kernel density estimate of the observed dN_obs/dz for pixel `pix`
    on zgrid, using a Gaussian kernel of width ``_SIGMA_SMOOTH``.

    Each galaxy at redshift z_i contributes a Gaussian centred at z_i.
    The result has units of galaxies per unit redshift, consistently
    with the smoothed expected dN_exp/dz from ``_smooth_expected_dndz``.

    Returns
    -------
    jnp.ndarray of shape (len(zgrid),)
    """
    zs = zgals[pix]                                   # (Ngal,)
    # (Nz, Ngal): Gaussian kernel evaluated at each (zgrid_i, z_j) pair
    K = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2
    ) / (jnp.sqrt(2.0 * jnp.pi) * _SIGMA_SMOOTH)
    return K.sum(axis=1)                              # (Nz,)


_kde_dndz_obs_vmap = jit(
    vmap(_kde_dndz_obs, in_axes=(0, None), out_axes=0)
)


# For the LSS overdensity we still need cumulative counts
def _Ngals_lessthanz_grid(pix: int, zgals) -> jnp.ndarray:
    """Cumulative observed count N_obs(<z) on zgrid for pixel `pix`."""
    zs = zgals[pix]
    mask = zs[None, :] < zgrid[:, None]
    return mask.sum(axis=1).astype(float)


_Ngals_lessthanz_grid_vmap = jit(
    vmap(_Ngals_lessthanz_grid, in_axes=(0, None), out_axes=0)
)


# ------------------------------------------------------------
# Smoothed expected dN/dz — lifted out of the vmap
# ------------------------------------------------------------


def _smooth_expected_dndz(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Gaussian-kernel-smoothed expected dN_exp/dz on zgrid.

    Depends only on (H0, Om0, n0, delta, apix) — independent of pixel
    and redshift — and must be called *once per likelihood evaluation*
    before any vmap over (z, pix) pairs.

    The smooth model dN_exp/dz is convolved with the same Gaussian kernel
    used for the observed KDE, so the ratio dN_obs/dN_exp is a consistent
    completeness estimator throughout.

    Returns
    -------
    jnp.ndarray of shape (len(zgrid),)
        Smoothed expected galaxy density in galaxies per unit redshift.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    n0, delta, apix = survey.n0, survey.delta, em_catalog.apix

    # Raw expected density (galaxies per unit z)
    dN_dz = n0 * apix * dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta

    # Gaussian kernel matrix: K[i, j] = G(zgrid[i] - zgrid[j]; sigma_smooth)
    # Row-normalised so that convolution preserves the integral.
    K = jnp.exp(
        -0.5 * ((zgrid[:, None] - zgrid[None, :]) / _SIGMA_SMOOTH) ** 2
    )
    K = K / K.sum(axis=1, keepdims=True)

    return K @ dN_dz   # (Nz,)


# ------------------------------------------------------------
# LSS overdensity pre-computation (run once at startup)
# ------------------------------------------------------------
@jit
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


def _survey_rolloff(z, z50: float, w: float) -> jnp.ndarray:
    """
    Logistic rolloff: P(complete) -> 1 at z << z50, -> 0 at z >> z50.

        P(z) = sigmoid((z50 - z) / w)
    """
    w = jnp.clip(w, 1e-6, None)
    return sigmoid((z50 - z) / w)


# ------------------------------------------------------------
# Inner completion kernel — takes precomputed dN_exp_smooth
# ------------------------------------------------------------


def _catalog_completion_inner(
    z: float,
    pix: int,
    dN_exp_smooth: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Core completion computation for a single (z, pix) pair.

    Accepts ``dN_exp_smooth`` as a pre-computed argument (from
    ``_smooth_expected_dndz``) so it is not recomputed for every element
    of the vmap.  Use the public functions ``catalog_completion`` or
    ``catalog_completion_vmap`` rather than calling this directly.

    Completeness is estimated as the differential ratio:

        C_iso(z) = clip( KDE_obs(z) / dN_exp_smooth(z) , 0, 1 ) * rolloff(z)

    where KDE_obs is a Gaussian KDE over the per-pixel galaxy positions.
    This responds locally to survey depth, unlike a cumulative ratio which
    smears over-densities to all higher redshifts.

    Returns
    -------
    f : float
        Pixel-level completeness fraction in [0, 1].
        Computed from the volume-integrated missing density; used for
        diagnostics and selection-correction bookkeeping.
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Redshift-dependent completeness C_eff(z | pix) evaluated at z.
        Used as the mixing weight in the redshift prior.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    z50, w, delta, b_miss, alpha = (
        survey.z50, survey.w, survey.delta, survey.b_miss, survey.alpha,
    )
    zgals, delta_g_pix_z = em_catalog.zgals, em_catalog.delta_g_pix_z

    # --- Step 1: Per-pixel observed dN/dz via Gaussian KDE ---
    dN_obs = _kde_dndz_obs(pix, zgals)                 # (Nz,)

    # --- Step 2: Differential completeness curve ---
    # Guard against zero expected density at the survey boundary.
    dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    C_iso = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

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

    # --- Step 7: Effective completeness curve C_eff(z) ---
    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff = jnp.clip(1.0 - rho_miss_eff / pvol_safe, 0.0, 1.0)

    # --- Step 8: Scalar pixel-level completeness fraction (diagnostics) ---
    # f is the volume-averaged completeness; it is NOT used as the mixing
    # weight in the prior — C_eff(z) is used there instead.
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


# vmap over (z, pix); dN_exp_smooth, cosmo, survey, em_catalog are constant.
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
    prefer ``catalog_completion_vmap``, which computes dN_exp_smooth
    only once.

    Returns
    -------
    f : float
        Scalar pixel-level completeness fraction in [0, 1] (diagnostics).
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Redshift-dependent completeness C_eff(z | pix) evaluated at z.
    """
    dN_exp_smooth = _smooth_expected_dndz(cosmo, survey, em_catalog)
    return _catalog_completion_inner(z, pix, dN_exp_smooth, cosmo, survey, em_catalog)


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

    Computes ``dN_exp_smooth`` once before the vmap so it is shared across
    all (z, pix) evaluations rather than recomputed for each one.

    Parameters
    ----------
    z : jnp.ndarray, shape (N,)
    pix : jnp.ndarray, shape (N,)
    cosmo, survey, em_catalog : as in ``catalog_completion``

    Returns
    -------
    f : jnp.ndarray, shape (N,)
        Scalar pixel-level completeness fractions (diagnostics / selection).
    p_miss : jnp.ndarray, shape (N,)
        Normalised missing-galaxy PDF evaluated at each z.
    C : jnp.ndarray, shape (N,)
        Redshift-dependent completeness C_eff(z | pix) at each z.
        This is the quantity used as the mixing weight in the prior.
    """
    dN_exp_smooth = _smooth_expected_dndz(cosmo, survey, em_catalog)
    return _catalog_completion_inner_vmap(z, pix, dN_exp_smooth, cosmo, survey, em_catalog)