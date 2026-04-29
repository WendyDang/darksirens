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

# Gaussian convolution kernel precomputed once at module load time.
# K[i, j] = G(zgrid[i] - zgrid[j]; _SIGMA_SMOOTH), row-normalised.
# Shape (Nz, Nz) = (1000, 1000) — 8 MB for float64.
# Rebuilding this 1000×1000 matrix on every likelihood call (as a local
# variable inside _smooth_expected_dndz) costs O(N²) allocation and
# floating-point work for no benefit: zgrid and _SIGMA_SMOOTH are both
# module-level constants that never change.
_K_SMOOTH: jnp.ndarray = None  # initialised below after zgrid is imported


def _build_kernel() -> jnp.ndarray:
    K = jnp.exp(
        -0.5 * ((zgrid[:, None] - zgrid[None, :]) / _SIGMA_SMOOTH) ** 2
    )
    return K / K.sum(axis=1, keepdims=True)  # row-normalise


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
# Precomputed grids bundle — lifted out of the vmap
# ------------------------------------------------------------

from typing import NamedTuple

class _CompletionGrids(NamedTuple):
    """
    Pixel-independent arrays computed once per likelihood evaluation.

    Both fields have shape (len(zgrid),).  Passing them as a NamedTuple
    lets the vmap treat the whole bundle with in_axes=None, broadcasting
    it as a constant across all (z, pix) pairs.
    """
    dN_exp_smooth: jnp.ndarray  # smoothed expected galaxy density [gal/dz]
    pvol:          jnp.ndarray  # dV_c/dz * (1+z)^delta [galaxy-weighted vol]


def _smooth_expected_dndz(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Gaussian-kernel-smoothed expected dN_exp/dz on zgrid.

    Depends only on (H0, Om0, n0, delta, apix).  Uses the module-level
    kernel matrix ``_K_SMOOTH`` so no matrix is allocated here.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    n0, delta, apix = survey.n0, survey.delta, em_catalog.apix

    dN_dz = n0 * apix * dV_of_z(zgrid, H0, Om0) * (1.0 + zgrid) ** delta
    return _K_SMOOTH @ dN_dz


def _pvol_grid(cosmo: CosmoParams, survey: SurveyParams) -> jnp.ndarray:
    """
    Galaxy-number-density-weighted comoving volume element on zgrid.

        pvol(z) = dV_c/dz * (1+z)^delta

    Depends only on (H0, Om0, delta) — independent of pixel and
    redshift — and must NOT be recomputed inside the vmap.
    """
    return dV_of_z(zgrid, cosmo.H0, cosmo.Om0) * (1.0 + zgrid) ** survey.delta


def _precompute_grids(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> _CompletionGrids:
    """
    Compute all pixel-independent grids once per likelihood evaluation.
    Pass the result to ``_catalog_completion_inner_vmap`` as a constant.
    """
    return _CompletionGrids(
        dN_exp_smooth=_smooth_expected_dndz(cosmo, survey, em_catalog),
        pvol=_pvol_grid(cosmo, survey),
    )


# ------------------------------------------------------------
# LSS overdensity pre-computation (run once at startup)
# ------------------------------------------------------------
def compute_lss_overdensity(zgals, nside: int) -> jnp.ndarray:
    """
    Compute the galaxy overdensity delta_g(pix, z) on the global zgrid
    for every HEALPix pixel at the given nside.

    Call once during catalog initialisation and store the result in
    EMCatalog.delta_g_pix_z.

    Not JIT-compiled: ``hp.nside2npix`` is a Python/NumPy function that
    returns a concrete Python int.  Decorating with ``@jit`` is fragile
    because JAX would need to retrace whenever ``nside`` changes, and
    calling a NumPy function inside a JIT boundary produces confusing
    errors if a traced integer is ever passed accidentally.  Since this
    function is called exactly once at startup the compilation overhead
    is irrelevant; the inner vmapped helper is JIT-compiled separately.

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
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Core completion computation for a single (z, pix) pair.

    Accepts ``grids`` (a ``_CompletionGrids`` NamedTuple) containing all
    pixel-independent arrays precomputed outside the vmap.  In particular:

    - ``grids.pvol`` replaces the previous ``dV_of_z(zgrid, H0, Om0) *
      (1+z)^delta`` call that was executed for every (z, pix) element.
      Since pvol depends only on (H0, Om0, delta) it must be computed
      once per likelihood call, not once per sample.

    - ``grids.dN_exp_smooth`` is the smoothed expected dN/dz, also
      pixel-independent and lifted for the same reason.

    Use the public wrappers ``catalog_completion`` / ``catalog_completion_vmap``
    rather than calling this directly.

    Returns
    -------
    f : float
        Pixel-level completeness fraction in [0, 1] (diagnostics).
    p_miss : float
        Normalised missing-galaxy PDF evaluated at z.
    C : float
        Redshift-dependent completeness C_eff(z | pix) evaluated at z.
    """
    z50, w, delta, b_miss, alpha = (
        survey.z50, survey.w, survey.delta, survey.b_miss, survey.alpha,
    )
    zgals, delta_g_pix_z = em_catalog.zgals, em_catalog.delta_g_pix_z

    dN_exp_smooth = grids.dN_exp_smooth
    pvol          = grids.pvol

    # --- Step 1: Per-pixel observed dN/dz via Gaussian KDE ---
    dN_obs = _kde_dndz_obs(pix, zgals)                 # (Nz,)

    # --- Step 2: Differential completeness curve ---
    dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    C_iso = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    # --- Step 3: Isotropic missing physical density ---
    rho_miss_iso = (1.0 - C_iso) * pvol

    # --- Step 4: LSS-modulated missing density ---
    delta_g_z = delta_g_pix_z[pix]
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)
    rho_miss_lss = rho_miss_iso * (1.0 + b_miss * delta_g_z)

    # --- Step 5: Effective missing density (alpha blend) ---
    rho_miss_eff = (1.0 - alpha) * rho_miss_iso + alpha * rho_miss_lss
    rho_miss_eff = jnp.clip(rho_miss_eff, 0.0, jnp.inf)

    # --- Step 6: Effective completeness curve C_eff(z) ---
    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff = jnp.clip(1.0 - rho_miss_eff / pvol_safe, 0.0, 1.0)

    # --- Step 7: Scalar completeness fraction (diagnostics) ---
    f = 1.0 - jnp.trapezoid(rho_miss_eff, zgrid) / jnp.trapezoid(pvol, zgrid)

    # --- Step 8: Normalised missing PDF ---
    norm_factor = jnp.trapezoid(rho_miss_eff, zgrid)
    norm_factor = jnp.where(norm_factor > 0.0, norm_factor, 1.0)
    p_miss_grid = rho_miss_eff / norm_factor

    return f, jnp.interp(z, zgrid, p_miss_grid), jnp.interp(z, zgrid, C_eff)


# _CompletionGrids is a NamedTuple (pytree); in_axes=None broadcasts
# the whole bundle as a constant across all (z, pix) elements.
_catalog_completion_inner_vmap = vmap(
    _catalog_completion_inner,
    in_axes=(0, 0, None, None, None),
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
    prefer ``catalog_completion_vmap``, which computes the grid bundle
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
    grids = _precompute_grids(cosmo, survey, em_catalog)
    return _catalog_completion_inner(z, pix, grids, survey, em_catalog)


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

    Computes the grid bundle (``dN_exp_smooth`` and ``pvol``) once before
    the vmap so both are shared across all (z, pix) evaluations.

    Parameters
    ----------
    z : jnp.ndarray, shape (N,)
    pix : jnp.ndarray, shape (N,)
    cosmo, survey, em_catalog : as in ``catalog_completion``

    Returns
    -------
    f, p_miss, C : jnp.ndarray, each shape (N,)
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    return _catalog_completion_inner_vmap(z, pix, grids, survey, em_catalog)


# Initialise the module-level kernel matrix now that zgrid is available.
_K_SMOOTH = _build_kernel()