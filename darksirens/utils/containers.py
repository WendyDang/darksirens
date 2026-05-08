# darksirens/utils/containers.py
from typing import NamedTuple, Any
import jax.numpy as jnp


class CosmoParams(NamedTuple):
    """Cosmological parameters for the background universe."""
    H0: Any
    Om0: Any


class SurveyParams(NamedTuple):
    """Parameters dictating galaxy survey completeness and selection."""
    n0: Any
    z50: Any
    w: Any
    delta: Any
    b_miss: Any
    alpha_miss: Any


class EMCatalog(NamedTuple):
    """
    EM galaxy catalog and precomputed grids for the redshift prior.

    Fields
    ------
    apix : float
        Solid angle per HEALPix pixel [sr].
    zgals : (N_pix, N_max_gals)
        Padded galaxy redshifts per pixel.
    dzgals : (N_pix, N_max_gals)
        Padded galaxy photo-z uncertainties.
    wgals : (N_pix, N_max_gals)
        Padded galaxy base weights (luminosity / completeness).
    ngals : (N_pix,)
        Number of real (non-padded) galaxies per pixel.
    delta_g_pix_z : (N_pix, N_grid)
        LSS overdensity field pre-computed on the redshift grid.
    sigma_kernel : float
        KDE bandwidth for the catalog prior [redshift units].
    dN_obs_kde : (N_unique_pix, N_grid) or None
        Precomputed per-pixel KDE grids for dN_obs/dz.
        None until ``build_pixel_kde_cache`` is called.
        If None, ``_catalog_completion_inner`` recomputes on the fly
        (correct but slower — only for backward compatibility).
    pixel_to_cache_idx : (N_pix_catalog,) or None
        Maps HEALPix pixel → row in ``dN_obs_kde``.
        Pixels not covered by the cache map to 0 (never visited
        during inference, so the value is immaterial).
    """
    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    delta_g_pix_z: Any
    sigma_kernel: Any
    dN_obs_kde: Any            # (N_unique_pix, N_grid) | None
    pixel_to_cache_idx: Any    # (N_pix_catalog,)       | None


class GWEvent(NamedTuple):
    """
    JAX-compatible PyTree container for Gravitational Wave Parameter Estimation (PE) samples.
    Supports either a single event or a stacked batch of multiple events.

    Notes
    -----
    Do NOT construct directly — use ``darksirens.inference.events.make_gw_event``,
    which applies ``lax.optimization_barrier`` to every field and pre-computes ``q``
    so it is never recomputed inside a vmap or ``lax.scan`` hot path.
    """
    m1det: Any      # Primary mass in the detector frame [M_sun]
    m2det: Any      # Secondary mass in the detector frame [M_sun]
    dL: Any         # Luminosity distance [Mpc]
    chieff: Any     # Effective inspiral spin parameter
    prior_wt: Any   # PE prior weights evaluated at the samples
    pixels: Any     # HEALPix pixel indices corresponding to the sky location
    q: Any          # Mass ratio m2det/m1det — stored at construction, never recomputed

    @property
    def chirp_mass(self):
        """Detector-frame chirp mass."""
        return (self.m1det * self.m2det) ** (3 / 5) / (self.m1det + self.m2det) ** (1 / 5)