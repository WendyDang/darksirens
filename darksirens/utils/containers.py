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
    alpha: Any

class EMCatalog(NamedTuple):
    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    delta_g_pix_z: Any
    sigma_kernel: Any

class GWEvent(NamedTuple):
    """
    JAX-compatible PyTree container for Gravitational Wave Parameter Estimation (PE) samples.
    Supports either a single event or a stacked batch of multiple events.
    """
    m1det: Any      # Primary mass in the detector frame [M_sun]
    m2det: Any      # Secondary mass in the detector frame [M_sun]
    dL: Any         # Luminosity distance [Mpc]
    chieff: Any     # Effective inspiral spin parameter
    prior_wt: Any    # PE prior/weights evaluated at the samples
    pixels: Any     # HEALPix pixel indices corresponding to the sky location

    @property
    def q(self):
        """Mass ratio (m2 / m1). Guaranteed to be <= 1 if m1 >= m2."""
        return self.m2det / self.m1det

    @property
    def chirp_mass(self):
        """Detector-frame chirp mass."""
        return (self.m1det * self.m2det)**(3/5) / (self.m1det + self.m2det)**(1/5)