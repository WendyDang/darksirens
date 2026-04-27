# darksirens/utils/containers.py
from typing import NamedTuple, Any

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
    delta_g_pix_z: Any

# (Optional but helpful) A container for GW Event data
class GWEvent(NamedTuple):
    """Data for a single Gravitational Wave event."""
    z: Any        # Event redshift
    # You can add m1, q, chieff, dl, etc. here later!