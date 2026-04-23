# data.py

import jax.numpy as jnp
import healpy as hp

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey


def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data. 
    Handles cases where survey_path might be None (non-dark sirens models).
    """

    # 1. Initialize survey variables as None/defaults
    nside = None
    zgals = dzgals = wgals = None
    zgals_pe = dzgals_pe = wgals_pe = None
    zgals_sel = dzgals_sel = wgals_sel = None
    apix = 0.0

    # 2. Load Survey only if path is provided
    if opts.survey_path is not None:
        nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)
        apix = hp.nside2pixarea(nside)
    else:
        # If no survey, we might still need a default nside for 
        # pixelization logic in other parts of the code
        nside = 1 

    # 3. Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents = load_gw_samples(
        opts.gw_path, nsamp=opts.nsamp
    )

    # 4. Load Selection samples (Always required)
    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = load_selection_samples(opts.gwselection_path)

    # 5. Pixel indexing and Galaxy lookups
    # Only perform these if a survey was actually loaded
    pixels_pe = hp.ang2pix(nside, jnp.pi/2 - dec, ra)
    pixels_sel = hp.ang2pix(nside, jnp.pi/2 - decsels, rasels)

    if zgals is not None:
        zgals_pe = zgals[pixels_pe]
        dzgals_pe = dzgals[pixels_pe]
        wgals_pe = wgals[pixels_pe]

        zgals_sel = zgals[pixels_sel]
        dzgals_sel = dzgals[pixels_sel]
        wgals_sel = wgals[pixels_sel]

    # 6. Pack into dictionary
    data = dict(
        # GW PE samples
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        p_pe=p_pe,
        pixels_pe=jnp.asarray(pixels_pe),
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,

        # Selection samples
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        p_draw=p_draw,
        pixels_sel=jnp.asarray(pixels_sel),
        zgals_sel=zgals_sel,
        dzgals_sel=dzgals_sel,
        wgals_sel=wgals_sel,

        # Survey metadata
        nEvents=nEvents,
        Ndraw=Ndraw,
        apix=apix,
        nside=nside,
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
    )

    return data