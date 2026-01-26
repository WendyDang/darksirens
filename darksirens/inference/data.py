# data.py

import jax.numpy as jnp
import healpy as hp

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey


def load_all_data(opts):
    """
    Loads all survey, GW posterior, and selection data.
    Returns a dictionary ready to be passed into make_likelihood().
    """

    # --------------------------------------------------------
    # Survey data
    # --------------------------------------------------------
    nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)

    # --------------------------------------------------------
    # GW posterior samples
    # --------------------------------------------------------
    m1det, m2det, dL, ra, dec, p_pe, nEvents = load_gw_samples(
        opts.gw_path, nsamp=opts.nsamp
    )

    # --------------------------------------------------------
    # Selection samples
    # --------------------------------------------------------
    (
        m1detsels,
        m2detsels,
        dLsels,
        rasels,
        decsels,
        p_draw,
        Ndraw,
    ) = load_selection_samples(opts.gwselection_path)

    # --------------------------------------------------------
    # Pixel indexing
    # --------------------------------------------------------
    npix = hp.nside2npix(nside)
    apix = hp.nside2pixarea(nside)

    pixels_pe = hp.ang2pix(nside, jnp.pi/2 - dec, ra)
    pixels_sel = hp.ang2pix(nside, jnp.pi/2 - decsels, rasels)

    # --------------------------------------------------------
    # Galaxy redshift arrays for PE and selection samples
    # --------------------------------------------------------
    zgals_pe = zgals[pixels_pe]
    dzgals_pe = dzgals[pixels_pe]
    wgals_pe = wgals[pixels_pe]

    zgals_sel = zgals[pixels_sel]
    dzgals_sel = dzgals[pixels_sel]
    wgals_sel = wgals[pixels_sel]

    # --------------------------------------------------------
    # Pack everything into a dictionary
    # --------------------------------------------------------
    data = dict(
        # GW PE samples
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        p_pe=p_pe,
        pixels_pe=jnp.asarray(pixels_pe),
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,

        # Selection samples
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
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
