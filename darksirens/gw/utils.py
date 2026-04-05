import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

import astropy
import numpy as np
import healpy as hp

import h5py
import astropy.units as u

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *

def load_gw_samples(gw_path, nsamp=64):
    """
    Load GW posterior samples from an HDF5 file and return flattened arrays
    of length (nEvents * nsamp).

    Parameters
    ----------
    gw_path : str
        Path to HDF5 file containing PE samples.
    nsamp : int
        Number of posterior samples per event to keep.

    Returns
    -------
    m1det, m2det, dL, ra, dec, p_pe : jnp.ndarray
        Flattened arrays of length (nEvents * nsamp).
    nEvents : int
        Number of GW events.
    """

    with h5py.File(gw_path, "r") as f:
        nsamps_file = int(f.attrs["nsamp"])
        nEvents = int(f.attrs["nobs"])

        # Load arrays as NumPy first (safer for reshaping)
        ra     = np.array(f["ra"])
        dec    = np.array(f["dec"])
        m1det  = np.array(f["m1det"])
        m2det  = np.array(f["m2det"])
        dL     = np.array(f["dL"]) * u.Mpc
        dL     = dL.value  # convert to float Mpc

        # Optional PE weights
        p_pe = np.array(f["p_pe"]) if "p_pe" in f else None

    # ------------------------------------------------------------
    # Reshape to (nEvents, nsamps_file)
    # ------------------------------------------------------------
    ra    = ra.reshape(nEvents, nsamps_file)
    dec   = dec.reshape(nEvents, nsamps_file)
    m1det = m1det.reshape(nEvents, nsamps_file)
    m2det = m2det.reshape(nEvents, nsamps_file)
    dL    = dL.reshape(nEvents, nsamps_file)

    # ------------------------------------------------------------
    # Truncate to requested nsamp
    # ------------------------------------------------------------
    if nsamp > nsamps_file:
        raise ValueError(
            f"Requested nsamp={nsamp}, but file only contains nsamp={nsamps_file}"
        )

    ra    = ra[:, :nsamp]
    dec   = dec[:, :nsamp]
    m1det = m1det[:, :nsamp]
    m2det = m2det[:, :nsamp]
    dL    = dL[:, :nsamp]

    # ------------------------------------------------------------
    # Flatten to (nEvents * nsamp,)
    # ------------------------------------------------------------
    ra    = ra.flatten()
    dec   = dec.flatten()
    m1det = m1det.flatten()
    m2det = m2det.flatten()
    dL    = dL.flatten()

    # ------------------------------------------------------------
    # p_pe handling
    # ------------------------------------------------------------
    if p_pe is None:
        p_pe = dL**2
    else:
        p_pe = np.array(p_pe).reshape(nEvents, nsamps_file)[:, :nsamp].flatten()

    # Convert to jnp
    return (
        jnp.array(m1det),
        jnp.array(m2det),
        jnp.array(dL),
        jnp.array(ra),
        jnp.array(dec),
        jnp.array(p_pe),
        nEvents,
    )


def load_selection_samples(
    file,
    far_threshold=1.0,
    rng=None,
):
    """
    Return (m1det, m2det, dL, ra, dec, pdraw, ndraw) for detected injections.

    - ndraw is the total number of generated injections (accepted + rejected).
    - If nsamp is not None, a subsample of detected injections is drawn with
      proper importance weighting so that ndraw stays the same and pdraw
      is corrected for the sampling probability.

    Parameters
    ----------
    file : str
        Path to injection file.
    nsamp : int or None
        Number of detected injections to return. If None, return all detected.
    far_threshold : float
        FAR threshold (per year) for detection.
    rng : np.random.Generator or None
        RNG for subsampling. If None, a new default_rng() is created.

    Returns
    -------
    m1detsels : jnp.ndarray
    m2detsels : jnp.ndarray
    dLsels    : jnp.ndarray
    rasels    : jnp.ndarray
    decsels   : jnp.ndarray
    pdraw_sel : jnp.ndarray
    ndraw     : int
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, "r") as f:
        # ------------------------------------------------------------
        # Branch 1: "injections/..." format
        # ------------------------------------------------------------
        if "injections" in f:
            m1det_all = np.array(f["injections"]["mass1"][:])
            m2det_all = np.array(f["injections"]["mass2"][:])
            dL_all    = np.array(f["injections"]["distance"][:])
            ra_all    = np.array(f["injections"]["right_ascension"][:])
            dec_all   = np.array(f["injections"]["declination"][:])

            # Cosmology for reference distribution
            H0Planck = Planck15.H0.value
            Om0Planck = Planck15.Om0

            z_all = z_of_dL(dL_all, H0Planck, Om0Planck)

            # Reference sampling PDF in (m1_source, m2_source, z)
            m1src_all = m1det_all / (1.0 + z_all)
            m2src_all = m2det_all / (1.0 + z_all)

            p_m1m2 = np.array(
                f["injections"]["mass1_source_mass2_source_sampling_pdf"][:]
            )
            p_z = np.array(f["injections"]["redshift_sampling_pdf"][:])

            # pdraw in detector-frame variables (Farr 2019 style)
            pdraw_all = (
                p_m1m2 * p_z
                / (1.0 + z_all) ** 2
                / ddL_of_z(z_all, dL_all, H0Planck, Om0Planck)
            )

            # FAR-based detection
            pycbc_far    = np.array(f["injections"]["far_pycbc_hyperbank"])
            pycbc_bbh_far = np.array(f["injections"]["far_pycbc_bbh"])
            gstlal_far   = np.array(f["injections"]["far_gstlal"])
            mbta_far     = np.array(f["injections"]["far_mbta"])

            detected = (
                (pycbc_far < far_threshold)
                | (pycbc_bbh_far < far_threshold)
                | (gstlal_far < far_threshold)
                | (mbta_far < far_threshold)
            )

            ndraw = int(f.attrs["n_accepted"] + f.attrs["n_rejected"])

            T = (f.attrs["end_time_s"] - f.attrs["start_time_s"]) / (
                3600.0 * 24.0 * 365.25
            )
            pdraw_all /= T

        # ------------------------------------------------------------
        # Branch 2: "events/..." format
        # ------------------------------------------------------------
        elif "events" in f:
            m1src_all = np.array(f["events"]["mass1_source"][:])
            m2src_all = np.array(f["events"]["mass2_source"][:])
            dL_all    = np.array(f["events"]["luminosity_distance"][:])
            ra_all    = np.array(f["events"]["right_ascension"][:])
            dec_all   = np.array(f["events"]["declination"][:])

            H0Planck = Planck15.H0.value
            Om0Planck = Planck15.Om0

            z_all = z_of_dL(dL_all, H0Planck, Om0Planck)
            m1det_all = m1src_all * (1.0 + z_all)
            m2det_all = m2src_all * (1.0 + z_all)

            weights = np.array(f["events"]["weights"][:])

            ln_pdraw = np.array(
                f[
                    "events"]["lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
                ][:]
            )
            pdraw_all = np.exp(ln_pdraw) / (1.0 + z_all) ** 2 / ddL_of_z(
                z_all, dL_all, H0Planck, Om0Planck
            )

            far_all = np.min(
                [np.array(f["events"]["%s_far" % s][:]) for s in f.attrs["searches"]],
                axis=0,
            )

            ndraw = int(f.attrs["total_generated"])

            T = f.attrs["total_analysis_time"] / (3600.0 * 24.0 * 365.25)
            pdraw_all /= T
            pdraw_all /= weights

            detected = far_all < far_threshold

        else:
            raise RuntimeError("Unrecognized injection file format: no 'injections' or 'events' group.")

    # ------------------------------------------------------------
    # Restrict to detected injections
    # ------------------------------------------------------------
    m1detsels = m1det_all[detected]
    m2detsels = m2det_all[detected]
    dLsels    = dL_all[detected]
    rasels    = ra_all[detected]
    decsels   = dec_all[detected]
    pdraw_sel = pdraw_all[detected]

    Ndet = len(m1detsels)
    
    pop_wt = pdraw_sel
    unnorm_wt = pop_wt/pdraw_sel
    sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
    pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)
    
    print(pdraw_wt.shape, ndraw, pdraw_wt.sum())

    return (
        jnp.array(m1detsels),
        jnp.array(m2detsels),
        jnp.array(dLsels),
        jnp.array(rasels),
        jnp.array(decsels),
        jnp.array(pdraw_wt),
        ndraw
    )

