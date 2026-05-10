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

from gwdistributions.distributions.spin import IsotropicUniformMagnitudeChiEffGivenComponentMass

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in arctanh")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

spin_prior = IsotropicUniformMagnitudeChiEffGivenComponentMass()
spin_prior._init_values(max_spin_magnitude=0.99)

def load_gw_samples(gw_path):
    """
    Load GW posterior samples from an HDF5 file and return flattened arrays
    of length (nEvents * nsamp).

    Parameters
    ----------
    gw_path : str
        Path to HDF5 file containing PE samples.

    Returns
    -------
    m1det, m2det, dL, chieff, ra, dec, p_pe : jnp.ndarray
        Flattened arrays of length (nEvents * nsamp).
    nEvents : int
        Number of GW events.
    """

    with h5py.File(gw_path, "r") as f:
        nsamp = int(f.attrs["nsamp"])
        nEvents = int(f.attrs["nobs"])

        # Load arrays as NumPy first (safer for reshaping)
        ra     = np.array(f["ra"])
        dec    = np.array(f["dec"])
        m1det  = np.array(f["m1det"])
        m2det  = np.array(f["m2det"])
        dL     = np.array(f["dL"]) * u.Mpc
        dL     = dL.value  # convert to float Mpc
        chieff = np.array(f["chieff"]) if "chieff" in f else np.zeros(dL.shape)
        p_pe = np.array(f["p_pe"]) if "p_pe" in f else np.ones(dL.shape)
        
        is_mock = bool(f.attrs.get("mock_data", False))
    
    # Define cosmology to prevent NameError
    H0Planck = Planck15.H0.value
    Om0Planck = Planck15.Om0

    redshift = z_of_dL(dL, H0Planck, Om0Planck)
    m1source = m1det/(1+redshift)
    m2source = m2det/(1+redshift)
    
    # ------------------------------------------------------------
    # p_pe handling
    # ------------------------------------------------------------
    if is_mock:
        print("This is using mock data.")
    else:
        p_pe_chieff = np.exp(spin_prior._logprob(chieff, m1source, m2source, 0.99))
        p_pe = p_pe * p_pe_chieff

    # Normalise per event so that each event's importance weights are
    # independent.  The per-event marginal likelihood is
    #   (1/nsamp) Σ_j  p_pop(θ_j) / p_pe(θ_j)
    # and dividing by the per-event sum makes the effective weights
    # dimensionless while preserving the correct relative scale within
    # each event.  Global normalisation (over nEvents*nsamp) would
    # introduce a factor of nEvents into every per-event sum, biasing
    # log μ and therefore the posterior on H0.
    p_pe = p_pe.reshape(nEvents, nsamp)
    p_pe = p_pe / p_pe.sum(axis=1, keepdims=True)
    p_pe = p_pe.flatten()
    
    # Convert to jnp in requested order: m1det, m2det, dL, chieff, ra, ...
    return (
        jnp.array(m1det),
        jnp.array(m2det),
        jnp.array(dL),
        jnp.array(chieff),
        jnp.array(ra),
        jnp.array(dec),
        jnp.array(p_pe),
        nEvents,
        nsamp
    )


def load_selection_samples(
    file,
    far_threshold=1.0,
    rng=None,
):
    """
    Return (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) for detected injections.

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
    chieffsels: jnp.ndarray
    rasels    : jnp.ndarray
    decsels   : jnp.ndarray
    pdraw_sel : jnp.ndarray
    ndraw     : int
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, "r") as f:

        # Branch 0 — mock selection file (generate_selection.py)
        if f.attrs.get("mock_data", False):
            print("This is using mock selection.")
            m1detsels  = np.array(f["m1detsels"][:])
            m2detsels  = np.array(f["m2detsels"][:])
            dLsels     = np.array(f["dLsels"][:])
            chieffsels = np.array(f["chieffsels"][:])
            rasels     = np.array(f["rasels"][:])
            decsels    = np.array(f["decsels"][:])
            pdraw_sel  = np.array(f["p_draw"][:])
            ndraw      = int(f.attrs["Ndraw"])

            n_det = len(m1detsels)
            print(f"    [mock selection] {n_det:,} injections  "
                  f"Ndraw={ndraw:,}  "
                  f"pop_model={f.attrs.get('pop_model', 'unknown')}")
            print(f"    p_draw: min={pdraw_sel.min():.3e}  "
                  f"max={pdraw_sel.max():.3e}  "
                  f"mean={pdraw_sel.mean():.3e}")

            return (
                jnp.array(m1detsels),
                jnp.array(m2detsels),
                jnp.array(dLsels),
                jnp.array(chieffsels),
                jnp.array(rasels),
                jnp.array(decsels),
                jnp.array(pdraw_sel),
                ndraw,
            )
        # ------------------------------------------------------------
        # Branch 1: "injections/..." format
        # ------------------------------------------------------------
        elif "injections" in f:
            m1det_all  = np.array(f["injections"]["mass1"][:])
            m2det_all  = np.array(f["injections"]["mass2"][:])
            dL_all     = np.array(f["injections"]["distance"][:])
            ra_all     = np.array(f["injections"]["right_ascension"][:])
            dec_all    = np.array(f["injections"]["declination"][:])
            s1z_all    = np.array(f["injections"]["spin1z"][:])
            s2z_all    = np.array(f["injections"]["spin2z"][:])
            
            # Cosmology for reference distribution
            H0Planck = Planck15.H0.value
            Om0Planck = Planck15.Om0

            z_all = z_of_dL(dL_all, H0Planck, Om0Planck)

            m1src_all = m1det_all / (1.0 + z_all)
            m2src_all = m2det_all / (1.0 + z_all)
            chieff_all = (m1src_all*s1z_all + m2src_all*s2z_all)/(m1src_all + m2src_all)        
            
            # Safely calculate 1D chi_eff draw probability (preventing -inf underflow)
            log_p_chi = spin_prior._logprob(chieff_all, m1src_all, m2src_all, 0.99)
            safe_log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
            p_chieff_draw = np.exp(safe_log_p_chi)

            # Load the joint PDF and the exact 3D spin PDFs
            p_joint = np.array(f["injections"]["sampling_pdf"][:])
            p_spin1 = np.array(f["injections"]["spin1x_spin1y_spin1z_sampling_pdf"][:])
            p_spin2 = np.array(f["injections"]["spin2x_spin2y_spin2z_sampling_pdf"][:])
            
            # Remove the 6D spin probability and replace it with the 1D chi_eff probability
            p_effective = (p_joint / (p_spin1 * p_spin2)) * p_chieff_draw

            # Convert to detector-frame variables
            pdraw_all = (
                p_effective
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
            
            # Extract all spin components needed for the 6D analytical prior
            s1x_all   = np.array(f["events"]["spin1x"][:])
            s1y_all   = np.array(f["events"]["spin1y"][:])
            s1z_all   = np.array(f["events"]["spin1z"][:])
            s2x_all   = np.array(f["events"]["spin2x"][:])
            s2y_all   = np.array(f["events"]["spin2y"][:])
            s2z_all   = np.array(f["events"]["spin2z"][:])
            
            chieff_all = (m1src_all*s1z_all + m2src_all*s2z_all)/(m1src_all + m2src_all)
            
            H0Planck = Planck15.H0.value
            Om0Planck = Planck15.Om0

            z_all = z_of_dL(dL_all, H0Planck, Om0Planck)
            m1det_all = m1src_all * (1.0 + z_all)
            m2det_all = m2src_all * (1.0 + z_all)

            weights = np.array(f["events"]["weights"][:])

            # Extract joint probability
            ln_pdraw_joint = np.array(
                f["events"]["lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"][:]
            )
            
            # Analytically compute the 6D spin log-probability
            a1 = np.sqrt(s1x_all**2 + s1y_all**2 + s1z_all**2)
            a2 = np.sqrt(s2x_all**2 + s2y_all**2 + s2z_all**2)
            
            # LVK standard prior: p(s1, s2) = 1 / (16 * pi^2 * a1^2 * a2^2 * a_max^2)
            ln_pdraw_spin = -np.log(16.0 * np.pi**2 * a1**2 * a2**2 * 0.99**2)
            
            # Safely calculate 1D chi_eff draw probability
            log_p_chi = spin_prior._logprob(chieff_all, m1src_all, m2src_all, 0.99)
            safe_log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
            
            # Swap the 6D spin probability for the 1D chi_eff probability in log-space
            ln_pdraw_effective = ln_pdraw_joint - ln_pdraw_spin + safe_log_p_chi

            pdraw_all = np.exp(ln_pdraw_effective) / (1.0 + z_all) ** 2 / ddL_of_z(
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
            raise RuntimeError("Unrecognized injection file format: "
                               "no 'mock_data' attr, 'injections', or 'events' group.")
    # ------------------------------------------------------------
    # Restrict to detected injections
    # ------------------------------------------------------------
    m1detsels  = m1det_all[detected]
    m2detsels  = m2det_all[detected]
    chieffsels = chieff_all[detected]
    dLsels     = dL_all[detected]
    rasels     = ra_all[detected]
    decsels    = dec_all[detected]
    pdraw_sel  = pdraw_all[detected]

    # The selection integral requires
    #
    #   μ = (1/N_draw) Σ_det  p_pop(d_i|λ) / p_draw(d_i)
    #
    # so p_draw must retain its physical scale (per unit volume per unit
    # mass per unit time).  The previous code normalised pdraw_wt to
    # sum=1, which removed that scale and rendered log μ parameter-
    # independent — effectively turning the selection correction into a
    # constant that cannot track changes in the population model.
    #
    # We keep pdraw_sel as-is (already in physical units after the
    # Jacobian and time corrections applied above) and do not renormalise.
    pdraw_wt = pdraw_sel
    print(f"    Selection samples: Ndet={len(pdraw_wt)}, Ndraw={ndraw}, "
          f"mean(p_draw)={pdraw_wt.mean():.3e}")

    # Convert to jnp in requested order: m1detsels, m2detsels, dLsels, chieffsels, rasels, ...
    return (
        jnp.array(m1detsels),
        jnp.array(m2detsels),
        jnp.array(dLsels),
        jnp.array(chieffsels),
        jnp.array(rasels),
        jnp.array(decsels),
        jnp.array(pdraw_wt),
        ndraw
    )