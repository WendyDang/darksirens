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
    with h5py.File(gw_path, 'r') as inp:
        nsamps = inp.attrs['nsamp']
        nEvents = inp.attrs['nobs']
        ra = jnp.array(inp['ra'])
        dec = jnp.array(inp['dec'])
        m1det = jnp.array(inp['m1det'])
        m2det = jnp.array(inp['m2det'])
        dL = jnp.array((jnp.array(inp['dL'])*u.Mpc).value)

    try:
        ra = ra.reshape(nEvents,nsamps)[0:nEvents,0:nsamp]
        dec = dec.reshape(nEvents,nsamps)[0:nEvents,0:nsamp]
        m1det = m1det.reshape(nEvents,nsamps)[0:nEvents,0:nsamp]
        m2det = m2det.reshape(nEvents,nsamps)[0:nEvents,0:nsamp]
        dL = dL.reshape(nEvents,nsamps)[0:nEvents,0:nsamp]
    except:
        ra = ra.reshape(nEvents,nsamps)[0:nEvents,0:nsamps]
        dec = dec.reshape(nEvents,nsamps)[0:nEvents,0:nsamps]
        m1det = m1det.reshape(nEvents,nsamps)[0:nEvents,0:nsamps]
        m2det = m2det.reshape(nEvents,nsamps)[0:nEvents,0:nsamps]
        dL = dL.reshape(nEvents,nsamps)[0:nEvents,0:nsamps]        

    ra = ra[0:nEvents].flatten()
    dec = dec[0:nEvents].flatten()
    m1det = m1det[0:nEvents].flatten()
    m2det = m2det[0:nEvents].flatten()
    dL = dL[0:nEvents].flatten()
    
    try:
        p_pe = jnp.array(inp['p_pe'])
    except:
        p_pe = dL**2
    
    return m1det, m2det, dL, ra, dec, p_pe, int(nEvents)

def load_selection_samples(file, nsamp=None, desired_pop_wt=None, far_threshold=1, rng=None):
    """Return `(m1, q, z, pdraw, nsel)` to estimate selection effects.
    
    :param file: The injection file.

    :param nsamp: The number of samples to be returned.

    :param desired_pop_wt: Function giving a weight in `(m1, q, z)` from which
        the population of injections should be drawn.  If none is given, the
        reference distribution for the actual injections will be used; otherwise
        the distribution of injections will be re-weighted to achieve the
        desired poplation.

    :param far_threshold: The threshold on the FAR (per year) at which an
        injection is considered detected.

    :param rng: A random number generator for the draws; if `None`, one will be
        initialized randomly.

    :return: A tuple `(m1, q, z, pdraw, nsel)`, giving a draw of detected
        injections from the desired population.  `pdraw` is properly normalized
        for estimating detectability as in, e.g., [Farr
        (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract).
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, 'r') as f:
        try:
            m1detsels = np.array(f['injections/mass1'][:])
            m2detsels = np.array(f['injections/mass2'][:])
            dLsels = np.array(f['injections/distance'][:])
            rasels = f['injections/right_ascension'][:]
            decsels = f['injections/declination'][:]
            zsels = z_of_dL(dLsels,H0Planck,Om0Planck)
            pdraw_sel = np.array(f['injections/mass1_source_mass2_source_sampling_pdf'])*np.array(f['injections/redshift_sampling_pdf'])/(1+zsels)**2/ddL_of_z(zsels,dLsels,H0Planck,Om0Planck)

            pycbc_far = np.array(f['injections/far_pycbc_hyperbank'])
            pycbc_bbh_far = np.array(f['injections/far_pycbc_bbh'])
            gstlal_far = np.array(f['injections/far_gstlal'])
            mbta_far = np.array(f['injections/far_mbta'])

            detected = (pycbc_far < far_threshold) | (pycbc_bbh_far < far_threshold) | (gstlal_far < far_threshold) | (mbta_far < far_threshold)

            ndraw = f.attrs['n_accepted'] + f.attrs['n_rejected']

            T = (f.attrs['end_time_s'] - f.attrs['start_time_s'])/(3600.0*24.0*365.25) 
            pdraw_sel /= T

        except:
            m1sels = np.array(f['events']['mass1_source'][:])
            m2sels = np.array(f['events']['mass2_source'][:])
            dLsels = np.array(f['events']['luminosity_distance'][:])
            rasels = f['events']['right_ascension'][:]
            decsels = f['events']['declination'][:]

            zsels = z_of_dL(dLsels,H0Planck,Om0Planck)
            m1detsels = m1sels*(1+zsels)
            m2detsels = m2sels*(1+zsels)

            weights = f['events']['weights'][:]

            pdraw_sel = np.exp(f['events']['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'][:])/(1+zsels)**2/ddL_of_z(zsels,dLsels,H0Planck,Om0Planck)

            far = np.min([f['events']['%s_far'%search][:] for search in f.attrs['searches']], axis=0)

            ndraw = f.attrs['total_generated'] 

            T = (f.attrs['total_analysis_time'])/(3600.0*24.0*365.25) 
            pdraw_sel /= T
            pdraw_sel /= weights

            far_thr = 1

            detected = (far < far_thr)

        m1detsels = m1detsels[detected]
        m2detsels = m2detsels[detected]
        dLsels = dLsels[detected]
        pdraw_sel = pdraw_sel[detected]
        rasels = rasels[detected]
        decsels = decsels[detected]
        print(len(m1detsels))

        pop_wt = pdraw_sel

        unnorm_wt = pop_wt/pdraw_sel
        sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
        pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)

        inds = rng.choice(len(m1detsels), size=nsamp, p=sum_norm_wt)
        m1detsels_cut = m1detsels[inds]
        m2detsels_cut = m2detsels[inds]
        dLsels_cut = dLsels[inds]
        rasels_cut = rasels[inds]
        decsels_cut = decsels[inds]

        pdraw_sel_cut = pdraw_wt[inds]
        ndraw_cut = nsamp
        
            
        return jnp.array(m1detsels_cut), jnp.array(m2detsels_cut), jnp.array(dLsels_cut), jnp.array(rasels_cut), jnp.array(decsels_cut), jnp.array(pdraw_sel_cut), int(ndraw_cut)
