import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

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
        p_pe = jnp.ones(len(dL))
    
    return ra, dec, m1det, m2det, dL, p_pe, nEvents