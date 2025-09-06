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

def load_survey(survey_path, dz=0.001):
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = jnp.asarray(f['zgals'])
        ngals = jnp.asarray(f['ngals'])
#     try: 
        dzgals = jnp.asarray(f['dzgals'])
        wgals = jnp.asarray(f['wgals'])
#     except:
#         print('No dzs or wts, loading default')
#         dzgals = dz*(1+zgals)
#         wgals = jnp.ones(zgals.shape)
    return nside, ngals, zgals, dzgals, wgals