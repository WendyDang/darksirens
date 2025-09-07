import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond
from functools import partial

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
from jax.scipy.stats import norm
from darksirens.gw.populations import pop_model_parser
from darksirens.em.completeness import universe_model_parser
from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *


@partial(jax.jit, static_argnames=['nEvents', 'Ndraw', 'nsamp', 'apix', 'pop_model', 'universe_model'])
def darksiren_log_likelihood(cosmo_params, survey_params, pop_params,
                             m1det, m2det, dL, ra, dec, p_pe, samples_ind,
                             m1detsels, m2detsels, dLsels, rasels, decsels, p_draw, selsamples_ind,
                             nEvents, nsamp, Ndraw, apix, zgals, dzgals, wgals, pop_model, universe_model):
    
    log_p_pop = pop_model_parser(pop_model=pop_model)
    
    logPriorUniverse = universe_model_parser(universe_model=universe_model)

    H0, Om0 = cosmo_params
    log10n0, z1, z50, delta, gamma = survey_params
    
    n0 = 10**log10n0

    zsels = z_of_dL(dLsels, H0, Om0)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)

    log_det_weights = log_p_pop(m1sels,m2sels,*pop_params)

    log_det_weights += - jnp.log(ddL_of_z(zsels,dLsels,H0,Om0)) - jnp.log(p_draw) - 2*jnp.log1p(zsels) + logPriorUniverse(zsels,selsamples_ind,H0,Om0,n0,z1,z50,delta,gamma,apix,zgals,dzgals,wgals)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * nEvents), ll, 0)
    ll += -nEvents*log_mu + nEvents*(3 + nEvents)/(2*Neff)

    z = z_of_dL(dL, H0, Om0)
    m1 = m1det/(1+z)
    m2 = m2det/(1+z)

    log_weights = log_p_pop(m1,m2,*pop_params)

    log_weights += - jnp.log(ddL_of_z(z,dL,H0,Om0)) - jnp.log(p_pe) - 2*jnp.log1p(z) + logPriorUniverse(z,samples_ind,H0,Om0,n0,z1,z50,delta,gamma,apix,zgals,dzgals,wgals)

    log_weights = log_weights.reshape((nEvents,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights,axis=-1))

    return jnp.nan_to_num(ll,nan=-jnp.inf)

