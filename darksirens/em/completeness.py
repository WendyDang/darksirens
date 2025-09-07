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
from jaxinterp2d import interp2d

from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *

from darksirens.gw.utils import load_gw_samples
from darksirens.em.utils import load_survey

from jax.scipy.special import expit


@jit
def Ngals_lessthanz(z,pix,zgals):
    Ngals = jnp.where((zgals[pix] < z), jnp.ones(len(zgals[pix])), 0).sum()
    return Ngals
#     zgals_shape = zgals.shape
#     zgals = zgals.flatten()
#     Ngals = jnp.where((zgals < z), jnp.ones(len(zgals)), 0).reshape(zgals_shape).sum(axis=-1)
#     return Ngals[pix]

Ngals_lessthanz_vmap = jit(vmap(Ngals_lessthanz, in_axes=(0,None,None), out_axes=0))

@partial(jax.jit, static_argnames=['apix'])
def Ngals_expected_lessthanz(z,H0,Om0,n0,delta,apix):
    dNexpected = n0*apix*dV_of_z(zgrid,H0,Om0)*(1+zgrid)**(delta-1)
    dNexpected = jnp.where((zgrid<z), dNexpected, 0)
    Nexpected = jnp.trapezoid(dNexpected,zgrid)
    return Nexpected

Ngals_expected_lessthanz_vmap = jit(vmap(Ngals_expected_lessthanz, in_axes=(0,None,None,None,None,None), out_axes=0))

@jit
def Pcomplete0(z,z1,z50):
    return expit(-z1*(z/z50)+z1)

@partial(jax.jit, static_argnames=['apix'])
def completeness_fraction(H0,Om0,n0,z1,z50,delta,z,pix,apix,zgals):
    Nexpected = 1+Ngals_expected_lessthanz_vmap(zgrid,H0,Om0,n0,delta,apix)
    Ngals = Ngals_lessthanz_vmap(zgrid,pix,zgals)
    ratio = Ngals/Nexpected

    ratio = jnp.where((ratio < 1), ratio, 0)
    ratio = jnp.where((ratio != 0), ratio, 1)
    ratio = ratio*Pcomplete0(zgrid,z1,z50)
    pvol = dV_of_z(zgrid, H0, Om0)*(1+zgrid)**(delta-1)

    V = jnp.trapezoid(ratio*pvol,zgrid)
    Vmax = jnp.trapezoid(pvol,zgrid)

    pmiss = (1-ratio)*pvol
    pmiss_normed = pmiss/jnp.trapezoid(pmiss,zgrid)

    pmiss_z = jnp.interp(z,zgrid,pmiss_normed)

    return V/Vmax, pmiss_z, ratio

completeness_fraction_vmap = jit(vmap(completeness_fraction, in_axes=(None,None,None,None,None,None,0,0,None,None), out_axes=0))

@jit
def logpcatalog(z, pix, H0, Om0, delta, zgals, dzgals, wgals):
    zs = zgals[pix] 
    ddzs = dzgals[pix]
    wts = wgals[pix]*dV_of_z(zs,H0,Om0)*(1+zs)**(delta-1)
    ngals = len(zs)
    wts = wts/jnp.sum(wts)
    return logsumexp(jnp.log(wts) + norm.logpdf(z,zs,ddzs))

logpcatalog_vmap = jit(vmap(logpcatalog, in_axes=(0,0,None,None,None,None,None,None), out_axes=0))


@partial(jax.jit, static_argnames=['apix'])
def logPriorUniverse_darksirens(z,pix,H0,Om0,n0,z1,z50,delta,gamma,apix,zgals,dzgals,wgals):
    f, pmiss, ratio = completeness_fraction_vmap(H0,Om0,n0,z1,z50,delta,z,pix,apix,zgals)

    logpmiss = jnp.nan_to_num(jnp.log(pmiss), -jnp.inf)

    logpcat = jnp.nan_to_num(logpcatalog_vmap(z, pix, H0, Om0, delta, zgals, dzgals, wgals), -jnp.inf)

    logprob = jnp.log( jnp.exp(jnp.log(f) + logpcat) + jnp.exp(jnp.log(1-f) + logpmiss) ) + (gamma-1)*jnp.log1p(z)

    return logprob


@partial(jax.jit, static_argnames=['apix'])
def logPriorUniverse_spectralsirens(z,pix,H0,Om0,n0,z1,z50,delta,gamma,apix,zgals,dzgals,wgals):
    f, pmiss, ratio = completeness_fraction_vmap(H0,Om0,n0,z1,z50,delta,z,pix,apix,zgals)
    
    f = 0
    logpmiss = jnp.nan_to_num(jnp.log(pmiss), -jnp.inf)

    logpcat = jnp.nan_to_num(logpcatalog_vmap(z, pix, H0, Om0, delta, zgals, dzgals, wgals), -jnp.inf)

    logprob = jnp.log( jnp.exp(jnp.log(f) + logpcat) + jnp.exp(jnp.log(1-f) + logpmiss) ) + (gamma-1)*jnp.log1p(z)

    return logprob


@partial(jax.jit, static_argnames=['apix'])
def logPriorUniverse_spectralsirens_fast(z,pix,H0,Om0,n0,z1,z50,delta,gamma,apix,zgals,dzgals,wgals):

    pvol = dV_of_z(zgrid, H0, Om0)*(1+zgrid)**(gamma-1)
    pvol = pvol/jnp.trapezoid(pvol,zgrid)
    logpvol = jnp.log(pvol)
    logp = jnp.interp(z,zgrid,logpvol)

    return logp


def universe_model_parser(universe_model='darksirens'):
    
    if universe_model=='darksirens':
        logp = logPriorUniverse_darksirens

    if universe_model=='spectral_sirens':
        logp = logPriorUniverse_spectralsirens
        

    if universe_model=='spectral_sirens_fast':
        logp = logPriorUniverse_spectralsirens_fast

    return logp