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
from jax.scipy.stats import norm

from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *

mass = jnp.linspace(1, 150, 2000)
mass_ratio =  jnp.linspace(0, 1, 2000)

@jit
def dV_of_z_normed(z,Om0,gamma):
    dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
    prob = dV/jnp.trapezoid(dV,zgrid)
    return jnp.interp(z,zgrid,prob)

@jit
def logpm1_peak(m1,mu,sigma):
    pm1 =  jnp.exp(-(mass - mu)**2 / (2 * sigma ** 2))
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logfq(m1,m2,beta):
    q = m2/m1
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)

    log_pq = jnp.log(jnp.interp(q,mass_ratio,pq))

    return log_pq

@jit
def log_p_pop_pl_pl(m1,m2,mu,sigma,beta):
    log_dNdm1 = logpm1_peak(m1,mu,sigma)
    log_dNdm2 = logpm1_peak(m2,mu,sigma)
    log_fq = logfq(m1,m2,beta)

    return log_dNdm1 + log_dNdm2 + log_fq