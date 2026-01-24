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
from jax.scipy.stats import norm
from jax.nn import log_sigmoid

from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *

from jax.scipy.stats import norm

mass = jnp.linspace(1, 250, 2000)
mass_ratio =  jnp.linspace(0, 1, 2000)
chieff_grid = jnp.linspace(-1, 1, 2000)

def Sfilter_low(m,m_min,dm_min):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    
    S_filter = 1./(f(m-m_min,dm_min) + 1.)
    S_filter = jnp.where(m<m_min+dm_min,S_filter,1.)
    S_filter = jnp.where(m>m_min,S_filter,0.)
    return S_filter

def Sfilter_high(m,m_max,dm_max):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    
    S_filter = 1./(f(m-m_max,-dm_max) + 1.)
    S_filter = jnp.where(m>m_max-dm_max,S_filter,1.)
    S_filter = jnp.where(m<m_max,S_filter,0.)
    return S_filter

def logpchieff(chieff,mu_chieff,sigma_chieff):
    pchieff =  jnp.exp(-(chieff - mu_chieff)**2 / (2 * sigma_chieff ** 2))/jnp.sqrt(2*jnp.pi*sigma_chieff**2)
    return jnp.log(pchieff)


@jit
def logpm1m2_plpeak_massratio(
    m1, q,
    m_min_1, m_max_1,
    alpha_1, dm_min_1,
    beta, mu, sigma,
    f
):
    alpha_1 = -alpha_1
    # --- p(m1): Power-law component ---
    norm_pl = (m_max_1**(1. + alpha_1) - m_min_1**(1. + alpha_1))
    p_m1_pl = (1. + alpha_1) * m1**alpha_1 / norm_pl

    # Mask out-of-range m1
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    # --- p(m1): Peak component ---
    p_m1_peak = jnp.exp(-0.5 * (m1 - mu)**2 / sigma**2) / jnp.sqrt(2. * jnp.pi * sigma**2)

    # Mixture
    p_m1 = Sfilter_low(m1,m_min_1,dm_min_1)*(f * p_m1_peak + (1. - f) * p_m1_pl)

    # --- p(q | m1): mass-ratio power law ---
    q_min = m_min_1/m1
    denom = 1 - q_min**(1. + beta)
    p_q = Sfilter_low(q*m1,m_min_1,dm_min_1) * (1. + beta) * q**beta / denom

    # Enforce m2 >= m_min_1
    p_q = jnp.where(q*m1 < m_min_1, 0.0, p_q)

    # --- log joint ---
    return jnp.log(p_m1) + jnp.log(p_q)

@jit
def log_p_pop_powerlaw_peak(m1, q, alpha_1, beta, m_min_1, m_max_1, dm_min_1, mu, sigma, f):
    log_dNdm1dq = logpm1m2_plpeak_massratio(m1, q, m_min_1, m_max_1, alpha_1, dm_min_1, beta, mu, sigma, f)
    #log_pchieff = logpchieff(chieff,mu_s1,sigma_s1)

    log_p_sz = np.log(0.25) # 1/2 for each spin dimension

    return log_p_sz + log_dNdm1dq #+ log_pchieff

def pop_model_parser(pop_model='powerlaw+peak'):
    
    if pop_model=='powerlaw+peak':
        log_p_pop = log_p_pop_powerlaw_peak
        
    if pop_model=='brokenpowerlaw+2peaks':
        log_p_pop = log_p_pop_brokenpowerlaw_2peaks

    if pop_model=='mock_data':
        log_p_pop = log_p_pop_mock_data

    return log_p_pop


def pop_model_prior_parser(pop_model='powerlaw+peak'):
    
    if pop_model=='powerlaw+peak':
        
        H0_lo = 20
        H0_hi = 140

        Om0_lo = Om0grid[0]
        Om0_hi = Om0grid[-1]
        
        log10n0_lo = -8.0
        log10n0_hi = 0.0

        z1_lo = 1
        z1_hi = 200

        z50_lo = 0
        z50_hi = 1
        
        delta_lo = -10
        delta_hi = 10
        
        gamma_low = -10
        gamma_high = 10

        m_min_1_low = 2
        m_min_1_high = 10

        m_max_1_low = 30
        m_max_1_high = 100

        alpha_1_low = -4
        alpha_1_high = 12

        dm_min_1_low = 0
        dm_min_1_high = 10

        beta_low = -2
        beta_high = 7

        mu_low = 20
        mu_high = 50

        sigma_low = 1
        sigma_high = 10

        f1_low = 0
        f1_high = 1

        f2_low = 0
        f2_high = 1

        mu_s1_low = -1
        mu_s1_high = 1

        sigma_s_low = 0.05
        sigma_s_high = 1


        lower_bound = [alpha_1_low, beta_low, m_min_1_low, m_max_1_low, dm_min_1_low, mu_low, sigma_low, f1_low]
        
        
        upper_bound = [alpha_1_high, beta_high, m_min_1_high, m_max_1_high, dm_min_1_high, mu_high, sigma_high, f1_high]
        
        labels = [r'$\alpha$',r'$\beta$', r'$m_{\rm min}$',r'$m_{\rm max}$', r'$dm_{\rm min}$', r'$\mu$', r'$\sigma$', r'$f$']
        
    if pop_model=='brokenpowerlaw+2peaks':
        H0_lo = 20
        H0_hi = 140

        Om0_lo = Om0grid[0]
        Om0_hi = Om0grid[-1]
        
        log10n0_lo = -8.0
        log10n0_hi = 0.0

        z1_lo = 1
        z1_hi = 200

        z50_lo = 0
        z50_hi = 1
        
        delta_lo = -50
        delta_hi = 50

        gamma_low = 0
        gamma_high = 10

        m_min_1_low = 2
        m_min_1_high = 10

        m_max_1_low = 40
        m_max_1_high = 200

        alpha_1_low = 0
        alpha_1_high = 6

        dm_min_1_low = 1
        dm_min_1_high = 100

        dm_max_1_low = 1
        dm_max_1_high = 100

        m_min_2_low = 6
        m_min_2_high = 50

        m_max_2_low = 40
        m_max_2_high = 100

        alpha_2_low = 0
        alpha_2_high = 6

        dm_min_2_low = 1
        dm_min_2_high = 100

        dm_max_2_low = 1
        dm_max_2_high = 100

        beta_low = 0
        beta_high = 6

        mu1_low = 5
        mu1_high = 20

        mu2_low = 25
        mu2_high = 40

        sigma_low = 1
        sigma_high = 10

        f1_low = 0
        f1_high = 1

        f2_low = 0
        f2_high = 1

        break_mass_lo = 20
        break_mass_hi = 50
        
        lower_bound = [H0_lo,Om0_lo,
                       log10n0_lo, z1_lo, z50_lo, delta_lo,gamma_low,
                       -4,-4,20,2,1,50,1,0,0,5,1,25,1,-2]
        upper_bound = [H0_hi,Om0_hi,
                       log10n0_hi, z1_hi, z50_hi, delta_hi,gamma_high,
                       12,12,50,10,100,300,100,1,1,20,10,40,10,7]
        
        labels = [
            "H0", "Om0",
            '$\log_{10}n_0$','z1', 'z50', r'$\delta$',
            "gamma","alpha_1", "alpha_2", "break_mass", "m_min", "dm_min", "m_max", "dm_max",
            "lam_1", "lam_2", "mpp_1", "sigpp_1",
            "mpp_2", "sigpp_2","beta"


        ]

    if pop_model=='mock_data':
        log_p_pop = log_p_pop_mock_data

    return lower_bound, upper_bound, labels