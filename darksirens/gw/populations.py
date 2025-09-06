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
from jax.nn import log_sigmoid

from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *

mass = jnp.linspace(1, 300, 1000)
mass_ratio =  jnp.linspace(0, 1, 2000)

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

@jit
def log_smooth_turnon(m, mmin, width=0.05):
    """A function that smoothly transitions from 0 to 1.
    :param m: The function argument.
    :param mmin: The location around which the function transitions.
    :param width: (optional) The fractional width of the transition.
    """
    dm = mmin*width
    return np.log(1) - jnp.log1p(jnp.exp(-(m-mmin)/dm))

@jit
def logpm1_powerlaw(m1,m_min,m_max,alpha,dm_min,dm_max):
    pm1 = mass**(-alpha)*jnp.exp(log_smooth_turnon(mass, m_min, width=dm_min))*jnp.exp(log_smooth_turnon(mass, m_max, width=-dm_max))
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

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
def powerlaw(data, slope, minimum, maximum):
    norm = jnp.where(
        jnp.isclose(slope, -1), 
        jnp.log(jnp.log(maximum / minimum)),
        -jnp.log(jnp.abs(slope + 1)) + jnp.log(jnp.abs(maximum**(slope+1) - minimum**(slope+1)))
    )
    window = jnp.logical_and(data >= minimum, data <= maximum)
    p = jnp.where(window, slope*jnp.log(data), -jnp.inf*jnp.ones_like(data))
    return p - norm

@jit
def logBrokenPowerLaw(data, slope_1, slope_2, xmin, xmax, break_fraction):
    slope_1 = -slope_1
    slope_2 = -slope_2
    m_break = xmin + break_fraction * (xmax - xmin)
    correction = powerlaw(m_break, slope_2, m_break, xmax) - powerlaw(
        m_break, slope_1, xmin, m_break
    )
    low_part = powerlaw(data, slope_1, xmin, m_break)
    high_part = powerlaw(data, slope_2, m_break, xmax)
    
    # this might be nan gradient?
    logprob = jnp.where(data < m_break, low_part + correction, high_part)

    return logprob + log_sigmoid(-correction) # - log(1+exp(correction))

def log_expit(x):
    """
    if (x < 0.0) {
        return x - std::log1p(std::exp(x));
    }
    else {
        return -std::log1p(std::exp(-x));
    }
    exact same expression, just more numerically stable for each side of 0
    log(1 / [1 + exp(-x)]) = log(exp(x) / (exp(x) + 1)) = x - log(1 + exp(x))

    exactly as done in scipy.special.log_expit

    DANGEROUS: https://github.com/google/jax/issues/1052 gradients will be nan.

    cannot use a single `where', must use the "double-where" trick

    consider differentiating this function for x = -1000, using naive single-where method
    the arguments of where are (True, -1000, jnp.inf). This causes trouble for gradients as the 
    divergence is propagated along, eventually it is multiplied by 0 to remove it, but 0*inf is an issue.
    With the double where method, now evaulating the function for x=-1000 gives you (True, -1000, 0.)
    and this works because the forward differentiation is done serially, so we never get an inf
    """
    condition = x < 0
    posx_valid = jnp.where(condition, 0, x) # in forward differentiation, gradient is 0 for condition, 1 where false
    negx_valid = jnp.where(condition, x, 0) # in forward differentiation, gradient is 0 for condition, 1 where false
    
    return jnp.where(condition, negx_valid-jnp.log1p(jnp.exp(negx_valid)), -jnp.log1p(jnp.exp(-posx_valid)))

def m_smoother(m1s, minimum, delta, buffer=1e-3):
    '''
    remember, logspace
    return log(1) if greater than minimum + delta
    return log(0) if less than minimum
    return log(1 / 1+f(m-mmin, delta)) if inside minimum and minimum + delta

    standard powerlaw + peak smoother: https://arxiv.org/pdf/2111.03634.pdf B5
    '''

    m_prime = jnp.clip(m1s - minimum, buffer, delta-buffer)
    return log_expit(-delta/m_prime - delta/(m_prime - delta))

@jit
def logpm1_brokenpowerlaw(m1,alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max):
    break_fraction = (break_mass  - m_min) / (m_max - m_min)
    logBPL = logBrokenPowerLaw(mass,alpha_1,alpha_2,m_min,m_max,break_fraction)
    logpm1 = logBPL +  jnp.log(Sfilter_low(mass,m_min,dm_min)) + jnp.log(Sfilter_high(mass,m_max,dm_max))
    pm1 = jnp.exp(logpm1)
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logpm1_powerlaw_peak(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1):
    p1 = jnp.exp(logpm1_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1))
    p2 = jnp.exp(logpm1_peak(m1,mu,sigma))
    
    pm1 = (1-f1)*p1 + f1*p2
    return jnp.log(pm1)

@jit
def logpm1_brokenpowerlaw_2peaks(m1, alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max, lam_1, lam_2, mpp_1, sigpp_1, mpp_2, sigpp_2):
    p1 = jnp.exp(logpm1_brokenpowerlaw(m1, alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max))
    p2 = jnp.exp(logpm1_peak(m1,mpp_1, sigpp_1))
    p3 = jnp.exp(logpm1_peak(m1,mpp_2, sigpp_2))
    
    pm1 = lam_1*p1 + lam_2*p2 + (1-lam_1-lam_2)*p3
    return jnp.log(pm1)

##################################################################################################
@jit
def log_p_pop_mock_data(m1,m2,mu,sigma,beta):
    log_dNdm1 = logpm1_peak(m1,mu,sigma)
    log_dNdm2 = logpm1_peak(m2,mu,sigma)
    log_fq = logfq(m1,m2,beta)

    return log_dNdm1 + log_dNdm2 + log_fq

@jit
def log_p_pop_powerlaw_peak(m1,m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_peak(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_peak(m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_fq = logfq(m1,m2,beta)

    return log_dNdm1 + log_dNdm2 + log_fq 

@jit
def log_p_pop_brokenpowerlaw_2peaks(m1,m2,alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max, lam_1, lam_2, mpp_1, sigpp_1, mpp_2, sigpp_2, beta):
    log_dNdm1 = logpm1_brokenpowerlaw_2peaks(m1,alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max, lam_1, lam_2, mpp_1, sigpp_1, mpp_2, sigpp_2)
    log_dNdm2 = logpm1_brokenpowerlaw_2peaks(m2,alpha_1, alpha_2, break_mass, m_min, dm_min, m_max, dm_max, lam_1, lam_2, mpp_1, sigpp_1, mpp_2, sigpp_2)
    log_fq = logfq(m1,m2,beta)

    return log_dNdm1 + log_dNdm2 + log_fq

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
        
        delta_lo = -50
        delta_hi = 50

        gamma_lo = -5.0
        gamma_hi = 5.0

        mu_lo = 20
        mu_hi = 50

        sigma_lo = 1
        sigma_hi = 10

        gamma_lo = -10
        gamma_hi = 10

        log10n0_lo = -10.0
        log10n0_hi = 1.0

        alpha_lo = 0
        alpha_hi = 10

        beta_lo = 0
        beta_hi = 6

        m_min_lo = 1
        m_min_hi = 10

        m_max_lo = 50
        m_max_hi = 100

        mu_lo = 20
        mu_hi = 50

        sigma_lo = 1
        sigma_hi = 10

        f_lo = 0.0
        f_hi = 1.0

        dm_lo = 0
        dm_hi = 1

        lower_bound = [H0_lo, Om0_lo,
                       log10n0_lo, z1_lo, z50_lo, delta_lo, gamma_lo,
                       alpha_lo, beta_lo, m_min_lo, m_max_lo, dm_lo, dm_lo, mu_lo, sigma_lo, f_lo]
        upper_bound = [H0_hi, Om0_hi,
                       log10n0_hi, z1_hi, z50_hi, delta_hi, gamma_hi,
                       alpha_hi, beta_hi, m_min_hi, m_max_hi, dm_hi, dm_hi, mu_hi, sigma_hi, f_hi]
        
        labels = [r'$H_0$', r'$\Omega_m$', 
                  '$\log_{10}n_0$','z1', 'z50', r'$\delta$',
                  r'$\gamma$', r'$\alpha$',r'$\beta$', r'$m_{\rm max}$', r'$m_{\rm min}$', r'$dm_{\rm max}$', r'$dm_{\rm min}$', r'$\mu$', r'$\sigma$', r'$f$']
        
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