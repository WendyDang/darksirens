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
from darksirens.utils.utils import *
from darksirens.inference.likelihood import *

from darksirens.gw.utils import load_gw_samples
from darksirens.em.utils import load_survey

import matplotlib

import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c=sns.color_palette('colorblind')

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')


def main():
    optp = ArgumentParser()
    optp.add_argument("--survey_path", help="path to survey pixelated data")
    optp.add_argument("--gw_path", help="path to gw data")
    optp.add_argument("--save_path", help="where to save", default='./')
    optp.add_argument("--nsteps", type=int, default=1000)
    optp.add_argument("--nsamp", type=int, default=64)
    #optp.add_argument("--seed", type=int, default=22)

    opts = optp.parse_args()

    survey_path = opts.survey_path
    gw_path = opts.gw_path
    save_path = opts.save_path
    nsteps = opts.nsteps
    nsamp = opts.nsamp
    #seed = opts.seed
    
    nside, ngals, zgals, dzgals, wgals = load_survey(survey_path, dz=0.001)
        
    ra, dec, m1det, m2det, dL, p_pe, nEvents = load_gw_samples(gw_path, nsamp=nsamp)
    
    npix = hp.pixelfunc.nside2npix(nside)
    apix = hp.pixelfunc.nside2pixarea(nside)

    print(npix)
    samples_ind = hp.pixelfunc.ang2pix(nside,np.pi/2-dec,ra)

    H0_lo = 20
    H0_hi = 80

    Om0_lo = Om0grid[0]
    Om0_hi = Om0grid[-1] 

    gamma_lo = -5.0
    gamma_hi = 5.0

    mu_lo = 20
    mu_hi = 50

    sigma_lo = 1
    sigma_hi = 10

    log10n0_lo = -8.0
    log10n0_hi = 0.0

    z1_lo = 1
    z1_hi = 200

    z50_lo = 0
    z50_hi = 1

    lower_bound = [H0_lo, log10n0_lo, z1_lo, z50_lo, gamma_lo, mu_lo, sigma_lo]
    upper_bound = [H0_hi, log10n0_hi, z1_hi, z50_hi,  gamma_hi, mu_hi, sigma_hi]

    ndims = len(lower_bound)
    nlive = 500

    labels = [r'$H_0$',r'$\log_{10}n_0$','z1', 'z50',# r'$\delta$',
              r'$\gamma$',r'$\mu$',r'$\sigma$']
              #, r'$\beta$']


    def prior_transform(theta):
        transformed_params = [
            theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i] 
            for i in range(len(theta))
        ]

        return tuple(transformed_params)

    def likelihood(coord):
        ll = darksiren_log_likelihood(*coord,m1det,m2det,dL,ra,dec,p_pe,samples_ind)
        if np.isnan(ll):
            return -np.inf
        else:
            return ll

    def likelihood_emcee(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf
        ll = darksiren_log_likelihood(*coord,m1det,m2det,dL,ra,dec,p_pe,samples_ind)
        if np.isnan(ll):
            return -np.inf
        else:
            return ll

    import emcee

    n_walkers = int(2*ndims)
    p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))
    n_steps = nsteps

    sampler = emcee.EnsembleSampler(n_walkers, ndims, likelihood_emcee,
                                    moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ])#, pool=pool)
    sampler.run_mcmc(p0, n_steps, progress=True)

    shape = sampler.flatchain.shape[0]
    print(shape)

    dpostsamples_backup = sampler.flatchain[int(shape/2):,:]

    shape = dpostsamples_backup.shape[0]
    print(shape)
    choose = np.random.randint(0,shape,10000)


    import corner
    truths = [H0Planck,-4,5,0.3,-1,35,5]

    dpostsamples = dpostsamples_backup[choose]

    fig = corner.corner(dpostsamples, labels=labels, hist_kwargs={'density': True}, truths=truths)#, range=ranges)
    plt.savefig(save_path+'corner.pdf')


if __name__ == "__main__":
    main()
