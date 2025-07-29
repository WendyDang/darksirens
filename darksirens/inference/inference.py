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
    #optp.add_argument("--nsamp", type=int, default=4096)
    #optp.add_argument("--seed", type=int, default=22)

    opts = optp.parse_args()

    survey_path = opts.survey_path
    gw_path = opts.gw_path
    save_path = opts.save_path
    nsteps = opts.nsteps
    #nsamp = opts.nsamp
    #seed = opts.seed
    
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = jnp.asarray(f['zgals'])
        dzgals = 0.001*(1+zgals)
        wgals = jnp.ones(zgals.shape)
        ngals = jnp.asarray(f['ngals'])
        
    with h5py.File(gw_path, 'r') as inp:
        nsamps = inp.attrs['nsamp']
        nEvents_ = inp.attrs['nobs']
        ra = jnp.array(inp['ra'])
        dec = jnp.array(inp['dec'])
        m1det = jnp.array(inp['m1det'])
        m2det = jnp.array(inp['m2det'])
        dL = jnp.array((jnp.array(inp['dL'])*u.Mpc).value)


    nsamp = 32
    nEvents = 1000
    ra = ra.reshape(nEvents_,nsamps)[0:nEvents,0:nsamp]
    dec = dec.reshape(nEvents_,nsamps)[0:nEvents,0:nsamp]
    m1det = m1det.reshape(nEvents_,nsamps)[0:nEvents,0:nsamp]
    m2det = m2det.reshape(nEvents_,nsamps)[0:nEvents,0:nsamp]
    dL = dL.reshape(nEvents_,nsamps)[0:nEvents,0:nsamp]
    print(ra.shape)
    ra = ra[0:nEvents].flatten()
    dec = dec[0:nEvents].flatten()
    m1det = m1det[0:nEvents].flatten()
    m2det = m2det[0:nEvents].flatten()
    dL = dL[0:nEvents].flatten()

    p_pe = jnp.ones(len(dL))
    
    npix = hp.pixelfunc.nside2npix(nside)
    apix = hp.pixelfunc.nside2pixarea(nside)

    print(npix)
    samples_ind = hp.pixelfunc.ang2pix(nside,np.pi/2-dec,ra)

    @jit
    def dV_of_z_normed(z,Om0,gamma):
        dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
        prob = dV/jnp.trapezoid(dV,zgrid)
        return jnp.interp(z,zgrid,prob)

    from jax.scipy.stats import norm

    mass = jnp.linspace(1, 150, 2000)
    mass_ratio =  jnp.linspace(0, 1, 2000)

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
    @jit
    def logdiffexp(x, y):
        return x + jnp.log1p(jnp.exp(y-x))

    @jit
    def Ngals_lessthanz(z,pix):
        Ngals = jnp.where((zgals[pix] < z), jnp.ones(len(zgals[pix])), 0).sum()
        return Ngals

    Ngals_lessthanz_vmap = jit(vmap(Ngals_lessthanz, in_axes=(0,None), out_axes=0))


    n0fid = 2e-4
    gammafid = 1

    @jit
    def Ngals_expected_lessthanz(z,H0=H0Planck,Om0=Om0Planck,n0=n0fid,gamma=gammafid):
        zz = jnp.expm1(jnp.linspace(jnp.log(1), jnp.log(z+1), 200))
        Nexpected = jnp.trapezoid(n0*apix*dV_of_z(zz,H0,Om0)*(1+zz)**(gamma-1),zz)
        return Nexpected

    Ngals_expected_lessthanz_vmap = jit(vmap(Ngals_expected_lessthanz, in_axes=(0,None,None,None,None), out_axes=0))


    from jax.scipy.special import expit

    z50 = 0.50
    z1=100
    def Pcomplete0(z,z1,z50):
        return expit(-z1*(z/z50)+z1)

    @jit
    def completeness_fraction(H0,Om0,n0,z1,z50,gamma,z,pix):
        Nexpected = 1+Ngals_expected_lessthanz_vmap(zgrid,H0,Om0,n0,gamma)
        Ngals = Ngals_lessthanz_vmap(zgrid,pix)
        ratio = Ngals/Nexpected

        #ratio = jnp.where((ratio < 1), ratio, 0)
        #ratio = jnp.where((ratio != 0), ratio, 1)
        ratio *= Pcomplete0(zgrid,z1,z50)
        pvol = dV_of_z(zgrid, H0, Om0)*(1+zgrid)**(gamma-1)

        V = jnp.trapezoid(ratio*pvol,zgrid)
        Vmax = jnp.trapezoid(pvol,zgrid)

        pmiss = (1-ratio)*pvol
        pmiss_normed = pmiss/jnp.trapezoid(pmiss,zgrid)

        pmiss_z = jnp.interp(z,zgrid,pmiss_normed)

        return V/Vmax, pmiss_z, ratio

    completeness_fraction_vmap = jit(vmap(completeness_fraction, in_axes=(None,None,None,None,None,None,0,0), out_axes=0))

    from jaxinterp2d import interp2d

    @jit
    def logpcatalog(z, pix, Om0, gamma):
        zs = zgals[pix] 
        ddzs = dzgals[pix]
        wts = wgals[pix]*dV_of_z(zs,H0Planck,Om0)**(1+zs)**(gamma-1)
        ngals = len(zs)
        wts = wts/jnp.sum(wts)
        return logsumexp(jnp.log(wts) + norm.logpdf(z,zs,ddzs))

    logpcatalog_vmap = jit(vmap(logpcatalog, in_axes=(0,0,None,None), out_axes=0))


    @jit
    def logPriorUniverse(z,pix,H0,Om0,n0,z1,z50,gamma):
        f, pmiss, ratio = completeness_fraction_vmap(H0,Om0,n0,z1,z50,gamma,z,pix)

        logpmiss = jnp.nan_to_num(jnp.log(pmiss), -jnp.inf)

        logpcat = jnp.nan_to_num(logpcatalog_vmap(z, pix, Om0, gamma), -jnp.inf)

        logprob = jnp.log( jnp.exp(jnp.log(f) + logpcat) + jnp.exp(jnp.log(1-f) + logpmiss) ) #+ (gamma-1)*jnp.log1p(z)

        return logprob

    seed = np.random.randint(1000)
    key = jax.random.PRNGKey(1000)

    Om0 = Om0Planck
    beta = 0

    @jit
    def darksiren_log_likelihood(H0,log10n0,z1,z50,gamma,mu,sigma):
        n0 = 10**log10n0

        z = z_of_dL(dL, H0, Om0)
        m1 = m1det/(1+z)
        m2 = m2det/(1+z)

        log_weights = log_p_pop_pl_pl(m1,m2,mu,sigma,beta)

        log_weights += - jnp.log(ddL_of_z(z,dL,H0,Om0)) - jnp.log(p_pe) - 2*jnp.log1p(z) + logPriorUniverse(z,samples_ind,H0,Om0Planck,n0,z1,z50,gamma)

        log_weights = log_weights.reshape((nEvents,nsamp))
        ll = jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights,axis=-1))

        return ll

    H0_lo = 20
    H0_hi = 100

    Om0_lo = Om0grid[0]
    Om0_hi = Om0grid[-1] 

    gamma_lo = -30.0
    gamma_hi = 30.0

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
        ll = darksiren_log_likelihood(*coord)
        if np.isnan(ll):
            return -np.inf
        else:
            return ll

    def likelihood_emcee(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf
        ll = darksiren_log_likelihood(*coord)
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
