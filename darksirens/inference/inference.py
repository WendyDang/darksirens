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

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser

from darksirens.gw.utils import load_gw_samples
from darksirens.gw.utils import load_selection_samples
from darksirens.em.utils import load_survey

import corner

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


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def main():
    optp = ArgumentParser()
    optp.add_argument("--survey_path", help="path to survey pixelated data")
    optp.add_argument("--gw_path", help="path to gw data")
    optp.add_argument("--gwselection_path", help="path to gw data")
    optp.add_argument("--save_path", help="where to save", default='./')
    optp.add_argument("--pop_model", help="specify pop model", default='powerlaw+peak')
    optp.add_argument("--universe_model", help="specify pop model", default='spectral_sirens_fast')
    optp.add_argument("--nsamp", type=int, default=256)
    optp.add_argument("--nsamp_sel", type=int, default=5000)
    optp.add_argument("--batch", type=int, default=1)
    optp.add_argument("--emcee", type=str_to_bool, nargs='?', const=False, default=True)
    optp.add_argument("--dynesty", type=str_to_bool, nargs='?', const=False, default=False)
    optp.add_argument("--numpyro", type=str_to_bool, nargs='?', const=False, default=False)
    optp.add_argument("--nlive", type=int, default=500)
    optp.add_argument("--nsteps", type=int, default=1000)
    optp.add_argument("--seed", type=int, default=22)

    opts = optp.parse_args()

    survey_path = opts.survey_path
    gw_path = opts.gw_path
    gwselection_path = opts.gwselection_path
    save_path = opts.save_path
    pop_model = opts.pop_model
    universe_model = opts.universe_model
    nsteps = opts.nsteps
    nsamp = opts.nsamp
    nsamp_sel = opts.nsamp_sel
    batch = opts.batch
    seed = opts.seed
    
    nside, ngals, zgals, dzgals, wgals = load_survey(survey_path)
        
    m1det, m2det, dL, ra, dec, p_pe, nEvents = load_gw_samples(gw_path, nsamp=nsamp)
    
    m1detsels, m2detsels, dLsels, rasels, decsels, p_draw, Ndraw = load_selection_samples(gwselection_path,nsamp=nsamp_sel)
    
    npix = hp.pixelfunc.nside2npix(nside)
    apix = hp.pixelfunc.nside2pixarea(nside)

    samples_ind = hp.pixelfunc.ang2pix(nside,np.pi/2-dec,ra)
    selsamples_ind = hp.pixelfunc.ang2pix(nside,np.pi/2-decsels,rasels)
    
    lower_bound, upper_bound, labels = pop_model_prior_parser(pop_model=pop_model)

    ndims = len(lower_bound)

    def prior_transform(theta):
        transformed_params = [
            theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i] 
            for i in range(len(theta))
        ]

        return tuple(transformed_params)

    def likelihood(coord):
        cosmo_params = coord[0:2]
        survey_params = coord[2:7]
        pop_params = coord[7:]
        
        ll = darksiren_log_likelihood(cosmo_params, survey_params, pop_params,
                                      m1det, m2det, dL, ra, dec, p_pe, samples_ind,
                                      m1detsels, m2detsels, dLsels, rasels, decsels, p_draw, selsamples_ind,
                                      nEvents, nsamp, Ndraw, apix, batch, zgals, dzgals, wgals, pop_model, universe_model)
        if np.isnan(ll):
            return -np.inf
        else:
            return ll

    def likelihood_emcee(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf
        
        cosmo_params = coord[0:2]
        survey_params = coord[2:7]
        pop_params = coord[7:]
        
        ll = darksiren_log_likelihood(cosmo_params, survey_params, pop_params,
                                      m1det, m2det, dL, ra, dec, p_pe, samples_ind,
                                      m1detsels, m2detsels, dLsels, rasels, decsels, p_draw, selsamples_ind,
                                      nEvents, nsamp, Ndraw, apix, batch, zgals, dzgals, wgals, pop_model, universe_model)
        if np.isnan(ll):
            return -np.inf
        else:
            return ll
        
    if opts.emcee is True:

        import emcee
        nwalkers = 2 * ndims

        p0 = np.random.uniform(lower_bound, upper_bound, size=(nwalkers, len(lower_bound)))

        sampler = emcee.EnsembleSampler(nwalkers, ndims, likelihood_emcee,
                                        moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ])
        sampler.run_mcmc(p0, nsteps, progress=True)

        shape = sampler.flatchain.shape[0]
        print(shape)

        dpostsamples_backup = sampler.flatchain[int(shape/2):,:]

        shape = dpostsamples_backup.shape[0]
        print(shape)
        choose = np.random.randint(0,shape,10000)

        dpostsamples = dpostsamples_backup[choose]
        
    if opts.numpyro is True:
        
        import numpyro
        from numpyro import distributions as dist
        from numpyro.infer import MCMC
        from numpyro.infer import NUTS
        from numpyro.infer.initialization import init_to_feasible, init_to_median

        seed = np.random.randint(1000)
        key = jax.random.PRNGKey(1000)


        def darksiren_model_numpyro():

            H0 = numpyro.sample("H0", dist.Uniform(20, 100))
            Om0 = numpyro.sample('Om0', dist.Uniform(Om0grid[0], Om0grid[-1]))

            n0 = numpyro.sample('n0', dist.Uniform(1.0e-6, 1.0))
            log10n0 = jnp.log10(n0)
            
            z1 = numpyro.sample('z1', dist.Uniform(1, 200))
            z50 = numpyro.sample('z50', dist.Uniform(0, 1))
            delta = numpyro.sample('delta', dist.Uniform(-10, 10))
            gamma = numpyro.sample('gamma', dist.Uniform(-10, 10))
            
            alpha = numpyro.sample('alpha', dist.Uniform(0, 6))
            beta = numpyro.sample('beta', dist.Uniform(0, 6))
            m_min = numpyro.sample('m_min', dist.Uniform(1, 10))
            m_max = numpyro.sample('m_max', dist.Uniform(50, 200))
            dm_min = numpyro.sample('dm_min', dist.Uniform(0, 1))
            dm_max = numpyro.sample('dm_max', dist.Uniform(0, 1))
            mu = numpyro.sample('mu', dist.Uniform(20, 50))
            sigma = numpyro.sample('sigma', dist.Uniform(1, 10))
            f = numpyro.sample('f', dist.Uniform(0, 1))
            
            cosmo_params = (H0, Om0)
            survey_params = (log10n0,z1,z50,delta,gamma)
            pop_params = (alpha,beta,m_min,m_max,dm_min,dm_max,mu,sigma,f)

            ll = darksiren_log_likelihood(cosmo_params, survey_params, pop_params,
                                          m1det, m2det, dL, ra, dec, p_pe, samples_ind,
                                          m1detsels, m2detsels, dLsels, rasels, decsels, p_draw, selsamples_ind,
                                          nEvents, nsamp, Ndraw, apix, zgals, dzgals, wgals, pop_model, universe_model)
            
            ll = numpyro.deterministic('ll', ll)

            numpyro.factor('log_likelihood', ll)


        def darksiren_sample(
                thinning=100,
                num_warmup=10,
                num_samples=100,
                num_chains=1,target_accept_prob=0.9):


            RNG = jax.random.PRNGKey(0)
            MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
            kernel = NUTS(darksiren_model_numpyro, target_accept_prob=target_accept_prob, init_strategy=init_to_median())
            mcmc = MCMC(
                kernel,
                thinning=thinning,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
            )

            mcmc.run(PRIOR_RNG)

            return mcmc.get_samples()
        
        samples = darksiren_sample(
                thinning=1,
                num_warmup=1000,
                num_samples=5000,
                num_chains=1,
                target_accept_prob=0.3)
        dpostsamples = np.column_stack(samples.values())
    
    if opts.dynesty is True:

        from dynesty.utils import resample_equal
        from dynesty import NestedSampler, DynamicNestedSampler
        
        nlive = opts.nlive

        bound = 'multi'
        sample = 'rwalk'
        nprocesses = 1
        Dynamic = False

        if Dynamic is True:
            dsampler = DynamicNestedSampler(likelihood, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
            dsampler.run_nested()
        else:
            dsampler = NestedSampler(likelihood, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
            dsampler.run_nested(dlogz=0.1)

        dres = dsampler.results

        dlogZdynesty = dres.logz[-1]        # value of logZ
        dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

        # output marginal likelihood
        print('Marginalised evidence (using dynamic sampler) is {} ± {}'.format(dlogZdynesty, dlogZerrdynesty))

        # get the posterior samples
        dweights = np.exp(dres['logwt'] - dres['logz'][-1])
        dpostsamples = resample_equal(dres.samples, dweights)

        print('Number of posterior samples (using dynamic sampler) is {}'.format(dpostsamples.shape[0]))

    import pickle
    import corner

    # open a file, where you ant to store the data
    #file = open(filename + '-samples', 'wb')
    file = open(save_path + 'samples', 'wb')

    # dump information to that file
    pickle.dump(dpostsamples, file)

    # close the file
    file.close()

    fig = corner.corner(dpostsamples, labels=labels, hist_kwargs={'density': True})#, truths=truths)#, range=ranges)
    plt.savefig(save_path+'corner.pdf')


if __name__ == "__main__":
    main()
