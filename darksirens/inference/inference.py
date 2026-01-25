#!/usr/bin/env python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
from jax import jit, vmap
import jax.numpy as jnp

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from argparse import ArgumentParser

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *
from darksirens.inference.likelihood import darksiren_log_likelihood
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

sns.set_context("talk")
sns.set_style("ticks")
sns.set_palette("colorblind")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------------------------------------
# LSS helpers
# ------------------------------------------------------------
zgrid = jnp.linspace(0.0, 5.0, 1024)

@jit
def Ngals_lessthanz_grid(pix, zgals):
    zs = zgals[pix]
    mask = (zs[None, :] < zgrid[:, None])
    return mask.sum(axis=1)

Ngals_lessthanz_grid_vmap = jit(
    vmap(Ngals_lessthanz_grid, in_axes=(0, None), out_axes=0)
)

@jit
def overdensity_from_counts(Ncum_pix_z):
    Nmean_z = jnp.mean(Ncum_pix_z, axis=0)
    Nmean_z = jnp.where(Nmean_z > 0.0, Nmean_z, 1.0)
    return (Ncum_pix_z - Nmean_z[None, :]) / Nmean_z[None, :]


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    if value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def main():
    optp = ArgumentParser()
    optp.add_argument("--survey_path")
    optp.add_argument("--gw_path")
    optp.add_argument("--gwselection_path")
    optp.add_argument("--save_path", default="./")
    optp.add_argument("--pop_model", default="powerlaw+peak")
    optp.add_argument("--universe_model", default="dark_sirens_LSS")
    optp.add_argument("--nsamp", type=int, default=256)
    optp.add_argument("--batch", type=int, default=1)
    optp.add_argument("--emcee", type=str_to_bool, nargs="?", const=False, default=False)
    optp.add_argument("--dynesty", type=str_to_bool, nargs="?", const=False, default=False)
    optp.add_argument("--jaxns", type=str_to_bool, nargs="?", const=True, default=False)
    optp.add_argument("--nlive", type=int, default=1000)
    optp.add_argument("--nwalkers", type=int, default=32)
    optp.add_argument("--nsteps", type=int, default=1000)
    optp.add_argument("--seed", type=int, default=22)
    optp.add_argument("--use_LSS", type=str_to_bool, nargs="?", const=True, default=True)
    optp.add_argument("--max_samples", type=int, default=1_000_000, help="Maximum number of likelihood calls for JAXNS.")

    opts = optp.parse_args()

    # --------------------------------------------------------
    # Load survey and GW data
    # --------------------------------------------------------
    nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)

    m1det, m2det, dL, ra, dec, p_pe, nEvents = load_gw_samples(
        opts.gw_path, nsamp=opts.nsamp
    )
    print(f"Analyzing {nEvents} events with {opts.nsamp} PE samples each.")

    (
        m1detsels,
        m2detsels,
        dLsels,
        rasels,
        decsels,
        p_draw,
        Ndraw,
    ) = load_selection_samples(opts.gwselection_path)

    npix = hp.nside2npix(nside)
    apix = hp.nside2pixarea(nside)

    samples_ind = hp.ang2pix(nside, np.pi/2 - dec, ra)
    selsamples_ind = hp.ang2pix(nside, np.pi/2 - decsels, rasels)

    zgals_pe = zgals[samples_ind]
    dzgals_pe = dzgals[samples_ind]
    wgals_pe = wgals[samples_ind]

    zgals_sel = zgals[selsamples_ind]
    dzgals_sel = dzgals[selsamples_ind]
    wgals_sel = wgals[selsamples_ind]

    pixels_pe = jnp.asarray(samples_ind)
    pixels_sel = jnp.asarray(selsamples_ind)

    # --------------------------------------------------------
    # LSS overdensity field
    # --------------------------------------------------------
    if opts.use_LSS:
        pix_indices = jnp.arange(npix)
        Ncum_pix_z = Ngals_lessthanz_grid_vmap(pix_indices, zgals)
        delta_g_pix_z = overdensity_from_counts(Ncum_pix_z)
    else:
        delta_g_pix_z = jnp.zeros((npix, len(zgrid)))

    # --------------------------------------------------------
    # Priors: population + survey + cosmology
    # --------------------------------------------------------
    pop_lower, pop_upper, pop_labels = pop_model_prior_parser(pop_model=opts.pop_model)

    # Survey hyperparameters (including LSS)
    survey_labels = ["log10n0", "z50", "w", "delta", "gamma", "b_miss", "alpha"]
    survey_lower = [-10.0, 0.0, 0.01, -3.0, -10.0, 0.0, 0.0]
    survey_upper = [10.0, 1.0, 1.0, 3.0, 10.0, 5.0, 1.0]

    # Cosmology parameters
    cosmo_labels = ["H0", "Om0"]
    cosmo_lower = [20.0, Om0Planck-0.1]
    cosmo_upper = [120.0, Om0Planck+0.1]

    labels = cosmo_labels + pop_labels + survey_labels
    lower_bound = np.array(cosmo_lower + list(pop_lower) + survey_lower)
    upper_bound = np.array(cosmo_upper + list(pop_upper) + survey_upper)

    ndims = len(labels)
    n_pop = len(pop_labels)

    # --------------------------------------------------------
    # Fiducial likelihood check
    # --------------------------------------------------------
    cosmo_params = (70.0, 0.3)
    pop_params = (2.0, 1.0, 5.0, 80.0, 0.5, 35.0, 5.0, 0.1)
    survey_params = (-2.5, 0.5, 0.05, 1.5, 2.0,
                     1.0 if opts.use_LSS else 0.0,
                     1.0 if opts.use_LSS else 0.0)

    ll_fid = darksiren_log_likelihood(
        cosmo_params, survey_params, pop_params,
        m1det, m2det, dL, p_pe, pixels_pe,
        zgals_pe, dzgals_pe, wgals_pe,
        m1detsels, m2detsels, dLsels, p_draw,
        pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
        nEvents, opts.nsamp, Ndraw, apix, opts.batch,
        opts.pop_model, opts.universe_model,
        delta_g_pix_z
    )
    print("Fiducial log-likelihood:", float(ll_fid))

    # --------------------------------------------------------
    # Likelihood wrappers
    # --------------------------------------------------------
    def prior_transform(theta):
        return tuple(
            theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
            for i in range(ndims)
        )

    def likelihood(coord):
        H0, Om0 = coord[:2]
        pop_params = coord[2:2+n_pop]
        survey_params = coord[2+n_pop:]

        ll = darksiren_log_likelihood(
            (H0, Om0), survey_params, pop_params,
            m1det, m2det, dL, p_pe, pixels_pe,
            zgals_pe, dzgals_pe, wgals_pe,
            m1detsels, m2detsels, dLsels, p_draw,
            pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
            nEvents, opts.nsamp, Ndraw, apix, opts.batch,
            opts.pop_model, opts.universe_model,
            delta_g_pix_z
        )
        return -np.inf if np.isnan(ll) else ll

    def likelihood_emcee(coord):
        if np.any(coord < lower_bound) or np.any(coord > upper_bound):
            return -np.inf

        return likelihood(coord)

    # --------------------------------------------------------
    # Samplers
    # --------------------------------------------------------
    dpostsamples = None

    # --------------------------------------------------------
    # JAXNS Sampler (API for JAXNS >= 3.x)
    # --------------------------------------------------------
    if opts.jaxns:
        import tensorflow_probability.substrates.jax as tfp
        tfpd = tfp.distributions

        from jaxns import NestedSampler
        from jaxns.framework.model import Model
        from jaxns.framework.prior import Prior

        print(f"Running JAXNS with {opts.nlive} live points")

        # ----------------------------------------------------
        # PRIOR MODEL (generator)
        # ----------------------------------------------------
        def prior_model():
            # Cosmology
            H0 = yield Prior(
                tfpd.Uniform(low=lower_bound[0], high=upper_bound[0]),
                name="H0"
            )
            Om0 = yield Prior(
                tfpd.Uniform(low=lower_bound[1], high=upper_bound[1]),
                name="Om0"
            )

            # Population parameters
            pop_vals = []
            for i, name in enumerate(pop_labels):
                p = yield Prior(
                    tfpd.Uniform(
                        low=lower_bound[2 + i],
                        high=upper_bound[2 + i]
                    ),
                    name=name
                )
                pop_vals.append(p)

            # Survey parameters
            survey_vals = []
            for j, name in enumerate(survey_labels):
                idx = 2 + n_pop + j
                s = yield Prior(
                    tfpd.Uniform(
                        low=lower_bound[idx],
                        high=upper_bound[idx]
                    ),
                    name=name
                )
                survey_vals.append(s)

            return (H0, Om0), jnp.array(pop_vals), jnp.array(survey_vals)

        # ----------------------------------------------------
        # LIKELIHOOD WRAPPER
        # ----------------------------------------------------
        def log_likelihood(cosmo_params, pop_params, survey_params):
            return darksiren_log_likelihood(
                cosmo_params,
                survey_params,
                pop_params,
                m1det, m2det, dL, p_pe, pixels_pe,
                zgals_pe, dzgals_pe, wgals_pe,
                m1detsels, m2detsels, dLsels, p_draw,
                pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
                nEvents, opts.nsamp, Ndraw, apix, opts.batch,
                opts.pop_model, opts.universe_model,
                delta_g_pix_z
            )

        # ----------------------------------------------------
        # BUILD MODEL
        # ----------------------------------------------------
        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood
        )

        # Optional sanity check
        # model.sanity_check(key=jax.random.PRNGKey(0), S=32)

        # ----------------------------------------------------
        # RUN NESTED SAMPLING
        # ----------------------------------------------------
        ns = NestedSampler(
            model=model,
            num_live_points=opts.nlive,
            max_samples=opts.max_samples,
            verbose=True
        )


        key = jax.random.PRNGKey(opts.seed)
        termination_reason, state = ns(key)
        results = ns.to_results(termination_reason, state)

        # ----------------------------------------------------
        # EXTRACT POSTERIOR SAMPLES
        # ----------------------------------------------------
        # results.samples_x is a dict of named variables
        posterior = results.samples_x

        # Convert to your dynesty/emcee-style array
        dpostsamples = jnp.column_stack(
            [posterior[name] for name in labels]
        )

        print("JAXNS sampling complete.")


    if opts.emcee:
        import emcee
        nwalkers = opts.nwalkers
        nsteps = opts.nsteps

        print(f"Running emcee: {nwalkers} walkers, {nsteps} steps")
        p0 = np.random.uniform(lower_bound, upper_bound, size=(nwalkers, ndims))

        sampler = emcee.EnsembleSampler(
            nwalkers, ndims, likelihood_emcee,
            moves=[(emcee.moves.DEMove(), 0.8),
                   (emcee.moves.DESnookerMove(), 0.2)]
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

        chain = sampler.flatchain
        half = chain.shape[0] // 2
        dpostsamples = chain[half:]

    if opts.dynesty:
        from dynesty.utils import resample_equal
        from dynesty import NestedSampler

        sampler = NestedSampler(
            likelihood, prior_transform, ndims,
            bound="multi", sample="rwalk", nlive=opts.nlive
        )
        sampler.run_nested(dlogz=0.1)
        res = sampler.results

        weights = np.exp(res["logwt"] - res["logz"][-1])
        dpostsamples = resample_equal(res.samples, weights)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    if dpostsamples is not None:
        with open(os.path.join(opts.save_path, "samples.pkl"), "wb") as f:
            pickle.dump(dpostsamples, f)

        import corner
        fig = corner.corner(dpostsamples, labels=labels)
        fig.savefig(os.path.join(opts.save_path, "corner.pdf"))
        print("Saved posterior samples and corner plot.")
    else:
        print("No sampler was run.")


if __name__ == "__main__":
    main()