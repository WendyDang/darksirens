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
import warnings

from argparse import ArgumentParser

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *
from darksirens.inference.likelihood import darksiren_log_likelihood
from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

from darksirens.inference.sampling import run_sampler
from darksirens.inference.prior import (
    build_parameter_space,
    get_fixed_population_params,
    make_prior_transform,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

sns.set_context("talk")
sns.set_style("ticks")
sns.set_palette("colorblind")

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
    optp.add_argument("--max_samples", type=int, default=1_000_000,
                      help="Maximum number of likelihood calls for JAXNS.")

    optp.add_argument(
        "--fix_population",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Fix population parameters instead of sampling them."
    )

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
    # Priors and parameter space
    # --------------------------------------------------------
    labels, lower_bound, upper_bound, n_pop_eff, pop_labels, survey_labels, cosmo_labels = \
        build_parameter_space(opts.pop_model, opts.fix_population)

    pop_params_fid = get_fixed_population_params(opts.pop_model)
    prior_transform = make_prior_transform(lower_bound, upper_bound)
    ndims = len(labels)

    # --------------------------------------------------------
    # Fiducial likelihood check
    # --------------------------------------------------------
    cosmo_params_fid = (70.0, 0.3)
    pop_params_fid_ll = np.array(pop_params_fid)
    survey_params_fid = (-2.5, 0.5, 0.05, 1.5, 2.0,
                         1.0 if opts.use_LSS else 0.0,
                         1.0 if opts.use_LSS else 0.0)

    ll_fid = darksiren_log_likelihood(
        cosmo_params_fid, survey_params_fid, pop_params_fid_ll,
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
    # Likelihood for samplers
    # --------------------------------------------------------
    def likelihood(coord):
        coord = np.asarray(coord)
        H0, Om0 = coord[:2]

        if opts.fix_population:
            pop_params_loc = pop_params_fid
            survey_params_loc = coord[2:]
        else:
            pop_params_loc = coord[2:2 + n_pop_eff]
            survey_params_loc = coord[2 + n_pop_eff:]

        ll = darksiren_log_likelihood(
            (H0, Om0), survey_params_loc, pop_params_loc,
            m1det, m2det, dL, p_pe, pixels_pe,
            zgals_pe, dzgals_pe, wgals_pe,
            m1detsels, m2detsels, dLsels, p_draw,
            pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
            nEvents, opts.nsamp, Ndraw, apix, opts.batch,
            opts.pop_model, opts.universe_model,
            delta_g_pix_z
        )
        return -np.inf if np.isnan(ll) else ll

    # --------------------------------------------------------
    # Choose sampler
    # --------------------------------------------------------
    method = None
    if opts.jaxns:
        method = "jaxns"
    elif opts.dynesty:
        method = "dynesty"
    elif opts.emcee:
        method = "emcee"

    if method is None:
        print("No sampler selected (use --jaxns / --dynesty / --emcee).")
        return

    print(f"Running sampler: {method}")
    samples = run_sampler(
        method=method,
        likelihood=likelihood,
        prior_transform=prior_transform,
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        opts=opts
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    if samples is not None:
        os.makedirs(opts.save_path, exist_ok=True)

        with open(os.path.join(opts.save_path, "samples.pkl"), "wb") as f:
            pickle.dump(samples, f)

        import corner
        fig = corner.corner(samples, labels=labels)
        fig.savefig(os.path.join(opts.save_path, "corner.pdf"))
        print("Saved posterior samples and corner plot.")
    else:
        print("No samples returned from sampler.")


if __name__ == "__main__":
    main()
