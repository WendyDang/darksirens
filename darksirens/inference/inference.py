#!/usr/bin/env python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import pickle
import warnings
import json
import datetime

from argparse import ArgumentParser

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *
from darksirens.inference.likelihood import darksiren_log_likelihood
from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey
from darksirens.em.completeness import ( 
    zgrid, Ngals_lessthanz_grid_vmap, overdensity_from_counts,
    compute_LSS_overdensity,
)

from darksirens.inference.data import load_all_data
from darksirens.inference.likelihood import make_likelihood
from darksirens.inference.sampling import run_sampler
from darksirens.inference.prior import (
    build_parameter_space,
    get_fixed_population_params,
    make_prior_transform,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    if value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def save_settings(opts, run_dir, extra=None):
    """Save all run settings to JSON for reproducibility."""
    d = vars(opts).copy()
    if extra is not None:
        d.update(extra)

    d["environment"] = {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "healpy_version": hp.__version__,
    }

    with open(os.path.join(run_dir, "settings.json"), "w") as f:
        json.dump(d, f, indent=2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------
    optp = ArgumentParser()
    optp.add_argument("--gw_path")
    optp.add_argument("--gwselection_path")
    optp.add_argument("--survey_path")
    optp.add_argument("--save_path", default="./")

    optp.add_argument("--universe_model", default="dark_sirens")
    optp.add_argument("--pop_model", default="powerlaw+peak")
    
    optp.add_argument("--fix_population", default=False,
                      type=str_to_bool, nargs="?", const=True)
    
    optp.add_argument("--fix_cosmology", default=False,
                      type=str_to_bool, nargs="?", const=True)

    optp.add_argument("--fix_survey", default=False,
                      type=str_to_bool, nargs="?", const=True)
    
    optp.add_argument("--nsamp", type=int, default=256)

    optp.add_argument("--emcee", type=str_to_bool, nargs="?", const=False, default=False)
    optp.add_argument("--dynesty", type=str_to_bool, nargs="?", const=False, default=False)
    optp.add_argument("--jaxns", type=str_to_bool, nargs="?", const=True, default=False)

    optp.add_argument("--nlive", type=int, default=1000)
    optp.add_argument("--nwalkers", type=int, default=32)
    optp.add_argument("--nsteps", type=int, default=1000)
    optp.add_argument("--seed", type=int, default=22)
    optp.add_argument("--use_LSS", type=str_to_bool, nargs="?", const=True, default=True)
    optp.add_argument("--max_samples", type=int, default=1_000_000)

    opts = optp.parse_args()

    # --------------------------------------------------------
    # Load survey and GW data
    # --------------------------------------------------------
    data = load_all_data(opts)

    nside = data["nside"]
    zgals = data["zgals"]

    # --------------------------------------------------------
    # LSS overdensity field
    # --------------------------------------------------------
    if opts.use_LSS:
        delta_g_pix_z = compute_LSS_overdensity(zgals, nside)
    else:
        delta_g_pix_z = jnp.zeros((hp.nside2npix(nside), len(zgrid)))

    # --------------------------------------------------------
    # Build parameter space
    # --------------------------------------------------------
    labels, lower_bound, upper_bound, n_pop_eff, pop_labels, survey_labels, cosmo_labels, n_cosmo_eff, n_survey_eff, model_name = \
        build_parameter_space(opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey)

    pop_params_fid = get_fixed_population_params(opts.pop_model)
    prior_transform = make_prior_transform(lower_bound, upper_bound)

    # --------------------------------------------------------
    # Build likelihood
    # --------------------------------------------------------
    likelihood = make_likelihood(
        opts=opts,
        data=data,
        delta_g_pix_z=delta_g_pix_z,
        pop_params_fid=pop_params_fid
    )

    # --------------------------------------------------------
    # Choose sampler
    # --------------------------------------------------------
    if opts.jaxns:
        method = "jaxns"
    elif opts.dynesty:
        method = "dynesty"
    elif opts.emcee:
        method = "emcee"
    else:
        print("No sampler selected (use --jaxns / --dynesty / --emcee).")
        return

    print(f"Running sampler: {method}")

    # --------------------------------------------------------
    # Run sampler
    # --------------------------------------------------------
    results = run_sampler(
        method=method,
        likelihood=likelihood,
        prior_transform=prior_transform,
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        opts=opts
    )
    
    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{opts.pop_model}_{opts.universe_model}_{method}_{timestamp}"
    run_dir = os.path.join(opts.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save settings immediately
    save_settings(opts, run_dir, extra={
        "labels": labels,
        "lower_bound": list(map(float, lower_bound)),
        "upper_bound": list(map(float, upper_bound)),
        "n_pop_eff": n_pop_eff,
        "n_cosmo_eff": n_cosmo_eff,
        "n_survey_eff": n_survey_eff,
        "sampler": method,
        "model_name": model_name,
    })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    if results is not None:
        # Save full dict (samples + logZ + logZerr)
        np.save(os.path.join(run_dir, "samples.npy"), results)

        # Extract just the samples array for corner plot
        samples = results["samples"]

        import corner
        fig = corner.corner(samples, labels=labels)
        fig.savefig(os.path.join(run_dir, "corner.pdf"))

        print(f"Saved samples, evidence info, and corner plot to {run_dir}")
    else:
        print("No samples returned from sampler.")

if __name__ == "__main__":
    main()
