#!/usr/bin/env python3
import os

# JAX Memory Management: Adjust if you still hit Resource Exhausted errors
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import pickle
import warnings
import json
import datetime
import sys

from argparse import ArgumentParser

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *
from darksirens.utils.plotting import make_production_corner
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

# Configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def str_to_bool(value):
    """Helper to parse boolean arguments from shell scripts."""
    if isinstance(value, bool):
        return value
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
    print("="*70)
    print(f" DARK SIRENS INFERENCE PIPELINE START: {datetime.datetime.now()}")
    print("="*70)

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------
    optp = ArgumentParser()
    optp.add_argument("--gw_path", required=True)
    optp.add_argument("--gwselection_path", required=True)
    optp.add_argument("--survey_path", default=None)
    optp.add_argument("--save_path", default="./")

    optp.add_argument("--universe_model", default="spectral_sirens")
    optp.add_argument("--pop_model", default="powerlaw+peak")
    
    optp.add_argument("--fix_population", type=str_to_bool, default=False)
    optp.add_argument("--fix_cosmology", type=str_to_bool, default=False)
    optp.add_argument("--fix_survey", type=str_to_bool, default=False)
    
    optp.add_argument("--nsamp", type=int, default=256)

    optp.add_argument("--emcee", type=str_to_bool, default=False)
    optp.add_argument("--dynesty", type=str_to_bool, default=False)
    optp.add_argument("--jaxns", type=str_to_bool, default=False)

    optp.add_argument("--nlive", type=int, default=1000)
    optp.add_argument("--nwalkers", type=int, default=32)
    optp.add_argument("--nsteps", type=int, default=1000)
    optp.add_argument("--seed", type=int, default=22)
    optp.add_argument("--use_LSS", type=str_to_bool, default=False)
    optp.add_argument("--max_samples", type=int, default=1_000_000)

    opts = optp.parse_args()

    # --- Verbose Config Report ---
    print(f"[*] RUN CONFIGURATION:")
    print(f"    - Universe Model: {opts.universe_model}")
    print(f"    - Population Model: {opts.pop_model}")
    print(f"    - Fix Cosmo: {opts.fix_cosmology} | Fix Pop: {opts.fix_population}")
    print(f"    - Sampler: {'jaxns' if opts.jaxns else 'dynesty' if opts.dynesty else 'emcee' if opts.emcee else 'NONE'}")
    print(f"    - Using LSS: {opts.use_LSS}")

    # Validation for models that require survey data
    GALAXY_AWARE_MODELS = ["dark_sirens", "spectral_sirens_from_dark"]
    if opts.universe_model in GALAXY_AWARE_MODELS and not opts.survey_path:
        print(f"[!] FATAL ERROR: Model '{opts.universe_model}' requires --survey_path.")
        sys.exit(1)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print(f"[*] Loading GW and Catalog data...")
    data = load_all_data(opts)
    
    nEvents = data.get("nEvents", "Unknown")
    nside = data.get("nside", "N/A")
    print(f"    - Data loaded. Found {nEvents} GW events.")
    print(f"    - HEALPix nside detected: {nside}")

    # --------------------------------------------------------
    # LSS overdensity field (Handle memory carefully)
    # --------------------------------------------------------
    print(f"[*] Preparing LSS/Overdensity Field...")
    if opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS:
        print(f"    - Calculating high-resolution overdensity grid...")
        delta_g_pix_z = compute_LSS_overdensity(data["zgals"], nside)
    else:
        print(f"    - Non-LSS run. Creating memory-efficient dummy (1, {len(zgrid)}) grid.")
        # We use shape (1, nz) to satisfy JAX broadcasting without 93GB allocations
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))

    mem_usage = delta_g_pix_z.nbytes / 1e9
    print(f"    - Overdensity array shape: {delta_g_pix_z.shape} ({mem_usage:.4f} GB)")

    # --------------------------------------------------------
    # Build parameter space
    # --------------------------------------------------------
    print(f"[*] Constructing parameter space...")
    res = build_parameter_space(opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey)
    labels, lower_bound, upper_bound = res[0], res[1], res[2]
    
    # Extra eff values for saving to settings.json
    n_pop_eff, n_cosmo_eff, n_survey_eff, model_name = res[3], res[7], res[8], res[9]

    pop_params_fid = get_fixed_population_params(opts.pop_model)
    prior_transform = make_prior_transform(lower_bound, upper_bound)
    
    print(f"    - Sampled Parameters: {labels}")
    print(f"    - Parameter Bounds: {list(zip(labels, lower_bound, upper_bound))}")

    # --------------------------------------------------------
    # Build likelihood
    # --------------------------------------------------------
    print(f"[*] Building and Jitting likelihood...")
    likelihood = make_likelihood(
        opts=opts,
        data=data,
        delta_g_pix_z=delta_g_pix_z,
        pop_params_fid=pop_params_fid
    )

    # --------------------------------------------------------
    # Choose sampler
    # --------------------------------------------------------
    if opts.jaxns: method = "jaxns"
    elif opts.dynesty: method = "dynesty"
    elif opts.emcee: method = "emcee"
    else:
        print("[!] No sampler selected. Please use --jaxns, --dynesty, or --emcee.")
        return

    # --------------------------------------------------------
    # Run sampler
    # --------------------------------------------------------
    print(f"[*] Starting {method} sampling... (Seed: {opts.seed})")
    start_time = datetime.datetime.now()

    results = run_sampler(
        method=method,
        likelihood=likelihood,
        prior_transform=prior_transform,
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        opts=opts
    )
    
    end_time = datetime.datetime.now()
    print(f"[*] Sampling finished. Wall time: {end_time - start_time}")

    # --------------------------------------------------------
    # Create run directory and Save Output
    # --------------------------------------------------------
    if results is not None:
        timestamp = end_time.strftime("%Y-%m-%dT%H-%M-%S")
        run_name = f"{opts.pop_model}_{opts.universe_model}_{method}_{timestamp}"
        run_dir = os.path.join(opts.save_path, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"[*] Writing results to {run_dir}")
        save_settings(opts, run_dir, extra={
            "labels": labels,
            "lower_bound": list(map(float, lower_bound)),
            "upper_bound": list(map(float, upper_bound)),
            "n_pop_eff": n_pop_eff,
            "n_cosmo_eff": n_cosmo_eff,
            "n_survey_eff": n_survey_eff,
            "sampler": method,
            "model_name": model_name,
            "total_runtime": str(end_time - start_time)
        })

        # Save full dictionary (includes samples and potentially logZ)
        np.save(os.path.join(run_dir, "samples.npy"), results)

        # Generate corner plot
        print(f"[*] Generating corner plot...")
        samples = results["samples"]
        fig = make_production_corner(samples, labels)
        fig.savefig(os.path.join(run_dir, "corner.pdf"), bbox_inches='tight', dpi=200)

        print(f"[*] SUCCESS: Run complete.")
    else:
        print("[!] FATAL: Sampler failed to return samples.")

if __name__ == "__main__":
    main()