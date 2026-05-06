#!/usr/bin/env python3
"""
darksirens_inference.py
-----------------------
Entry point for the dark-siren / spectral-siren hierarchical inference pipeline.

Usage examples
--------------
# Spectral sirens, dynesty, free cosmology + population:
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --sampler           dynesty \
    --pop_model         powerlaw+peak \
    --universe_model    spectral_sirens \
    --nlive             2000

# Dark sirens with galaxy catalog, fixed cosmology, emcee:
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --survey_path       catalog_nside64.h5 \
    --sampler           emcee \
    --pop_model         brokenpowerlaw+2peaks \
    --universe_model    dark_sirens \
    --fix_cosmology     true \
    --sigma_kernel      0.005

# Fix individual parameters via JSON:
    --fixed_parameter_values '{"$v_1$": 0.1}'
    --prior_overrides        '{"H0": [60.0, 80.0]}'
"""

import os

# ── JAX memory configuration (before any JAX import) ──────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",  "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR",    "platform")

import sys
import json
import datetime
import warnings
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.gw.populations import get_fixed_population_params
from darksirens.em.utils import load_survey
from darksirens.inference.data import load_all_data
from darksirens.inference.likelihood import make_likelihood
from darksirens.inference.sampling import run_sampler
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.utils.plotting import make_production_corner

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
warnings.simplefilter("ignore", FutureWarning)


# ── Formatting helpers ─────────────────────────────────────────────────────────

W = 72   # total line width

def _banner(text: str):
    pad = max(0, W - 4 - len(text))
    left  = pad // 2
    right = pad - left
    print(f"{'─' * W}")
    print(f"  {'·' * left} {text} {'·' * right}  ")
    print(f"{'─' * W}")

def _section(title: str):
    print()
    print(f"  ┌─ {title} {'─' * max(0, W - 6 - len(title))}┐")

def _row(label: str, value, indent: int = 4, width: int = 26):
    print(f"  │  {label:<{width}} {value}")

def _end():
    print(f"  └{'─' * (W - 3)}┘")

def _ok(msg: str):  print(f"  ✓  {msg}")
def _warn(msg: str): print(f"  ⚠  {msg}")
def _err(msg: str):  print(f"  ✗  {msg}")

def _fatal(msg: str):
    print()
    _err(f"FATAL: {msg}")
    print()
    sys.exit(1)


# ── CLI helpers ────────────────────────────────────────────────────────────────

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "t", "1", "yes", "y"}:
        return True
    if str(value).lower() in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean.")


def parse_json_arg(value: str | None, argname: str) -> dict:
    """Parse an optional JSON string argument; return {} if None."""
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object (dict).")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        _fatal(f"--{argname} must be a valid JSON object. Error: {e}\n"
               f"  Example: --{argname} '{{\"H0\": [60, 80]}}'")


# ── Settings persistence ───────────────────────────────────────────────────────

def save_settings(opts, run_dir: str, extra: dict | None = None):
    """Serialise the full resolved run configuration to settings.json."""
    d = {}

    # opts namespace — convert everything serialisable
    for key, val in vars(opts).items():
        try:
            json.dumps(val)
            d[key] = val
        except (TypeError, ValueError):
            d[key] = str(val)

    # Caller-supplied extras (labels, bounds, derived values)
    if extra:
        d.update(extra)

    # Runtime environment
    devices = jax.devices()
    d["environment"] = {
        "jax_version":   jax.__version__,
        "numpy_version": np.__version__,
        "healpy_version": hp.__version__,
        "jax_backend":   jax.default_backend(),
        "jax_devices":   [str(d_) for d_ in devices],
        "python_version": sys.version,
        "timestamp":     datetime.datetime.now().isoformat(),
    }

    path = os.path.join(run_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)

    return path


# ── Parameter table ────────────────────────────────────────────────────────────

def _print_parameter_table(labels, lower_bound, upper_bound, fixed_parameter_values: dict):
    """Pretty-print the sampled parameter space."""
    _section("Parameter Space")
    _row("Parameter", f"{'Lower':>12}  {'Upper':>12}  {'Fixed?':>8}")
    _row("─" * 24,    f"{'─' * 12}  {'─' * 12}  {'─' * 8}")
    for label, lo, hi in zip(labels, lower_bound, upper_bound):
        is_fixed = label in fixed_parameter_values or (lo == hi)
        fixed_str = f"= {lo:.4g}" if is_fixed else ""
        print(f"  │    {label:<24} {lo:>12.4g}  {hi:>12.4g}  {fixed_str}")
    _row("─" * 24,    f"{'─' * 12}  {'─' * 12}  {'─' * 8}")
    n_free  = sum(1 for lo, hi in zip(lower_bound, upper_bound) if lo != hi)
    n_fixed = len(labels) - n_free
    _row("Total parameters", len(labels))
    _row("Free (sampled)", n_free)
    if n_fixed > 0:
        _row("Fixed (collapsed)", n_fixed)
    _end()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = datetime.datetime.now()

    print()
    _banner(f"DARK SIRENS  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    # ──────────────────────────────────────────────────────────────
    # Argument parsing
    # ──────────────────────────────────────────────────────────────

    optp = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    # --- Data paths ---
    grp_data = optp.add_argument_group("Data")
    grp_data.add_argument("--gw_path",           required=True,
                          help="Path to GW posterior samples HDF5 file.")
    grp_data.add_argument("--gwselection_path",  required=True,
                          help="Path to injection / selection samples HDF5 file.")
    grp_data.add_argument("--survey_path",       default=None,
                          help="Path to pixelated galaxy survey HDF5 file "
                               "(required for dark_sirens / dark_sirens_complete).")
    grp_data.add_argument("--save_path",         default="./",
                          help="Root directory for output. Run sub-directory created automatically.")

    # --- Physical model ---
    grp_model = optp.add_argument_group("Physical model")
    grp_model.add_argument("--universe_model",  default="spectral_sirens",
                           choices=["spectral_sirens", "dark_sirens", "dark_sirens_complete"],
                           help="Redshift prior model.")
    grp_model.add_argument("--pop_model",       default="powerlaw+peak",
                           help="Population model name (e.g. powerlaw+peak, brokenpowerlaw+2peaks).")

    grp_model.add_argument("--fix_population",  type=str_to_bool, default=False,
                           metavar="BOOL",
                           help="Hold population parameters at fiducial values.")
    grp_model.add_argument("--fix_cosmology",   type=str_to_bool, default=False,
                           metavar="BOOL",
                           help="Hold H0, Om0 at Planck15 values.")
    grp_model.add_argument("--fix_survey",      type=str_to_bool, default=False,
                           metavar="BOOL",
                           help="Hold galaxy survey parameters fixed.")

    grp_model.add_argument("--prior_overrides", default=None,
                           metavar="JSON",
                           help='JSON dict of per-parameter bound overrides. '
                                'Example: \'{"H0": [60, 80]}\'')
    grp_model.add_argument("--fixed_parameter_values", default=None,
                           metavar="JSON",
                           help='JSON dict of parameters to hold fixed at specific values. '
                                'Example: \'{"$v_1$": 0.1}\'')

    # --- Survey / catalog options ---
    grp_cat = optp.add_argument_group("Catalog")
    grp_cat.add_argument("--sigma_kernel",  type=float, default=0.0,
                         help="KDE bandwidth for catalog galaxies [redshift]. "
                              "Use ~0.005 for spectroscopic, ~0.02 for photometric surveys.")
    grp_cat.add_argument("--use_LSS",       type=str_to_bool, default=False,
                         metavar="BOOL",
                         help="Compute and use the LSS overdensity field (memory-intensive).")

    # --- Sampler ---
    grp_sampler = optp.add_argument_group("Sampler")
    grp_sampler.add_argument("--sampler",    required=True,
                             choices=["jaxns", "dynesty", "emcee"],
                             help="Sampling algorithm.")
    grp_sampler.add_argument("--nlive",      type=int,   default=1000,
                             help="[jaxns/dynesty] Number of live points.")
    grp_sampler.add_argument("--dlogz",     type=float, default=0.1,
                             help="[dynesty] Evidence stopping criterion ΔlogZ.")
    grp_sampler.add_argument("--max_samples",type=int,  default=1_000_000,
                             help="[jaxns] Maximum number of likelihood evaluations.")
    grp_sampler.add_argument("--nwalkers",   type=int,   default=32,
                             help="[emcee] Number of walkers.")
    grp_sampler.add_argument("--nsteps",     type=int,   default=1000,
                             help="[emcee] Number of MCMC steps per walker.")
    grp_sampler.add_argument("--seed",       type=int,   default=22,
                             help="Random seed.")
    grp_sampler.add_argument("--show_progress", type=str_to_bool, default=True,
                             metavar="BOOL",
                             help="Show live sampling progress bar.")

    # --- Performance ---
    grp_perf = optp.add_argument_group("Performance")
    grp_perf.add_argument("--sel_batch_size", type=int, default=None,
                          metavar="N",
                          help="Process selection samples in chunks of N. "
                               "None = all at once (faster for standard models). "
                               "Set to e.g. 25000 for GP models that OOM with full selection arrays.")

    opts = optp.parse_args()

    # ──────────────────────────────────────────────────────────────
    # Parse JSON args
    # ──────────────────────────────────────────────────────────────

    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────

    _section("Validating configuration")

    GALAXY_AWARE = {"dark_sirens", "dark_sirens_complete"}

    if opts.universe_model in GALAXY_AWARE and not opts.survey_path:
        _fatal(f"--universe_model '{opts.universe_model}' requires --survey_path.")

    if opts.universe_model not in GALAXY_AWARE and opts.survey_path:
        _warn(f"--survey_path provided but universe_model='{opts.universe_model}' does not use it. "
              f"The catalog will be ignored.")

    if opts.fix_population and opts.fix_cosmology and opts.fix_survey:
        _warn("All parameter blocks are fixed — the likelihood is constant. "
              "Nothing will be inferred. Did you mean to fix everything?")

    if opts.sigma_kernel == 0.0 and opts.universe_model in GALAXY_AWARE:
        _warn("--sigma_kernel is 0.0. Galaxy redshift uncertainties will not be smoothed. "
              "For DESI-quality spectroscopic catalogs use sigma_kernel ~ 0.003–0.005; "
              "for photometric catalogs use ~ 0.02–0.05.")

    _ok("Configuration is valid.")
    _end()

    # ──────────────────────────────────────────────────────────────
    # Print resolved configuration
    # ──────────────────────────────────────────────────────────────

    devices     = jax.devices()
    backend     = jax.default_backend()
    device_strs = ", ".join(str(d) for d in devices)

    _section("Run Configuration")

    _row("Universe model",   opts.universe_model)
    _row("Population model", opts.pop_model)
    print("  │")

    _row("Fix cosmology",    "yes" if opts.fix_cosmology  else "no")
    _row("Fix population",   "yes" if opts.fix_population else "no")
    _row("Fix survey",       "yes" if opts.fix_survey     else "no")
    if prior_overrides:
        _row("Prior overrides", json.dumps(prior_overrides))
    if fixed_parameter_values:
        _row("Fixed values",    json.dumps(fixed_parameter_values))
    print("  │")

    _row("Sampler",          opts.sampler)
    if opts.sampler in ("jaxns", "dynesty"):
        _row("  live points",  opts.nlive)
    if opts.sampler == "dynesty":
        _row("  ΔlogZ stop",   opts.dlogz)
    if opts.sampler == "jaxns":
        _row("  max samples",  f"{opts.max_samples:,}")
    if opts.sampler == "emcee":
        _row("  walkers",      opts.nwalkers)
        _row("  steps",        opts.nsteps)
    _row("  seed",             opts.seed)
    print("  │")

    _row("JAX backend",      backend)
    _row("JAX devices",      device_strs)
    print("  │")

    _row("GW events path",   opts.gw_path)
    _row("Selection path",   opts.gwselection_path)
    if opts.survey_path:
        _row("Survey path",  opts.survey_path)
        _row("σ_kernel",     opts.sigma_kernel)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
    _row("Output root",      opts.save_path)
    if opts.sel_batch_size:
        _row("Sel. batch",   f"{opts.sel_batch_size:,} samples/batch")
    _end()

    # ──────────────────────────────────────────────────────────────
    # Load data
    # ──────────────────────────────────────────────────────────────

    _section("Loading data")
    print("  │")

    data = load_all_data(opts)

    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    nside   = data.get("nside", "N/A")

    _ok(f"GW posterior samples:   {nEvents} events × {nsamp} samples/event "
        f"= {nEvents * nsamp:,} total")
    _ok(f"Selection injections:   {int(Ndraw):,} total generated  "
        f"(detected subset loaded)")

    if opts.survey_path:
        ngals_pe  = data.get("ngals", None)
        ngals_sel = data.get("ngals_sel", None)
        _ok(f"HEALPix nside:          {nside}")
        if ngals_pe is not None:
            _ok(f"Catalog galaxies (PE pixels):  {int(np.asarray(ngals_pe).sum()):,}")
        if ngals_sel is not None:
            _ok(f"Catalog galaxies (sel pixels): {int(np.asarray(ngals_sel).sum()):,}")

    dg = data.get("delta_g_pix_z")
    if dg is not None:
        gb = np.asarray(dg).nbytes / 1e9
        _ok(f"δ_g field shape:        {np.asarray(dg).shape}  ({gb:.3f} GB)")

    _end()

    # ──────────────────────────────────────────────────────────────
    # Parameter space
    # ──────────────────────────────────────────────────────────────

    _section("Building parameter space")

    res = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        prior_overrides        = prior_overrides,
        fixed_parameter_values = fixed_parameter_values,
    )
    labels, lower_bound, upper_bound = res[0], res[1], res[2]
    n_pop_eff, n_cosmo_eff, n_survey_eff, model_name = res[3], res[7], res[8], res[9]

    pop_params_fid  = get_fixed_population_params(opts.pop_model)
    prior_transform = make_prior_transform(lower_bound, upper_bound)

    _ok(f"Parameter space built:  {len(labels)} dimensions")
    _end()

    _print_parameter_table(labels, lower_bound, upper_bound, fixed_parameter_values)

    # ──────────────────────────────────────────────────────────────
    # Build likelihood
    # ──────────────────────────────────────────────────────────────

    _section("Building likelihood")
    print("  │  Applying optimization barriers to catalog arrays...")
    print("  │  JIT compilation deferred to first call.")
    print("  │")

    likelihood = make_likelihood(
        opts                   = opts,
        data                   = data,
        pop_params_fid         = pop_params_fid,
        fixed_parameter_values = fixed_parameter_values,
    )
    _ok("Likelihood closure ready.")
    _end()

    # ──────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────

    _section(f"Sampling  [{opts.sampler.upper()}]")

    sampler_info = {
        "jaxns":   f"nlive={opts.nlive}  max_samples={opts.max_samples:,}  seed={opts.seed}",
        "dynesty": f"nlive={opts.nlive}  dlogz={opts.dlogz}  seed={opts.seed}",
        "emcee":   f"nwalkers={opts.nwalkers}  nsteps={opts.nsteps}  seed={opts.seed}",
    }
    _row("Configuration", sampler_info[opts.sampler])
    _row("ndim",          len(labels))
    print("  │")

    t_sample_start = datetime.datetime.now()

    results = run_sampler(
        method          = opts.sampler,
        likelihood      = likelihood,
        prior_transform = prior_transform,
        labels          = labels,
        lower_bound     = lower_bound,
        upper_bound     = upper_bound,
        opts            = opts,
    )

    t_sample_end  = datetime.datetime.now()
    wall_sampling = t_sample_end - t_sample_start

    print("  │")
    _ok(f"Sampling complete.  Wall time: {wall_sampling}")

    if results is not None:
        n_samples = results["samples"].shape[0]
        _ok(f"Posterior samples:  {n_samples:,}")
        logZ    = results.get("logZ")
        logZerr = results.get("logZerr")
        if logZ is not None:
            _ok(f"log Z = {logZ:.3f} ± {logZerr:.3f}" if logZerr else f"log Z = {logZ:.3f}")

    _end()

    # ──────────────────────────────────────────────────────────────
    # Save outputs
    # ──────────────────────────────────────────────────────────────

    if results is None:
        _fatal("Sampler returned no results.")

    t_end = datetime.datetime.now()
    timestamp = t_end.strftime("%Y-%m-%dT%H-%M-%S")
    run_name  = f"{opts.pop_model}__{opts.universe_model}__{opts.sampler}__{timestamp}"
    run_dir   = os.path.join(opts.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    _section("Saving outputs")
    _row("Run directory", run_dir)
    print("  │")

    # Settings JSON — full resolved state
    settings_path = save_settings(opts, run_dir, extra={
        "labels":                list(labels),
        "lower_bound":           list(map(float, lower_bound)),
        "upper_bound":           list(map(float, upper_bound)),
        "fixed_parameter_values": fixed_parameter_values,
        "prior_overrides":       prior_overrides,
        "n_pop_eff":             n_pop_eff,
        "n_cosmo_eff":           n_cosmo_eff,
        "n_survey_eff":          n_survey_eff,
        "model_name":            model_name,
        "sampler":               opts.sampler,
        "n_events":              nEvents,
        "n_samp_per_event":      nsamp,
        "n_draw":                int(Ndraw),
        "total_runtime":         str(t_end - t_start),
        "sampling_runtime":      str(wall_sampling),
    })
    _ok(f"settings.json  →  {settings_path}")

    # Samples
    samples_path = os.path.join(run_dir, "samples.npy")
    np.save(samples_path, results)
    _ok(f"samples.npy    →  {samples_path}")

    # Corner plot
    print("  │  Generating corner plot...")
    try:
        fig = make_production_corner(results["samples"], labels)
        corner_path = os.path.join(run_dir, "corner.pdf")
        fig.savefig(corner_path, bbox_inches="tight", dpi=200)
        _ok(f"corner.pdf     →  {corner_path}")
    except Exception as e:
        _warn(f"Corner plot failed: {e}")

    _end()

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────

    total = t_end - t_start
    print()
    _banner(f"DONE  │  total wall time {total}")
    print()


if __name__ == "__main__":
    main()