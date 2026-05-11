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

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import h5py

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.gw.populations import get_fixed_population_params, pop_model_prior_parser
from darksirens.em.utils import load_survey
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.inference.likelihood import make_likelihood
from darksirens.inference.sampling import run_sampler
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.utils.plotting import make_production_corner

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
warnings.simplefilter("ignore", FutureWarning)


# ── Formatting helpers ─────────────────────────────────────────────────────────

W = 72

def _banner(text: str):
    pad   = max(0, W - 4 - len(text))
    left  = pad // 2
    right = pad - left
    print(f"{'─' * W}")
    print(f"  {'·' * left} {text} {'·' * right}  ")
    print(f"{'─' * W}")

def _section(title: str):
    print()
    print(f"  ┌─ {title} {'─' * max(0, W - 6 - len(title))}┐")

def _row(label: str, value, width: int = 26):
    print(f"  │  {label:<{width}} {value}")

def _end():
    print(f"  └{'─' * (W - 3)}┘")

def _ok(msg: str):   print(f"  ✓  {msg}")
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


def parse_counterpart_arg(value: list[str] | None) -> tuple[float, float, float] | None:
    """Parse ``--counterpart RA DEC Z`` into floats.

    Angles are expected in radians, matching the GW sample convention used by
    ``load_gw_samples`` and HEALPix indexing throughout the pipeline.
    """
    if value is None:
        return None
    if len(value) != 3:
        _fatal("--counterpart requires exactly three values: RA DEC Z (angles in radians).")
    try:
        ra, dec, z = (float(x) for x in value)
    except ValueError as e:
        _fatal(f"--counterpart values must be numeric RA DEC Z. Error: {e}")
    if not (0.0 <= ra < 2.0 * np.pi):
        _fatal("--counterpart RA must be in radians with 0 <= RA < 2π.")
    if not (-0.5 * np.pi <= dec <= 0.5 * np.pi):
        _fatal("--counterpart Dec must be in radians with -π/2 <= Dec <= π/2.")
    if z <= 0.0:
        _fatal("--counterpart redshift Z must be positive.")
    return ra, dec, z


# ── Parameter table ────────────────────────────────────────────────────────────

def _print_parameter_table(
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    fix_cosmology:          bool,
    fix_population:         bool,
    fix_survey:             bool,
    pop_params_fid:         np.ndarray,
    pop_labels_all:         list,
):
    """
    Print sampled parameters with bounds, individually-fixed params with their
    values, and block-fixed parameters with their fiducial values.
    """
    COSMO_FID  = {"H0": 67.74, "Om0": 0.3075}
    SURVEY_FID = {"log10n0": -2.0, "z50": 1.0, "w": 0.5,
                  "delta": 0.0, "b_miss": 1.0, "alpha": 0.5}
    pop_fid_map = {lbl: float(pop_params_fid[i])
                   for i, lbl in enumerate(pop_labels_all)}

    block_fixed: dict[str, float] = {}
    if fix_cosmology:  block_fixed.update(COSMO_FID)
    if fix_population: block_fixed.update(pop_fid_map)
    if fix_survey:     block_fixed.update(SURVEY_FID)

    _section("Parameter Space")
    _row("Parameter", f"{'Lower':>12}  {'Upper':>12}  Status", width=24)
    _row("─" * 24,    f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)

    for label, lo, hi in zip(labels, lower_bound, upper_bound):
        if label in fixed_parameter_values:
            val  = fixed_parameter_values[label]
            note = f"fixed = {val:.6g}"
        elif label in block_fixed:
            note = f"fixed = {block_fixed[label]:.6g}  (block)"
        else:
            note = "← overridden" if label in prior_overrides else ""
        print(f"  │    {label:<24} {lo:>12.4g}  {hi:>12.4g}  {note}")

    _row("─" * 24, f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)

    n_free = sum(1 for lo, hi in zip(lower_bound, upper_bound) if lo != hi)
    n_fix_ind   = len(fixed_parameter_values)
    n_fix_block = (2 if fix_cosmology else 0
                   + len(pop_labels_all) if fix_population else 0
                   + 6 if fix_survey else 0)
    _row("Free (sampled)",      n_free)
    if n_fix_ind:   _row("Fixed individually", n_fix_ind)
    if n_fix_block: _row("Fixed (block)",      n_fix_block)
    _row("Total in coord vec",  len(labels))
    _end()


# ── Data saving ────────────────────────────────────────────────────────────────

def save_results_hdf5(
    results:                dict,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    opts,
    meta:                   dict,
) -> str:
    """
    Save posterior samples and all metadata to a single HDF5 file.

    Structure
    ---------
    results.hdf5
    ├── attrs           — run metadata (model, sampler, evidence, runtime, …)
    ├── samples         — (N_samples, N_dim) posterior samples
    ├── labels          — (N_dim,) parameter label strings (UTF-8)
    ├── lower_bound     — (N_dim,) prior lower bounds
    ├── upper_bound     — (N_dim,) prior upper bounds
    ├── fixed_labels    — (N_fixed,) individually-fixed param labels
    ├── fixed_values    — (N_fixed,) their values
    ├── log_weights     — (N_samples,) log importance weights  [if available]
    └── log_likelihood  — (N_samples,) per-sample log-likelihoods [if available]
    """
    path = os.path.join(run_dir, "results.hdf5")
    kw   = dict(compression="gzip", shuffle=True)
    dt   = h5py.string_dtype(encoding="utf-8")

    samples = np.asarray(results["samples"])
    N, ndim = samples.shape

    with h5py.File(path, "w") as f:

        # Samples and bounds
        f.create_dataset("samples",     data=samples,                          **kw)
        f.create_dataset("lower_bound", data=np.array(lower_bound, dtype=float), **kw)
        f.create_dataset("upper_bound", data=np.array(upper_bound, dtype=float), **kw)
        f.create_dataset("labels",      data=np.array(labels, dtype=object), dtype=dt)

        # Optional per-sample arrays
        if results.get("log_weights") is not None:
            f.create_dataset("log_weights",    data=np.asarray(results["log_weights"]), **kw)
        if results.get("log_likelihood") is not None:
            f.create_dataset("log_likelihood", data=np.asarray(results["log_likelihood"]), **kw)

        # Individually-fixed parameters — store so post-processing can reconstruct
        # the full parameter vector without reading the settings JSON separately.
        if fixed_parameter_values:
            fix_labels = list(fixed_parameter_values.keys())
            fix_vals   = [float(v) for v in fixed_parameter_values.values()]
            f.create_dataset("fixed_labels", data=np.array(fix_labels, dtype=object), dtype=dt)
            f.create_dataset("fixed_values", data=np.array(fix_vals, dtype=float), **kw)

        # Run metadata
        f.attrs["pop_model"]       = opts.pop_model
        f.attrs["universe_model"]  = opts.universe_model
        f.attrs["sampler"]         = opts.sampler
        f.attrs["fix_cosmology"]   = bool(opts.fix_cosmology)
        f.attrs["fix_population"]  = bool(opts.fix_population)
        f.attrs["fix_survey"]      = bool(opts.fix_survey)
        f.attrs["gw_path"]         = opts.gw_path
        f.attrs["gwselection_path"] = opts.gwselection_path
        f.attrs["survey_path"]     = opts.survey_path or ""
        if getattr(opts, "counterpart", None) is not None:
            f.attrs["counterpart_ra"] = float(opts.counterpart[0])
            f.attrs["counterpart_dec"] = float(opts.counterpart[1])
            f.attrs["counterpart_z"] = float(opts.counterpart[2])
            f.attrs["counterpart_dz"] = float(opts.counterpart_dz)
            f.attrs["counterpart_nside"] = int(opts.counterpart_nside)
        f.attrs["sigma_kernel"]    = float(opts.sigma_kernel)
        f.attrs["nlive"]           = int(opts.nlive)
        f.attrs["dlogz"]           = float(opts.dlogz)
        f.attrs["nwalkers"]        = int(opts.nwalkers)
        f.attrs["nsteps"]          = int(opts.nsteps)
        f.attrs["seed"]            = int(opts.seed)
        f.attrs["n_samples"]       = N
        f.attrs["n_dim"]           = ndim
        f.attrs["n_events"]        = int(meta["n_events"])
        f.attrs["n_samp_per_event"] = int(meta["n_samp_per_event"])
        f.attrs["n_draw"]          = int(meta["n_draw"])
        f.attrs["total_runtime"]   = meta["total_runtime"]
        f.attrs["sampling_runtime"] = meta["sampling_runtime"]
        f.attrs["timestamp"]       = meta["timestamp"]

        logZ    = results.get("logZ")
        logZerr = results.get("logZerr")
        if logZ is not None:
            f.attrs["logZ"]    = float(logZ)
            f.attrs["logZerr"] = float(logZerr) if logZerr is not None else float("nan")

        if prior_overrides:
            f.attrs["prior_overrides"] = json.dumps(prior_overrides)

        f.attrs["environment"] = json.dumps({
            "jax_version":    jax.__version__,
            "numpy_version":  np.__version__,
            "healpy_version": hp.__version__,
            "jax_backend":    jax.default_backend(),
            "jax_devices":    [str(d) for d in jax.devices()],
            "python_version": sys.version,
        })

    return path


def save_settings_json(
    opts,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    meta:                   dict,
) -> str:
    """Human-readable settings.json for easy inspection and re-runs."""
    d: dict = {}

    for key, val in vars(opts).items():
        try:
            json.dumps(val)
            d[key] = val
        except (TypeError, ValueError):
            d[key] = str(val)

    # Emit None explicitly so it's obvious when not set — not an empty dict
    d["fixed_parameter_values"] = fixed_parameter_values if fixed_parameter_values else None
    d["prior_overrides"]        = prior_overrides        if prior_overrides        else None

    d["labels"]      = list(labels)
    d["lower_bound"] = list(map(float, lower_bound))
    d["upper_bound"] = list(map(float, upper_bound))
    d.update(meta)

    d["environment"] = {
        "jax_version":    jax.__version__,
        "numpy_version":  np.__version__,
        "healpy_version": hp.__version__,
        "jax_backend":    jax.default_backend(),
        "jax_devices":    [str(dv) for dv in jax.devices()],
        "python_version": sys.version,
    }

    path = os.path.join(run_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = datetime.datetime.now()
    print()
    _banner(f"DARK SIRENS  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    # ── Argument parsing ───────────────────────────────────────────

    optp = ArgumentParser(description=__doc__,
                          formatter_class=RawDescriptionHelpFormatter)

    g = optp.add_argument_group("Data")
    g.add_argument("--gw_path",          required=True)
    g.add_argument("--gwselection_path", required=True)
    g.add_argument("--survey_path",      default=None)
    g.add_argument("--save_path",        default="./")

    g = optp.add_argument_group("Physical model")
    g.add_argument("--universe_model", default="spectral_sirens",
                   choices=["spectral_sirens", "dark_sirens", "dark_sirens_complete", "bright_sirens"])
    g.add_argument("--pop_model",       default="powerlaw+peak")
    g.add_argument("--fix_population",  type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--fix_cosmology",   type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--fix_survey",      type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--prior_overrides", default=None, metavar="JSON")
    g.add_argument("--fixed_parameter_values", default=None, metavar="JSON")
    g.add_argument("--counterpart", nargs=3, metavar=("RA", "DEC", "Z"),
                   help="Bright-siren counterpart coordinates and redshift; angles are radians.")
    g.add_argument("--counterpart_dz", type=float, default=1.0e-4,
                   help="Gaussian redshift uncertainty for --counterpart.")
    g.add_argument("--counterpart_nside", type=int, default=1,
                   help="HEALPix NSIDE for the synthetic bright-siren counterpart catalog.")

    g = optp.add_argument_group("Catalog")
    g.add_argument("--sigma_kernel", type=float, default=0.0)
    g.add_argument("--use_LSS",      type=str_to_bool, default=False, metavar="BOOL")

    g = optp.add_argument_group("Sampler")
    g.add_argument("--sampler",      required=True, choices=["jaxns", "dynesty", "emcee"])
    g.add_argument("--nlive",        type=int,   default=1000)
    g.add_argument("--dlogz",        type=float, default=0.1)
    g.add_argument("--max_samples",  type=int,   default=1_000_000)
    g.add_argument("--nwalkers",     type=int,   default=32)
    g.add_argument("--nsteps",       type=int,   default=1000)
    g.add_argument("--seed",         type=int,   default=22)
    g.add_argument("--show_progress",type=str_to_bool, default=True, metavar="BOOL")

    g = optp.add_argument_group("Performance")
    g.add_argument("--sel_batch_size", type=int, default=None, metavar="N")

    opts = optp.parse_args()

    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    opts.counterpart       = parse_counterpart_arg(opts.counterpart)

    if opts.universe_model == "bright_sirens":
        # Bright sirens use a synthetic one-object catalog fixed by the
        # counterpart rather than survey-completion hyperparameters.
        opts.fix_survey = True

    # ── Validation ─────────────────────────────────────────────────

    _section("Validating configuration")
    GALAXY_AWARE = {"dark_sirens", "dark_sirens_complete"}

    if opts.universe_model == "bright_sirens" and opts.counterpart is None:
        _fatal("'bright_sirens' requires --counterpart RA DEC Z (angles in radians).")
    if opts.universe_model != "bright_sirens" and opts.counterpart is not None:
        _warn("--counterpart is ignored unless --universe_model bright_sirens.")
    if opts.counterpart_dz <= 0.0:
        _fatal("--counterpart_dz must be positive.")
    if opts.counterpart_nside < 1 or not hp.isnsideok(opts.counterpart_nside):
        _fatal("--counterpart_nside must be a valid positive HEALPix NSIDE.")

    if opts.universe_model in GALAXY_AWARE and not opts.survey_path:
        _fatal(f"'{opts.universe_model}' requires --survey_path.")
    if opts.universe_model not in GALAXY_AWARE and opts.survey_path:
        _warn(f"--survey_path provided but '{opts.universe_model}' does not use it.")
    if opts.fix_population and opts.fix_cosmology and opts.fix_survey:
        _warn("All blocks fixed — nothing will be inferred.")
    if opts.sigma_kernel == 0.0 and opts.universe_model in GALAXY_AWARE:
        _warn("--sigma_kernel=0 — galaxy redshift uncertainties unsmoothed.")

    _ok("Configuration is valid.")
    _end()

    # ── Run configuration printout ─────────────────────────────────

    _section("Run Configuration")
    _row("Universe model",   opts.universe_model)
    if opts.counterpart is not None:
        ra_cp, dec_cp, z_cp = opts.counterpart
        _row("Counterpart", f"ra={ra_cp:.8g}, dec={dec_cp:.8g}, z={z_cp:.8g}")
        _row("Counterpart dz", opts.counterpart_dz)
        _row("Counterpart nside", opts.counterpart_nside)
    _row("Population model", opts.pop_model)
    print("  │")
    _row("Fix cosmology",    "yes" if opts.fix_cosmology  else "no")
    _row("Fix population",   "yes" if opts.fix_population else "no")
    _row("Fix survey",       "yes" if opts.fix_survey     else "no")
    _row("Prior overrides",  json.dumps(prior_overrides) if prior_overrides else "none")
    if fixed_parameter_values:
        for lbl, val in fixed_parameter_values.items():
            _row(f"  fixed: {lbl}", val)
    else:
        _row("Fixed param values", "none")
    print("  │")
    _row("Sampler", opts.sampler)
    if opts.sampler in ("jaxns", "dynesty"):
        _row("  live points", opts.nlive)
    if opts.sampler == "dynesty":
        _row("  ΔlogZ stop",  opts.dlogz)
    if opts.sampler == "jaxns":
        _row("  max samples", f"{opts.max_samples:,}")
    if opts.sampler == "emcee":
        _row("  walkers", opts.nwalkers)
        _row("  steps",   opts.nsteps)
    _row("  seed", opts.seed)
    print("  │")
    _row("JAX backend", jax.default_backend())
    _row("JAX devices",  ", ".join(str(d) for d in jax.devices()))
    print("  │")
    _row("GW events path",  opts.gw_path)
    _row("Selection path",  opts.gwselection_path)
    if opts.survey_path:
        _row("Survey path",  opts.survey_path)
        _row("σ_kernel",     opts.sigma_kernel)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
    _row("Output root",     opts.save_path)
    if opts.sel_batch_size:
        _row("Sel. batch",   f"{opts.sel_batch_size:,} samples/batch")
    _end()

    # ── Load data ──────────────────────────────────────────────────

    _section("Loading data")
    print("  │")
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)

    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    nside   = data.get("nside", "N/A")

    _ok(f"GW posterior samples:   {nEvents} events × {nsamp} samples/event = {nEvents*nsamp:,} total")
    _ok(f"Selection injections:   {int(Ndraw):,} total generated")

    if opts.survey_path:
        ngals_pe  = data.get("ngals_pe",  None)
        ngals_sel = data.get("ngals_sel", None)
        _ok(f"HEALPix nside:          {nside}")
        if ngals_pe  is not None:
            _ok(f"Catalog galaxies (PE pixels):  {int(np.asarray(ngals_pe).sum()):,}")
        if ngals_sel is not None:
            _ok(f"Catalog galaxies (sel pixels): {int(np.asarray(ngals_sel).sum()):,}")
        catalog_memory = data.get("catalog_memory")
        if catalog_memory is not None:
            _ok(
                "Unique catalog pixels:   "
                f"PE {catalog_memory['unique_pe_pixels']:,}, "
                f"selection {catalog_memory['unique_sel_pixels']:,}"
            )
            _ok(
                "Duplicated catalog bytes avoided: "
                f"{catalog_memory['duplicated_catalog_bytes_avoided'] / 1e9:.3f} GB"
            )
            _ok(
                "Max galaxies/unique pixel: "
                f"{catalog_memory['max_galaxies_per_unique_pixel']:,}"
            )

    dg = data.get("delta_g_pix_z")
    if dg is not None:
        gb = np.asarray(dg).nbytes / 1e9
        _ok(f"δ_g field shape:        {np.asarray(dg).shape}  ({gb:.3f} GB)")
    _end()

    # ── Parameter space ────────────────────────────────────────────

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

    _, _, pop_labels_all, _ = pop_model_prior_parser(opts.pop_model)
    pop_params_fid  = get_fixed_population_params(opts.pop_model)
    prior_transform = make_prior_transform(lower_bound, upper_bound)

    _ok(f"Parameter space built:  {len(labels)} free dimensions")
    _end()

    _print_parameter_table(
        labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides,
        opts.fix_cosmology, opts.fix_population, opts.fix_survey,
        pop_params_fid, pop_labels_all,
    )

    # ── Build likelihood ───────────────────────────────────────────

    _section("Building likelihood")
    print("  │  Applying optimization barriers...")
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

    # ── Sampling ───────────────────────────────────────────────────

    _section(f"Sampling  [{opts.sampler.upper()}]")
    sampler_info = {
        "jaxns":   f"nlive={opts.nlive}  max_samples={opts.max_samples:,}  seed={opts.seed}",
        "dynesty": f"nlive={opts.nlive}  dlogz={opts.dlogz}  seed={opts.seed}",
        "emcee":   f"nwalkers={opts.nwalkers}  nsteps={opts.nsteps}  seed={opts.seed}",
    }
    _row("Configuration", sampler_info[opts.sampler])
    _row("ndim", len(labels))
    print("  │")

    t_sample_start = datetime.datetime.now()
    results = run_sampler(
        method=opts.sampler, likelihood=likelihood,
        prior_transform=prior_transform, labels=labels,
        lower_bound=lower_bound, upper_bound=upper_bound, opts=opts,
    )
    t_sample_end  = datetime.datetime.now()
    wall_sampling = t_sample_end - t_sample_start

    if results is None or results.get("samples") is None:
        _fatal("Sampler returned no results.")

    n_samples = np.asarray(results["samples"]).shape[0]
    print("  │")
    _ok(f"Sampling complete.  Wall time: {wall_sampling}")
    _ok(f"Posterior samples:  {n_samples:,}")

    logZ    = results.get("logZ")
    logZerr = results.get("logZerr")
    if logZ is not None:
        zerr = float(logZerr) if logZerr is not None else float("nan")
        _ok(f"log Z = {float(logZ):.3f} ± {zerr:.3f}")
    _end()

    # ── Save outputs ───────────────────────────────────────────────

    t_end     = datetime.datetime.now()
    timestamp = t_end.strftime("%Y-%m-%dT%H-%M-%S")
    run_name  = f"{opts.pop_model}__{opts.universe_model}__{opts.sampler}__{timestamp}"
    run_dir   = os.path.join(opts.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    meta = {
        "n_events":         nEvents,
        "n_samp_per_event": nsamp,
        "n_draw":           int(Ndraw),
        "n_pop_eff":        n_pop_eff,
        "n_cosmo_eff":      n_cosmo_eff,
        "n_survey_eff":     n_survey_eff,
        "model_name":       model_name,
        "total_runtime":    str(t_end - t_start),
        "sampling_runtime": str(wall_sampling),
        "timestamp":        timestamp,
    }

    _section("Saving outputs")
    _row("Run directory", run_dir)
    print("  │")

    hdf5_path = save_results_hdf5(
        results, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, opts, meta,
    )
    _ok(f"results.hdf5   →  {hdf5_path}")

    json_path = save_settings_json(
        opts, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, meta,
    )
    _ok(f"settings.json  →  {json_path}")

    print("  │  Generating corner plot...")
    try:
        fig = make_production_corner(results["samples"], labels)
        corner_path = os.path.join(run_dir, "corner.pdf")
        fig.savefig(corner_path, bbox_inches="tight", dpi=200)
        _ok(f"corner.pdf     →  {corner_path}")
    except Exception as e:
        _warn(f"Corner plot failed: {e}")

    _end()

    print()
    _banner(f"DONE  │  total wall time {t_end - t_start}")
    print()


if __name__ == "__main__":
    main()