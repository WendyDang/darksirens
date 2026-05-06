"""
likelihood.py
-------------
Hierarchical dark-siren log-likelihood.

Structure
---------
This module is a *thin orchestrator*.  The heavy lifting lives in:

    inference/utils.py       — log_sample_weight, log_jacobian_dL_to_z
    inference/selection.py   — compute_selection_term, selection_log_correction
    inference/events.py      — make_gw_event (barrier wrapping, q pre-computation)

darksiren_log_likelihood
    Assembles log_wt closure → selection term + PE term → total log-likelihood.
    Decorated with @jax.jit; static args trigger recompilation only when the
    model or sampler changes, not on every proposal.

make_likelihood
    Closure factory called once at startup.  Applies optimization barriers
    to catalog arrays and GW data before they are captured by JIT.
    Returns a scalar callable ``likelihood(coord) → log_likelihood``.

Barrier strategy
----------------
``lax.optimization_barrier`` must be applied BEFORE arrays enter any JIT
closure, i.e. here in make_likelihood, not inside darksiren_log_likelihood.
Inside a JIT body the arrays are already abstract tracers and the barrier
has no effect on constant-folding.

All floating-point data arrays (GW samples, catalog) are barrier-wrapped.
The single canonical barrier helper is ``inference.events._barrier``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from jax.scipy.special import logsumexp

from astropy.cosmology import Planck15

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em import get_redshift_prior
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

from darksirens.inference.events import make_gw_event, pad_gw_event_to_multiple, _barrier
from darksirens.inference.utils import log_sample_weight
from darksirens.inference.selection import (
    compute_selection_term,
    selection_log_correction,
)

# Fiducial cosmology (Planck15)
H0_FID  = float(Planck15.H0.value)
OM0_FID = float(Planck15.Om0)

# Fiducial survey params: log10(n0), z50, w, delta, b_miss, alpha
SURVEY_PARAMS_FID = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])


# ============================================================
# Core jitted likelihood
# ============================================================

@partial(
    jax.jit,
    static_argnames=[
        "nEvents", "nsamp", "pop_model", "universe_model", "sel_batch_size",
    ],
)
def darksiren_log_likelihood(
    cosmo:          CosmoParams,
    survey:         SurveyParams,
    pop_params:     jnp.ndarray,
    gw_pe:          GWEvent,
    em_catalog_pe:  EMCatalog,
    gw_sel:         GWEvent,
    em_catalog_sel: EMCatalog,
    nEvents:        int,
    nsamp:          int,
    Ndraw:          float,
    pop_model:      str,
    universe_model: str,
    sel_batch_size: int | None = None,
) -> jnp.ndarray:
    """
    Hierarchical dark-siren log-likelihood.

    Returns log p({d_i} | cosmo, survey, pop_params).

    Parameters
    ----------
    cosmo, survey, pop_params
        Inference parameters.
    gw_pe, em_catalog_pe
        GW posterior samples and associated EM catalog for the PE term.
    gw_sel, em_catalog_sel
        Injection samples and associated EM catalog for the selection term.
    nEvents
        Number of observed GW events (static — recompiles if changed).
    nsamp
        PE samples per event (static).
    Ndraw
        Total number of injections drawn for the selection integral.
    pop_model, universe_model
        String keys selecting the population and redshift-prior models (static).
    sel_batch_size
        If not None, process selection samples in chunks via lax.scan.
        The selection GWEvent must be pre-padded to a multiple of this value.
    """
    log_p_pop        = pop_model_parser(pop_model=pop_model)
    raw_log_prior_z  = get_redshift_prior(universe_model)

    # Guard -inf → -1e6 to keep the scan numerically stable.
    # Done here (not in utils.py) because the guard is an inference policy,
    # not a property of the prior itself.
    def log_prior_z(z, pix, catalog):
        lp = raw_log_prior_z(z, pix, cosmo, survey, catalog)
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    # Single weight kernel — used identically for PE and selection.
    def log_wt(m1det, q, dL, chieff, pix, prior_wt, catalog):
        return log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            log_p_pop, log_prior_z,
        )

    # ------------------------------------------------------------------
    # Selection term
    # ------------------------------------------------------------------
    log_mu, Neff = compute_selection_term(
        gw_sel, em_catalog_sel, log_wt, Ndraw, nEvents, sel_batch_size
    )
    ll = selection_log_correction(log_mu, Neff, nEvents)

    # ------------------------------------------------------------------
    # PE term: scan over events to keep peak memory O(nsamp × N_grid)
    # ------------------------------------------------------------------
    def _pe_event_fn(_, event_idx):
        s  = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
        ldw = log_wt(
            sl(gw_pe.m1det),
            sl(gw_pe.q),
            sl(gw_pe.dL),
            sl(gw_pe.chieff),
            sl(gw_pe.pixels),
            sl(gw_pe.prior_wt),
            em_catalog_pe,
        )
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


# ============================================================
# Likelihood closure factory
# ============================================================

def make_likelihood(
    opts,
    data: dict,
    pop_params_fid,
    fixed_parameter_values: dict | None = None,
):
    """
    Build and return the likelihood callable for the sampler.

    This function runs *once* at startup.  It:
      1. Applies ``lax.optimization_barrier`` to all GW and catalog arrays.
      2. Pre-pads the selection GWEvent if ``sel_batch_size`` is set.
      3. Returns a scalar ``likelihood(coord) → log_likelihood`` that the
         sampler calls millions of times.

    Parameters
    ----------
    opts
        Argument namespace.  Used attributes:
        ``pop_model``, ``universe_model``, ``fix_cosmology``,
        ``fix_population``, ``fix_survey``, ``sel_batch_size``.
    data
        Dictionary produced by ``load_all_data``.
    pop_params_fid
        Fiducial population parameters (used when ``fix_population=True``).
    fixed_parameter_values
        Dict of {label: value} for parameters held fixed inside the
        sampled coordinate vector (finer-grained than the block-level flags).
    """
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    # ------------------------------------------------------------------
    # Sizes and static options
    # ------------------------------------------------------------------
    nEvents        = data["nEvents"]
    nsamp          = data["nsamp"]
    Ndraw          = data["Ndraw"]
    apix           = data["apix"]
    pop_model      = opts.pop_model
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)

    # ------------------------------------------------------------------
    # Catalog arrays — barrier-wrapped before JIT closure capture
    # ------------------------------------------------------------------
    def _to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    def _load_catalog(prefix):
        return dict(
            zgals  = _barrier(_to_jax(f"zgals_{prefix}")),
            dzgals = _barrier(_to_jax(f"dzgals_{prefix}")),
            wgals  = _barrier(_to_jax(f"wgals_{prefix}")),
            ngals  = _barrier(_to_jax(f"ngals_{prefix}")),
        )

    # delta_g_pix_z is shared between PE and selection catalogs
    delta_g_pix_z = _barrier(_to_jax("delta_g_pix_z"))
    sigma_kernel  = data["sigma_kernel"]

    cat_pe  = _load_catalog("pe")
    cat_sel = _load_catalog("sel")

    # ------------------------------------------------------------------
    # GW data arrays — barrier-wrapped via make_gw_event
    # All arrays (including q) receive barriers in one canonical place.
    # ------------------------------------------------------------------
    gw_pe = make_gw_event(
        m1det    = data["m1det"],
        m2det    = data["m2det"],
        dL       = data["dL"],
        chieff   = data["chieff"],
        prior_wt = data["p_pe"],
        pixels   = data["pixels_pe"],
    )

    gw_sel_raw = make_gw_event(
        m1det    = data["m1detsels"],
        m2det    = data["m2detsels"],
        dL       = data["dLsels"],
        chieff   = data["chieffsels"],
        prior_wt = data["p_draw"],
        pixels   = data["pixels_sel"],
    )

    # Pad selection event if batching is requested
    if sel_batch_size is not None:
        gw_sel, n_pad = pad_gw_event_to_multiple(gw_sel_raw, sel_batch_size)
        if n_pad > 0:
            N_sel     = gw_sel_raw.dL.shape[0]
            N_batches = (N_sel + n_pad) // sel_batch_size
            print(
                f"    [sel_batch] {N_sel} → {N_sel + n_pad} samples "
                f"({N_batches} × {sel_batch_size}, {n_pad} padding entries)"
            )
    else:
        gw_sel = gw_sel_raw

    # ------------------------------------------------------------------
    # Parameter space bookkeeping
    # ------------------------------------------------------------------
    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    cosmo_labels        = ["H0", "Om0"]
    survey_labels       = ["log10n0", "z50", "w", "delta", "b_miss", "alpha"]

    # Convert fiducial pop params to plain Python list NOW (before JIT closure).
    # Indexing a JAX array inside a JIT body returns an abstract tracer —
    # float() on a tracer raises ConcretizationTypeError.
    pop_params_fid_list = [float(v) for v in pop_params_fid]

    sampled_labels = []
    if not opts.fix_cosmology:  sampled_labels += cosmo_labels
    if not opts.fix_population: sampled_labels += pop_labels
    if not opts.fix_survey:     sampled_labels += survey_labels

    # Parameters that appear in the coordinate vector but are pinned to a value
    fixed_in_coord = {
        k: float(v)
        for k, v in fixed_parameter_values.items()
        if k in set(sampled_labels)
    }

    # ------------------------------------------------------------------
    # Inner likelihood callable
    # ------------------------------------------------------------------
    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        coord = jnp.asarray(coord)

        # --- Unpack coordinate vector ---
        values = {}
        offset = 0
        for label in sampled_labels:
            if label in fixed_in_coord:
                values[label] = fixed_in_coord[label]
                continue
            if offset >= coord.shape[0]:
                raise ValueError(
                    f"Too few coordinates: needed index {offset} for '{label}', "
                    f"got {coord.shape[0]}."
                )
            values[label] = coord[offset]
            offset += 1

        if offset != coord.shape[0]:
            raise ValueError(
                f"Coordinate mismatch: consumed {offset}, got {coord.shape[0]}."
            )

        def _get(label, default):
            if label in values:
                return values[label]
            if label in fixed_parameter_values:
                return fixed_parameter_values[label]
            return default

        # --- Cosmology ---
        H0  = _get("H0",  H0_FID)  if opts.fix_cosmology  else values["H0"]
        Om0 = _get("Om0", OM0_FID) if opts.fix_cosmology  else values["Om0"]

        # --- Population ---
        if opts.fix_population:
            pop_params = jnp.array([
                _get(label, pop_params_fid_list[i])
                for i, label in enumerate(pop_labels)
            ])
        else:
            pop_params = jnp.array([values[label] for label in pop_labels])

        # --- Survey ---
        if opts.fix_survey:
            sp = jnp.array([
                _get(label, float(SURVEY_PARAMS_FID[i]))
                for i, label in enumerate(survey_labels)
            ])
        else:
            sp = jnp.array([values[label] for label in survey_labels])

        # --- Build PyTree containers ---
        cosmo  = CosmoParams(H0=H0, Om0=Om0)
        survey = SurveyParams(
            n0     = 10.0 ** sp[0],
            z50    = sp[1],
            w      = sp[2],
            delta  = sp[3],
            b_miss = sp[4],
            alpha  = sp[5],
        )

        em_catalog_pe = EMCatalog(
            apix          = apix,
            zgals         = cat_pe["zgals"],
            dzgals        = cat_pe["dzgals"],
            wgals         = cat_pe["wgals"],
            ngals         = cat_pe["ngals"],
            delta_g_pix_z = delta_g_pix_z,
            sigma_kernel  = sigma_kernel,
        )
        em_catalog_sel = EMCatalog(
            apix          = apix,
            zgals         = cat_sel["zgals"],
            dzgals        = cat_sel["dzgals"],
            wgals         = cat_sel["wgals"],
            ngals         = cat_sel["ngals"],
            delta_g_pix_z = delta_g_pix_z,
            sigma_kernel  = sigma_kernel,
        )

        return darksiren_log_likelihood(
            cosmo, survey, pop_params,
            gw_pe,  em_catalog_pe,
            gw_sel, em_catalog_sel,
            nEvents, nsamp, Ndraw,
            pop_model, universe_model,
            sel_batch_size=sel_batch_size,
        )

    return likelihood