"""
likelihood.py
-------------
Hierarchical dark-siren log-likelihood.

Structure
---------
darksiren_log_likelihood   — JAX-jitted likelihood, stateless
    _sel_log_weights        — log weights for one batch of selection samples
    _sel_term               — selection integral μ  (batched or unbatched)
    _pe_term                — per-event log-likelihood (always event-scanned)

make_likelihood            — closure factory: captures data, returns callable
    _barrier                — apply optimization_barrier to large catalog arrays
                             before they are seen by JAX tracing

RAM note
--------
The optimization barriers MUST be applied before the arrays are captured
in the JIT closure (i.e. in make_likelihood, not inside likelihood()).
If barriers are placed inside the likelihood function, JAX has already
ingested the raw arrays during tracing, and XLA's constant-folding pass
will still try to evaluate per-pixel operations at compile time, producing
multi-gigabyte intermediate HLO tensors and exhausting RAM.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from jax.scipy.special import logsumexp
import numpy as np

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em import get_redshift_prior
from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.utils import logdiffexp
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

from astropy.cosmology import Planck15

# Fiducial cosmology
H0_FID   = float(Planck15.H0.value)
OM0_FID  = float(Planck15.Om0)

# Fiducial survey params: log10(n0), z50, w, delta, b_miss, alpha
SURVEY_PARAMS_FID = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])


# ============================================================
# Core jitted likelihood
# ============================================================

@partial(
    jax.jit,
    static_argnames=["nEvents", "nsamp", "pop_model", "universe_model",
                     "sel_batch_size"],
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
    Hierarchical dark-siren log-posterior likelihood.

    Returns log p({d_i} | cosmo, survey, pop_params).

    Parameters
    ----------
    cosmo, survey, pop_params
        Inference parameters.
    gw_pe, em_catalog_pe
        GW posterior samples and associated EM catalog for the PE term.
    gw_sel, em_catalog_sel
        GW injection samples and associated EM catalog for the selection term.
    nEvents
        Number of observed GW events (static — triggers recompilation if changed).
    nsamp
        PE samples per event (static).
    Ndraw
        Total number of injections drawn for the selection integral.
    pop_model, universe_model
        String keys selecting the population and redshift-prior models (static).
    sel_batch_size
        If not None, process selection samples in chunks of this size via
        lax.scan.  Required for models that create large per-sample
        intermediates (e.g. GP population models).  Selection array must
        be pre-padded to a multiple of sel_batch_size in make_likelihood.
    """
    log_p_pop         = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniv  = get_redshift_prior(universe_model)
    H0, Om0           = cosmo.H0, cosmo.Om0

    # ------------------------------------------------------------------
    # Redshift prior with -inf → -1e6 guard
    # ------------------------------------------------------------------
    def log_prior_z(z, pix, catalog):
        lp = raw_logPriorUniv(z, pix, cosmo, survey, catalog)
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    # ------------------------------------------------------------------
    # Per-sample log weight shared by selection and PE terms
    # ------------------------------------------------------------------
    def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        z    = z_of_dL(dL, H0, Om0)
        m1   = m1det / (1.0 + z)
        logw = (
            log_p_pop(m1, q, z, chieff, pop_params)
            + log_prior_z(z, pix, catalog)
            - jnp.log(ddL_of_z(z, dL, H0, Om0))
            - jnp.log(prior_wt)
            - 2.0 * jnp.log1p(z)
        )
        return logw

    # ------------------------------------------------------------------
    # Selection term: estimate log μ and its variance for Neff diagnostic
    # ------------------------------------------------------------------
    def _sel_batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        """logsumexp and logsumexp(2x) for one batch of selection samples."""
        ldw = log_weight(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b,
                         em_catalog_sel)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        # --- Unbatched ---
        lse, lse2 = _sel_batch_lse(
            gw_sel.dL, gw_sel.m1det, gw_sel.q,
            gw_sel.chieff, gw_sel.pixels, gw_sel.prior_wt,
        )
        log_mu = lse  - jnp.log(Ndraw)
        log_s2 = lse2 - 2.0 * jnp.log(Ndraw)

    else:
        # --- Batched via lax.scan ---
        # Peak GPU memory: O(sel_batch_size × N_grid) instead of O(N_sel × N_grid)
        N_sel     = gw_sel.dL.shape[0]
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            lse_b, lse2_b = _sel_batch_lse(
                sl(gw_sel.dL), sl(gw_sel.m1det), sl(gw_sel.q),
                sl(gw_sel.chieff), sl(gw_sel.pixels), sl(gw_sel.prior_wt),
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(_scan_fn, None, jnp.arange(N_batches))
        log_mu = logsumexp(lse_all)  - jnp.log(Ndraw)
        log_s2 = logsumexp(lse2_all) - 2.0 * jnp.log(Ndraw)

    # Effective number of selection samples (Farr 2019 diagnostic)
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    Neff       = jnp.exp(2.0 * log_mu - log_sigma2)

    ll  = jnp.where(Neff <= 5 * nEvents, -jnp.inf, 0.0)
    ll += -nEvents * log_mu + nEvents * (3 + nEvents) / (2.0 * Neff)

    # ------------------------------------------------------------------
    # PE term: scan over events to keep peak memory O(nsamp × N_grid)
    # ------------------------------------------------------------------
    def _pe_event_fn(_, event_idx):
        s  = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
        # Pass detector-frame quantities directly; log_weight converts internally
        ldw = log_weight(
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

def _barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Apply lax.optimization_barrier to a single array.

    This tells XLA to treat the array as an opaque runtime value rather
    than a compile-time constant, preventing constant-folding from
    materializing large intermediate tensors in the HLO graph during JIT
    compilation.  The result is numerically identical but the compiler
    cannot propagate values through the barrier.

    Call this on all large catalog arrays (zgals, delta_g_pix_z, etc.)
    BEFORE they are captured in the JIT closure — i.e. here in
    make_likelihood, not inside the returned likelihood function.
    Applying barriers inside the likelihood body is too late: JAX has
    already ingested the raw constant values during tracing.
    """
    return lax.optimization_barrier(arr)


def make_likelihood(opts, data: dict, pop_params_fid,
                    fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    Large catalog arrays are barrier-wrapped here so that JIT
    compilation does not try to constant-fold through them.

    Parameters
    ----------
    opts
        Argument namespace (attributes: pop_model, universe_model,
        fix_cosmology, fix_population, fix_survey, sel_batch_size).
    data
        Dictionary produced by load_all_data.  Must contain:
          nEvents, nsamp, Ndraw, apix, delta_g_pix_z, sigma_kernel,
          m1det, m2det, dL, chieff, p_pe, pixels_pe,
          m1detsels, m2detsels, dLsels, chieffsels, p_draw, pixels_sel,
          (optional) zgals_pe/sel, dzgals_pe/sel, wgals_pe/sel.
    pop_params_fid
        Fiducial population parameters used when --fix_population=True.
    fixed_parameter_values
        Dict of {label: value} for parameters held fixed inside the
        sampled coordinate vector.
    """
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------
    nEvents   = data["nEvents"]
    nsamp     = data["nsamp"]
    Ndraw     = data["Ndraw"]
    apix      = data["apix"]

    pop_model      = opts.pop_model
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)

    # ------------------------------------------------------------------
    # Optional galaxy catalog arrays → JAX; dummy if absent
    # ------------------------------------------------------------------
    def _to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    # Apply optimization barriers HERE, before closure capture.
    # Large arrays: zgals (npix × Ngal_max), delta_g_pix_z (npix × Nz)
    # Small arrays: dzgals, wgals — barriers cost nothing but are harmless
    def _load_catalog(prefix):
        return dict(
            zgals          = _barrier(_to_jax(f"zgals_{prefix}")),
            dzgals         = _barrier(_to_jax(f"dzgals_{prefix}")),
            wgals          = _barrier(_to_jax(f"wgals_{prefix}")),
            ngals          = _barrier(_to_jax(f"ngals_{prefix}")),
        )

    # delta_g_pix_z is shared between PE and selection catalogs
    delta_g_pix_z = _barrier(_to_jax("delta_g_pix_z"))
    sigma_kernel  = data["sigma_kernel"]

    cat_pe  = _load_catalog("pe")
    cat_sel = _load_catalog("sel")

    # ------------------------------------------------------------------
    # Pad selection arrays when batching is requested
    # ------------------------------------------------------------------
    if sel_batch_size is not None:
        N_sel     = data["dLsels"].shape[0]
        remainder = N_sel % sel_batch_size
        if remainder != 0:
            pad = sel_batch_size - remainder

            def _pad1d(arr, fill=0.0):
                return np.concatenate([arr, np.full(pad, fill)])

            data = dict(data)   # shallow copy so we don't mutate the caller's dict
            data["dLsels"]     = _pad1d(data["dLsels"])
            data["m1detsels"]  = _pad1d(data["m1detsels"])
            data["m2detsels"]  = _pad1d(data["m2detsels"])
            data["chieffsels"] = _pad1d(data["chieffsels"])
            data["pixels_sel"] = _pad1d(data["pixels_sel"].astype(np.int32), fill=0)
            data["p_draw"]     = _pad1d(data["p_draw"], fill=1.0)   # → log weight = -inf

            n_batches = N_sel // sel_batch_size + 1
            print(f"    [sel_batch] {N_sel} → {N_sel + pad} samples  "
                  f"({n_batches} × {sel_batch_size})")

    # ------------------------------------------------------------------
    # Parameter space bookkeeping
    # ------------------------------------------------------------------
    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    cosmo_labels        = ["H0", "Om0"]
    survey_labels       = ["log10n0", "z50", "w", "delta", "b_miss", "alpha"]

    # Convert to a plain Python list of floats NOW, at closure-capture time.
    # If pop_params_fid is a JAX array, indexing it inside a JIT-traced
    # function returns an abstract tracer — calling float() on that tracer
    # raises ConcretizationTypeError.  A plain Python list is indexed at
    # Python (trace-time) level and returns concrete floats.
    pop_params_fid_list = [float(v) for v in pop_params_fid]

    sampled_labels = []
    if not opts.fix_cosmology:  sampled_labels += cosmo_labels
    if not opts.fix_population: sampled_labels += pop_labels
    if not opts.fix_survey:     sampled_labels += survey_labels

    # Labels that appear in the coordinate vector but are fixed to a value
    fixed_in_coord = {
        k: float(v) for k, v in fixed_parameter_values.items()
        if k in set(sampled_labels)
    }

    # ------------------------------------------------------------------
    # Inner likelihood callable (called by the sampler per proposal)
    # ------------------------------------------------------------------
    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        coord = jnp.asarray(coord)

        # --- Unpack coordinate vector into a label → value dict ---
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
                f"Too many coordinates: consumed {offset}, got {coord.shape[0]}."
            )

        def _get(label, default):
            if label in values:                  return values[label]
            if label in fixed_parameter_values:  return fixed_parameter_values[label]
            return default

        # --- Cosmology ---
        if opts.fix_cosmology:
            H0, Om0 = _get("H0", H0_FID), _get("Om0", OM0_FID)
        else:
            H0, Om0 = values["H0"], values["Om0"]

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

        gw_pe = GWEvent(
            m1det     = data["m1det"],
            m2det     = data["m2det"],
            dL        = data["dL"],
            chieff    = data["chieff"],
            prior_wt  = data["p_pe"],
            pixels    = data["pixels_pe"],
        )
        gw_sel = GWEvent(
            m1det     = data["m1detsels"],
            m2det     = data["m2detsels"],
            dL        = data["dLsels"],
            chieff    = data["chieffsels"],
            prior_wt  = data["p_draw"],
            pixels    = data["pixels_sel"],
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