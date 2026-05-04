import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp
from jax import lax

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em import get_redshift_prior
from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.utils import logdiffexp
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

from astropy.cosmology import Planck15

H0_fid = float(Planck15.H0.value)
Om0_fid = float(Planck15.Om0)

survey_params_fid = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])

@partial(
    jax.jit,
    static_argnames=[
        "nEvents",
        "nsamp",
        "pop_model",
        "universe_model",
        "sel_batch_size",   # None = process all selection samples at once (default)
                            # int  = process in chunks; sel samples must be pre-padded
                            #        to a multiple of this value in make_likelihood
    ],
)
def darksiren_log_likelihood(
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params,
    gw_pe: GWEvent, em_catalog_pe: EMCatalog,
    gw_sel: GWEvent, em_catalog_sel: EMCatalog,
    nEvents, nsamp, Ndraw,
    pop_model, universe_model,
    sel_batch_size=None,
):
    """
    Hierarchical dark-siren log-likelihood.

    Memory management
    -----------------
    For models with expensive per-sample computations (e.g. correlated GP
    population models), processing all 695k selection samples simultaneously
    creates O(N_sel × N_grid) intermediate tensors that exhaust GPU memory.

    Set ``sel_batch_size`` to a value like 10_000–50_000 to process the
    selection term in chunks via ``lax.scan``.  The logsumexp identity

        logsumexp(x_1,...,x_N) = logsumexp(logsumexp(batch_1),...,logsumexp(batch_B))

    ensures the result is numerically identical to the unbatched case.

    The PE term always scans over events (``nsamp`` samples per event),
    keeping peak memory at O(nsamp × N_grid) regardless of nEvents.

    ``sel_batch_size`` is a static argument so changing it triggers
    recompilation.  Pre-pad selection samples in ``make_likelihood`` to
    ensure N_sel is divisible by ``sel_batch_size``.
    """
    log_p_pop = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniverse = get_redshift_prior(universe_model)

    def logPriorUniverse_safe(z, pix, cosmo, survey, em_catalog):
        lp = raw_logPriorUniverse(z, pix, cosmo, survey, em_catalog)
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    H0, Om0 = cosmo.H0, cosmo.Om0

    # ------------------------------------------------------------------
    # Helper: compute per-sample log weights for a selection sub-batch
    # ------------------------------------------------------------------
    def _sel_log_weights(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        z_b  = z_of_dL(dL_b, H0, Om0)
        m1_b = m1det_b / (1.0 + z_b)
        ldw  = log_p_pop(m1_b, q_b, z_b, chi_b, pop_params)
        ldw += logPriorUniverse_safe(z_b, pix_b, cosmo, survey, em_catalog_sel)
        ldw += -jnp.log(ddL_of_z(z_b, dL_b, H0, Om0))
        ldw += -jnp.log(pwt_b) - 2.0 * jnp.log1p(z_b)
        return ldw

    # ==========================================
    # --- Selection term μ ---
    # ==========================================
    if sel_batch_size is None:
        # Unbatched path — original behaviour, fine for standard models
        ldw = _sel_log_weights(
            gw_sel.dL, gw_sel.m1det, gw_sel.q,
            gw_sel.chieff, gw_sel.pixels, gw_sel.prior_wt,
        )
        log_mu = logsumexp(ldw)             - jnp.log(Ndraw)
        log_s2 = logsumexp(2.0 * ldw)      - 2.0 * jnp.log(Ndraw)

    else:
        # Batched path — process selection samples in chunks of sel_batch_size.
        #
        # lax.scan sequentially maps _sel_batch_fn over batch indices,
        # accumulating per-batch logsumexp contributions.  Peak memory is
        # O(sel_batch_size × N_grid) instead of O(N_sel × N_grid).
        #
        # Correctness: logsumexp(lse_per_batch) = logsumexp(all_ldw) because
        #   exp(logsumexp(lse_i)) = sum_i sum_{j in batch_i} exp(ldw_j)
        #                         = sum_all exp(ldw)
        N_sel    = gw_sel.dL.shape[0]           # static at trace time
        N_batches = N_sel // sel_batch_size       # static

        def _sel_batch_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            ldw = _sel_log_weights(
                sl(gw_sel.dL), sl(gw_sel.m1det), sl(gw_sel.q),
                sl(gw_sel.chieff), sl(gw_sel.pixels), sl(gw_sel.prior_wt),
            )
            return None, (logsumexp(ldw), logsumexp(2.0 * ldw))

        _, (lse, lse2) = lax.scan(_sel_batch_fn, None, jnp.arange(N_batches))
        log_mu = logsumexp(lse)  - jnp.log(Ndraw)
        log_s2 = logsumexp(lse2) - 2.0 * jnp.log(Ndraw)

    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0 * log_mu - log_sigma2)

    ll = jnp.where((Neff <= 5 * nEvents), -jnp.inf, 0.0)
    ll += -nEvents * log_mu + nEvents * (3 + nEvents) / (2.0 * Neff)

    # ==========================================
    # --- Event term (always scans over events)
    # ==========================================
    # Scanning over events keeps peak memory at O(nsamp × N_grid) regardless
    # of nEvents, without any change to the result.
    z_all  = z_of_dL(gw_pe.dL, H0, Om0)
    m1_all = gw_pe.m1det / (1.0 + z_all)

    def _pe_event_fn(_, event_idx):
        start = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, nsamp)
        m1_e  = sl(m1_all);       q_e   = sl(gw_pe.q)
        z_e   = sl(z_all);        chi_e = sl(gw_pe.chieff)
        pix_e = sl(gw_pe.pixels); dL_e  = sl(gw_pe.dL)
        pwt_e = sl(gw_pe.prior_wt)

        ldw  = log_p_pop(m1_e, q_e, z_e, chi_e, pop_params)
        ldw += logPriorUniverse_safe(z_e, pix_e, cosmo, survey, em_catalog_pe)
        ldw += -jnp.log(ddL_of_z(z_e, dL_e, H0, Om0))
        ldw += -jnp.log(pwt_e) - 2.0 * jnp.log1p(z_e)
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def make_likelihood(opts, data, pop_params_fid, fixed_parameter_values=None):
    """
    Enhanced wrapper that converts None values into dummy JAX arrays 
    to ensure compatibility with JIT, and dynamically builds PyTrees.
    """
    
    def to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    nEvents, nsamp, Ndraw = data["nEvents"], data["nsamp"], data["Ndraw"]
    apix, delta_g_pix_z, sigma_kernel = data["apix"], data["delta_g_pix_z"], data["sigma_kernel"]
    pop_model, universe_model = opts.pop_model, opts.universe_model

    # sel_batch_size controls chunked processing of selection samples.
    # None   → unbatched (default; suitable for standard population models)
    # int    → process selection samples in chunks of this size
    #          (required for correlated/GP models that create large intermediates)
    #
    # The selection arrays must be padded to an exact multiple of sel_batch_size
    # so lax.scan can slice fixed-size chunks without remainder handling.
    # Padding samples receive weight -inf and do not affect logsumexp results.
    sel_batch_size = getattr(opts, "sel_batch_size", None)

    if sel_batch_size is not None:
        import numpy as np
        N_sel = data["dLsels"].shape[0]
        remainder = N_sel % sel_batch_size
        if remainder != 0:
            pad = sel_batch_size - remainder
            def _pad(arr, fill=0.0):
                if arr.ndim == 1:
                    return np.concatenate([arr, np.full(pad, fill)])
                return np.concatenate([arr, np.full((pad,) + arr.shape[1:], fill)])
            # Pad selection arrays; p_draw padded to 1.0 → log weight = -inf
            data = dict(data)
            data["dLsels"]    = _pad(data["dLsels"])
            data["m1detsels"] = _pad(data["m1detsels"])
            data["m2detsels"] = _pad(data["m2detsels"])
            data["chieffsels"]= _pad(data["chieffsels"])
            data["pixels_sel"]= _pad(data["pixels_sel"].astype(np.int32), fill=0)
            data["p_draw"]    = _pad(data["p_draw"], fill=1.0)
            print(f"    [sel_batch] Padded {N_sel} → {N_sel+pad} selection samples "
                  f"({N_sel//sel_batch_size + 1} batches of {sel_batch_size})")
            Ndraw = data["Ndraw"]  # unchanged — padding doesn't affect Ndraw

     # Convert optional survey data to JAX arrays
    zgals_pe = to_jax("zgals_pe")
    dzgals_pe = to_jax("dzgals_pe")
    wgals_pe = to_jax("wgals_pe")
    ngals_pe = to_jax("ngals_pe")

    zgals_sel = to_jax("zgals_sel")
    dzgals_sel = to_jax("dzgals_sel")
    wgals_sel = to_jax("wgals_sel")
    ngals_sel = to_jax("ngals_sel")

    delta_g_pix_z = to_jax("delta_g_pix_z")

    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    cosmo_labels = ["H0", "Om0"]
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha"]

    expected_labels = []
    if not opts.fix_cosmology: expected_labels.extend(cosmo_labels)
    if not opts.fix_population: expected_labels.extend(pop_labels)
    if not opts.fix_survey: expected_labels.extend(survey_labels)

    expected_set = set(expected_labels)
    fixed_inside_sampled = {k: float(v) for k, v in fixed_parameter_values.items() if k in expected_set}

    def likelihood(coord):
        coord = jnp.asarray(coord)
        sampled_values = {}
        offset = 0

        for label in expected_labels:
            if label in fixed_inside_sampled:
                sampled_values[label] = fixed_inside_sampled[label]
                continue
            if offset >= coord.shape[0]:
                raise ValueError(f"Likelihood received too few coordinates: expected {offset + 1}, got {coord.shape[0]}.")
            sampled_values[label] = coord[offset]
            offset += 1

        if offset != coord.shape[0]:
            raise ValueError(f"Likelihood received too many coordinates: consumed {offset}, got {coord.shape[0]}.")

        def _resolved_value(label, default):
            if label in sampled_values: return sampled_values[label]
            if label in fixed_parameter_values: return fixed_parameter_values[label]
            return default

        if opts.fix_cosmology:
            H0 = _resolved_value("H0", H0_fid)
            Om0 = _resolved_value("Om0", Om0_fid)
        else:
            H0, Om0 = sampled_values["H0"], sampled_values["Om0"]

        if opts.fix_population:
            pop_params = jnp.asarray([_resolved_value(label, pop_params_fid[i]) for i, label in enumerate(pop_labels)])
        else:
            pop_params = jnp.asarray([_resolved_value(label, sampled_values[label]) for label in pop_labels])

        if opts.fix_survey:
            survey_params = jnp.asarray([_resolved_value(label, survey_params_fid[i]) for i, label in enumerate(survey_labels)])
        else:
            survey_params = jnp.asarray([_resolved_value(label, sampled_values[label]) for label in survey_labels])

        # ==========================================
        # Instantiate PyTrees / Dataclasses
        # ==========================================
        cosmo = CosmoParams(H0=H0, Om0=Om0)
        
        survey = SurveyParams(
            n0=10.0**survey_params[0], z50=survey_params[1], w=survey_params[2],
            delta=survey_params[3], b_miss=survey_params[4], alpha=survey_params[5]
        )
        
        em_catalog_pe = EMCatalog(
            apix=apix, zgals=zgals_pe, dzgals=dzgals_pe, wgals=wgals_pe, ngals=ngals_pe, delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel
        )
        em_catalog_sel = EMCatalog(
            apix=apix, zgals=zgals_sel, dzgals=dzgals_sel, wgals=wgals_sel, ngals=ngals_sel, delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel
        )

        gw_pe = GWEvent(
            m1det=data["m1det"],
            m2det=data["m2det"],
            dL=data["dL"],
            chieff=data["chieff"],
            prior_wt=data["p_pe"],
            pixels=data["pixels_pe"]
        )

        gw_sel = GWEvent(
            m1det=data["m1detsels"],
            m2det=data["m2detsels"],
            dL=data["dLsels"],
            chieff=data["chieffsels"],
            prior_wt=data["p_draw"],
            pixels=data["pixels_sel"]
        )

        # Surgical optimization barriers on the two large per-pixel arrays.
        #
        # Root cause of the slow constant-folding warning:
        #   zgals and delta_g_pix_z are captured as compile-time constants
        #   in this JIT closure.  XLA sees them as constants and tries to
        #   evaluate per-pixel operations (KDE mask, overdensity lookup)
        #   at trace time, producing the massive s64[695927,2] constant
        #   tensor visible in the XLA warning.
        #
        # Fix: place optimization_barrier only on these two fields so XLA
        #   treats them as opaque dynamic values during tracing.  All other
        #   fields (apix, wgals, dzgals, GW arrays) are small and benefit
        #   from XLA fusion — leave them unblocked.
        #
        # Why not barrier the whole container (old behaviour):
        #   tree_map(optimization_barrier, container) blocks XLA from fusing
        #   ops across ALL leaves, including the GW mass/distance arrays
        #   where fusion gives real runtime gains.  The blanket approach was
        #   correct in motivation but too broad in scope.
        em_catalog_pe = em_catalog_pe._replace(
            zgals=lax.optimization_barrier(em_catalog_pe.zgals),
            dzgals=lax.optimization_barrier(em_catalog_pe.dzgals),
            wgals=lax.optimization_barrier(em_catalog_pe.wgals),
            ngals=lax.optimization_barrier(em_catalog_pe.ngals),
            delta_g_pix_z=lax.optimization_barrier(em_catalog_pe.delta_g_pix_z),
        )
        em_catalog_sel = em_catalog_sel._replace(
            zgals=lax.optimization_barrier(em_catalog_sel.zgals),
            dzgals=lax.optimization_barrier(em_catalog_sel.dzgals),
            wgals=lax.optimization_barrier(em_catalog_sel.wgals),
            ngals=lax.optimization_barrier(em_catalog_sel.ngals),
            delta_g_pix_z=lax.optimization_barrier(em_catalog_sel.delta_g_pix_z),
        )

        return darksiren_log_likelihood(
            cosmo,
            survey,
            pop_params,
            gw_pe, em_catalog_pe,
            gw_sel, em_catalog_sel,
            nEvents, nsamp, Ndraw,
            pop_model, universe_model,
            sel_batch_size=sel_batch_size,
        )

    return likelihood