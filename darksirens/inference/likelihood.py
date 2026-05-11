"""
likelihood.py
-------------
Hierarchical dark-siren log-likelihood.

Sentinel convention
-------------------
All log-probability floors are -jnp.inf, not finite magic numbers.
  - log_p_pop: p=0 → -inf  (changed in base.py)
  - log_prior_z: no finite guard; -inf propagates correctly through
    logsumexp and is caught by the final jnp.isfinite(ll) check.
  - Neff: guarded against NaN when log_mu=-inf (all weights -inf).

RAM note
--------
optimization_barrier MUST be applied before arrays enter any JIT closure
(i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from jax.scipy.special import logsumexp
import numpy as np
import warnings

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em import get_redshift_prior
from darksirens.em.completion import build_pixel_kde_cache
from darksirens.inference.utils import log_sample_weight
from darksirens.inference.events import pad_gw_event_to_multiple
from darksirens.utils.utils import logdiffexp
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

from astropy.cosmology import Planck15

H0_FID          = float(Planck15.H0.value)
OM0_FID         = float(Planck15.Om0)
SURVEY_PARAMS_FID = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])


DARK_SIREN_CACHE_MODELS = {"dark_sirens"}


def _unique_inference_pixels(pixels_pe, pixels_sel) -> np.ndarray:
    """Return the sorted union of unique PE and selection HEALPix pixels."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    return np.unique(np.concatenate([unique_pe, unique_sel])).astype(
        np.int32, copy=False
    )


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
    Hierarchical dark-siren log-likelihood.

    Returns log p({d_i} | cosmo, survey, pop_params).
    """
    log_p_pop        = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniv = get_redshift_prior(universe_model)
    H0, Om0          = cosmo.H0, cosmo.Om0

    # No finite guard on the redshift prior.  -inf propagates correctly
    # through logsumexp and is caught by the final isfinite check.
    def log_prior_z(z, pix, catalog):
        return raw_logPriorUniv(z, pix, cosmo, survey, catalog)

    def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        """
        Selection weight in the canonical integration variables.

        The detected-injection ``GWEvent`` stores ``m1det`` and ``m2det`` for
        provenance, but the likelihood integrates over ``(m1det, q, dL)``
        with ``q = m2det / m1det``.  ``prior_wt`` must therefore be a
        proposal density in that same basis.
        """
        return log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog, log_p_pop, log_prior_z,
        )

    def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalog):
        """
        PE weight in the same ``(m1det, q, dL)`` variables as selection.

        This intentionally calls the same helper as ``log_weight``.  If a PE
        file supplies a native ``(m1det, m2det, dL)`` density, it must be
        converted to the ``q`` basis before it reaches the likelihood.
        """
        return log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog, log_p_pop, log_prior_z,
        )

    # ------------------------------------------------------------------
    # Selection term
    # ------------------------------------------------------------------
    def _sel_batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        ldw = log_weight(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        valid = pwt_b > 0.0
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        lse, lse2 = _sel_batch_lse(
            gw_sel.dL, gw_sel.m1det, gw_sel.q,
            gw_sel.chieff, gw_sel.pixels, gw_sel.prior_wt,
        )
        log_mu = lse  - jnp.log(Ndraw)
        log_s2 = lse2 - 2.0 * jnp.log(Ndraw)
    else:
        N_sel = gw_sel.dL.shape[0]
        if N_sel % sel_batch_size != 0:
            raise ValueError(
                "gw_sel length must be divisible by sel_batch_size; "
                "pad with pad_gw_event_to_multiple before calling "
                "darksiren_log_likelihood"
            )
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

    # Effective sample size (Farr 2019).
    # Guard: when all selection weights are -inf, log_mu = log_s2 = -inf.
    # The subtraction 2*(-inf) - (-inf) = nan without the guard.
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    Neff = jnp.where(
        jnp.isfinite(log_mu),
        jnp.exp(2.0 * log_mu - log_sigma2),
        0.0,                              # → too_sparse=True → ll=-inf below
    )

    ll  = jnp.where(Neff <= 5 * nEvents, -jnp.inf, 0.0)
    ll += -nEvents * log_mu + nEvents * (3 + nEvents) / (2.0 * Neff)

    # ------------------------------------------------------------------
    # PE term: scan over events
    # ------------------------------------------------------------------
    def _pe_event_fn(_, event_idx):
        s  = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
        ldw = log_weight_ev(
            sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL),
            sl(gw_pe.chieff), sl(gw_pe.pixels), sl(gw_pe.prior_wt),
            em_catalog_pe,
        )
        ldw = jnp.where(jnp.isfinite(ldw), ldw, -jnp.inf)
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


# ============================================================
# Likelihood closure factory
# ============================================================

def _barrier(arr: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(jnp.asarray(arr))


def make_likelihood(opts, data: dict, pop_params_fid,
                    fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    Applies optimization_barrier to all large catalog and GW data arrays
    before they are captured in the JIT closure.
    """
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    nEvents        = data["nEvents"]
    nsamp          = data["nsamp"]
    Ndraw          = data["Ndraw"]
    apix           = data["apix"]
    pop_model      = opts.pop_model
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)

    def _to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    # Catalog arrays — barrier-wrapped before closure capture.  Prefer the
    # compact unique-pixel PE/selection slices built by load_all_data.  The
    # GWEvent ``pixels`` arrays below are then sample-to-compact-row indices;
    # EMCatalog.unique_pixels preserves the global HEALPix pixel only for the
    # operations that truly need it (currently the LSS overdensity field).
    def _catalog_key(compact_key, full_key):
        return compact_key if data.get(compact_key) is not None else full_key

    pe_uses_compact = data.get("zgals_pe") is not None
    sel_uses_compact = data.get("zgals_sel") is not None

    # Backward compatibility for tests or callers that still provide only full
    # survey arrays: synthesize the same compact views that load_all_data now
    # creates, without reintroducing per-sample duplicated catalog rows.
    def _ensure_compact(prefix, pixels_key):
        if data.get(f"zgals_{prefix}") is not None:
            return True
        full_z = data.get("zgals_catalog") if data.get("zgals_catalog") is not None else data.get("zgals")
        full_dz = data.get("dzgals_catalog") if data.get("dzgals_catalog") is not None else data.get("dzgals")
        full_w = data.get("wgals_catalog") if data.get("wgals_catalog") is not None else data.get("wgals")
        full_n = data.get("ngals_catalog")
        pixels = data.get(pixels_key)
        if any(value is None for value in (full_z, full_dz, full_w, full_n, pixels)):
            return False
        unique_pixels, sample_to_unique_idx = np.unique(
            np.asarray(pixels, dtype=np.int32), return_inverse=True
        )
        unique_pixels = unique_pixels.astype(np.int32, copy=False)
        data[f"unique_pixels_{prefix}"] = unique_pixels
        data[f"sample_to_unique_{prefix}"] = sample_to_unique_idx.astype(np.int32, copy=False)
        data[f"zgals_{prefix}"] = full_z[unique_pixels]
        data[f"dzgals_{prefix}"] = full_dz[unique_pixels]
        data[f"wgals_{prefix}"] = full_w[unique_pixels]
        data["ngals_pe" if prefix == "pe" else "ngals_sel"] = full_n[unique_pixels]
        return True

    pe_uses_compact = pe_uses_compact or _ensure_compact("pe", "pixels_pe")
    sel_uses_compact = sel_uses_compact or _ensure_compact("sel", "pixels_sel")

    zgals_pe_catalog = _barrier(_to_jax(_catalog_key("zgals_pe", "zgals_catalog")))
    dzgals_pe_catalog = _barrier(_to_jax(_catalog_key("dzgals_pe", "dzgals_catalog")))
    wgals_pe_catalog = _barrier(_to_jax(_catalog_key("wgals_pe", "wgals_catalog")))
    ngals_pe_raw = data.get("ngals_pe") if pe_uses_compact else data.get("ngals_catalog")
    ngals_pe_catalog = (
        _barrier(jnp.asarray(ngals_pe_raw, dtype=jnp.int32))
        if ngals_pe_raw is not None else None
    )

    zgals_sel_catalog = _barrier(_to_jax(_catalog_key("zgals_sel", "zgals_catalog")))
    dzgals_sel_catalog = _barrier(_to_jax(_catalog_key("dzgals_sel", "dzgals_catalog")))
    wgals_sel_catalog = _barrier(_to_jax(_catalog_key("wgals_sel", "wgals_catalog")))
    ngals_sel_raw = data.get("ngals_sel") if sel_uses_compact else data.get("ngals_catalog")
    ngals_sel_catalog = (
        _barrier(jnp.asarray(ngals_sel_raw, dtype=jnp.int32))
        if ngals_sel_raw is not None else None
    )

    unique_pixels_pe_raw = data.get("unique_pixels_pe") if pe_uses_compact else None
    unique_pixels_sel_raw = data.get("unique_pixels_sel") if sel_uses_compact else None
    unique_pixels_pe = (
        _barrier(jnp.asarray(unique_pixels_pe_raw, dtype=jnp.int32))
        if unique_pixels_pe_raw is not None else None
    )
    unique_pixels_sel = (
        _barrier(jnp.asarray(unique_pixels_sel_raw, dtype=jnp.int32))
        if unique_pixels_sel_raw is not None else None
    )
    sample_to_unique_pe_raw = (
        data.get("sample_to_unique_pe") if pe_uses_compact else data.get("pixels_pe")
    )
    sample_to_unique_sel_raw = (
        data.get("sample_to_unique_sel") if sel_uses_compact else data.get("pixels_sel")
    )
    sample_to_unique_pe = _barrier(jnp.asarray(sample_to_unique_pe_raw, dtype=jnp.int32))
    sample_to_unique_sel = _barrier(jnp.asarray(sample_to_unique_sel_raw, dtype=jnp.int32))

    # Use a single compact inference catalog for PE and selection so common
    # pixels are stored and cached once.  The separate PE/selection compact
    # arrays remain in ``data`` for diagnostics and callers that inspect them.
    full_z = data.get("zgals_catalog") if data.get("zgals_catalog") is not None else data.get("zgals")
    full_dz = data.get("dzgals_catalog") if data.get("dzgals_catalog") is not None else data.get("dzgals")
    full_w = data.get("wgals_catalog") if data.get("wgals_catalog") is not None else data.get("wgals")
    full_n = data.get("ngals_catalog")
    union_unique_pixels = None
    if all(
        value is not None
        for value in (
            full_z, full_dz, full_w, full_n,
            data.get("pixels_pe"), data.get("pixels_sel"),
        )
    ):
        union_unique_pixels = _unique_inference_pixels(data["pixels_pe"], data["pixels_sel"])
        pe_global_pixels = np.asarray(data["pixels_pe"], dtype=np.int32)
        sel_global_pixels = np.asarray(data["pixels_sel"], dtype=np.int32)
        sample_to_union_pe_raw = np.searchsorted(
            union_unique_pixels, pe_global_pixels
        ).astype(np.int32, copy=False)
        sample_to_union_sel_raw = np.searchsorted(
            union_unique_pixels, sel_global_pixels
        ).astype(np.int32, copy=False)

        zgals_union_catalog = _barrier(jnp.asarray(full_z[union_unique_pixels]))
        dzgals_union_catalog = _barrier(jnp.asarray(full_dz[union_unique_pixels]))
        wgals_union_catalog = _barrier(jnp.asarray(full_w[union_unique_pixels]))
        ngals_union_catalog = _barrier(jnp.asarray(full_n[union_unique_pixels], dtype=jnp.int32))
        unique_pixels_union = _barrier(jnp.asarray(union_unique_pixels, dtype=jnp.int32))
        sample_to_unique_pe = _barrier(jnp.asarray(sample_to_union_pe_raw, dtype=jnp.int32))
        sample_to_unique_sel = _barrier(jnp.asarray(sample_to_union_sel_raw, dtype=jnp.int32))

        zgals_pe_catalog = zgals_sel_catalog = zgals_union_catalog
        dzgals_pe_catalog = dzgals_sel_catalog = dzgals_union_catalog
        wgals_pe_catalog = wgals_sel_catalog = wgals_union_catalog
        ngals_pe_catalog = ngals_sel_catalog = ngals_union_catalog
        unique_pixels_pe = unique_pixels_sel = unique_pixels_union

    delta_g_pix_z = _barrier(_to_jax("delta_g_pix_z"))
    sigma_kernel = data["sigma_kernel"]

    # Unique-pixel KDE cache.  The common path builds one cache from the full
    # survey rows using global HEALPix ids, then compact catalogs translate
    # their row ids back through EMCatalog.unique_pixels for the lookup.
    dN_obs_kde_pe = dN_obs_kde_sel = None
    pixel_to_cache_idx_pe = pixel_to_cache_idx_sel = None
    cache_required = universe_model in DARK_SIREN_CACHE_MODELS

    if cache_required:
        if union_unique_pixels is not None:
            dN_obs_kde_pe, pixel_to_cache_idx_pe = build_pixel_kde_cache(
                unique_pixels=union_unique_pixels,
                zgals=full_z,
                n_pix_catalog=int(data.get("n_pix_catalog", np.asarray(full_z).shape[0])),
                wgals=full_w,
                ngals=full_n,
            )
            dN_obs_kde_sel = dN_obs_kde_pe
            pixel_to_cache_idx_sel = pixel_to_cache_idx_pe
        else:
            missing_cache_inputs = [
                name for name, value in (
                    ("PE compact galaxy redshifts", data.get("zgals_pe")),
                    ("selection compact galaxy redshifts", data.get("zgals_sel")),
                    ("PE sample-to-unique map", data.get("sample_to_unique_pe")),
                    ("selection sample-to-unique map", data.get("sample_to_unique_sel")),
                    (
                        "PE galaxy mask (wgals or ngals)",
                        data.get("wgals_pe")
                        if data.get("wgals_pe") is not None
                        else data.get("ngals_pe"),
                    ),
                    (
                        "selection galaxy mask (wgals or ngals)",
                        data.get("wgals_sel")
                        if data.get("wgals_sel") is not None
                        else data.get("ngals_sel"),
                    ),
                ) if value is None
            ]
            if missing_cache_inputs:
                message = (
                    "Dark-siren inference requires the per-pixel KDE cache; "
                    f"cannot build it because these inputs are missing: {', '.join(missing_cache_inputs)}."
                )
                if getattr(opts, "allow_uncached_dark_sirens", False):
                    warnings.warn(
                        message
                        + " Falling back to uncached completion for tests/backward compatibility.",
                        RuntimeWarning,
                    )
                else:
                    raise RuntimeError(message)
            else:
                n_pe_rows = int(np.asarray(data["zgals_pe"]).shape[0])
                n_sel_rows = int(np.asarray(data["zgals_sel"]).shape[0])
                dN_obs_kde_pe, pixel_to_cache_idx_pe = build_pixel_kde_cache(
                    unique_pixels=np.arange(n_pe_rows, dtype=np.int32),
                    zgals=data["zgals_pe"],
                    n_pix_catalog=n_pe_rows,
                    wgals=data.get("wgals_pe"),
                    ngals=data.get("ngals_pe"),
                )
                dN_obs_kde_sel, pixel_to_cache_idx_sel = build_pixel_kde_cache(
                    unique_pixels=np.arange(n_sel_rows, dtype=np.int32),
                    zgals=data["zgals_sel"],
                    n_pix_catalog=n_sel_rows,
                    wgals=data.get("wgals_sel"),
                    ngals=data.get("ngals_sel"),
                )

    dN_obs_kde_pe = _barrier(dN_obs_kde_pe) if dN_obs_kde_pe is not None else None
    dN_obs_kde_sel = _barrier(dN_obs_kde_sel) if dN_obs_kde_sel is not None else None
    pixel_to_cache_idx_pe = (
        _barrier(jnp.asarray(pixel_to_cache_idx_pe, dtype=jnp.int32))
        if pixel_to_cache_idx_pe is not None else None
    )
    pixel_to_cache_idx_sel = (
        _barrier(jnp.asarray(pixel_to_cache_idx_sel, dtype=jnp.int32))
        if pixel_to_cache_idx_sel is not None else None
    )

    # GW data arrays — barrier-wrapped.
    m1det_pe   = _barrier(_to_jax("m1det"))
    m2det_pe   = _barrier(_to_jax("m2det"))
    dL_pe      = _barrier(_to_jax("dL"))
    chieff_pe  = _barrier(_to_jax("chieff"))
    p_pe       = _barrier(_to_jax("p_pe"))
    pixels_pe  = sample_to_unique_pe
    q_pe       = _barrier(m2det_pe / m1det_pe)

    m1det_sel  = _barrier(_to_jax("m1detsels"))
    m2det_sel  = _barrier(_to_jax("m2detsels"))
    dL_sel     = _barrier(_to_jax("dLsels"))
    chieff_sel = _barrier(_to_jax("chieffsels"))
    p_draw     = _barrier(_to_jax("p_draw"))
    pixels_sel = sample_to_unique_sel
    q_sel      = _barrier(m2det_sel / m1det_sel)

    # Parameter space.
    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    cosmo_labels  = ["H0", "Om0"]
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha_miss"]
    pop_params_fid_list = [float(v) for v in pop_params_fid]

    sampled_labels = []
    if not opts.fix_cosmology:  sampled_labels += cosmo_labels
    if not opts.fix_population: sampled_labels += pop_labels
    if not opts.fix_survey:     sampled_labels += survey_labels

    fixed_in_coord = {
        k: float(v) for k, v in fixed_parameter_values.items()
        if k in set(sampled_labels)
    }

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        coord = jnp.asarray(coord)

        values = {}
        offset = 0
        for label in sampled_labels:
            if label in fixed_in_coord:
                values[label] = fixed_in_coord[label]
                continue
            if offset >= coord.shape[0]:
                raise ValueError(
                    f"Too few coordinates: needed '{label}' at index {offset}, "
                    f"got {coord.shape[0]}."
                )
            values[label] = coord[offset]
            offset += 1

        if offset != coord.shape[0]:
            raise ValueError(
                f"Coordinate mismatch: consumed {offset}, got {coord.shape[0]}."
            )

        def _get(label, default):
            if label in values:                  return values[label]
            if label in fixed_parameter_values:  return fixed_parameter_values[label]
            return default

        if opts.fix_cosmology:
            H0, Om0 = _get("H0", H0_FID), _get("Om0", OM0_FID)
        else:
            H0, Om0 = values["H0"], values["Om0"]

        if opts.fix_population:
            pop_params = jnp.array([
                _get(label, pop_params_fid_list[i])
                for i, label in enumerate(pop_labels)
            ])
        else:
            pop_params = jnp.array([values[label] for label in pop_labels])

        if opts.fix_survey:
            sp = jnp.array([
                _get(label, float(SURVEY_PARAMS_FID[i]))
                for i, label in enumerate(survey_labels)
            ])
        else:
            sp = jnp.array([values[label] for label in survey_labels])

        cosmo  = CosmoParams(H0=H0, Om0=Om0)
        survey = SurveyParams(
            n0=10.0 ** sp[0], z50=sp[1], w=sp[2],
            delta=sp[3], b_miss=sp[4], alpha_miss=sp[5],
        )

        em_catalog_pe = EMCatalog(
            apix=apix, zgals=zgals_pe_catalog, dzgals=dzgals_pe_catalog,
            wgals=wgals_pe_catalog, ngals=ngals_pe_catalog,
            delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel,
            dN_obs_kde=dN_obs_kde_pe, pixel_to_cache_idx=pixel_to_cache_idx_pe,
            unique_pixels=unique_pixels_pe, sample_to_unique_idx=sample_to_unique_pe,
        )
        em_catalog_sel = EMCatalog(
            apix=apix, zgals=zgals_sel_catalog, dzgals=dzgals_sel_catalog,
            wgals=wgals_sel_catalog, ngals=ngals_sel_catalog,
            delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel,
            dN_obs_kde=dN_obs_kde_sel, pixel_to_cache_idx=pixel_to_cache_idx_sel,
            unique_pixels=unique_pixels_sel, sample_to_unique_idx=sample_to_unique_sel,
        )

        gw_pe = GWEvent(
            m1det=m1det_pe, m2det=m2det_pe, dL=dL_pe,
            chieff=chieff_pe, prior_wt=p_pe, pixels=pixels_pe, q=q_pe,
        )
        gw_sel = GWEvent(
            m1det=m1det_sel, m2det=m2det_sel, dL=dL_sel,
            chieff=chieff_sel, prior_wt=p_draw, pixels=pixels_sel, q=q_sel,
        )
        if sel_batch_size is not None:
            gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

        return darksiren_log_likelihood(
            cosmo, survey, pop_params,
            gw_pe,  em_catalog_pe,
            gw_sel, em_catalog_sel,
            nEvents, nsamp, Ndraw,
            pop_model, universe_model,
            sel_batch_size=sel_batch_size,
        )

    return likelihood