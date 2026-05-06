"""
selection.py
------------
Hierarchical selection integral for gravitational-wave population inference.

Physical picture
~~~~~~~~~~~~~~~~
The observed GW event rate depends on which signals pass the detection
threshold.  To avoid biasing the population inference, we must correct
for this selection effect via Thrane & Talbot (2019) / Farr (2019):

    log L_sel = -N_obs * log μ  +  N_obs(N_obs + 3) / (2 N_eff)

where μ is the expected number of detections per unit time under the
proposed population model, estimated as a Monte Carlo average over
injection samples:

    μ = (1/N_draw) Σ_{det inj}  p_pop(d_i|λ) / p_draw(d_i)

and N_eff is the effective sample size of the selection integral
(Farr 2019, arXiv:1904.10879, eq. 15):

    N_eff = μ² / Var(μ)   ≈   [Σ w_i]² / Σ w_i²

The coefficient N_obs(N_obs+3)/(2 N_eff) is the leading-order correction
from the uncertainty in μ on the log-likelihood (see the derivation in
the appendix of Talbot & Golomb 2023, arXiv:2209.02209, eq. A9).  This
is *not* the simpler N_obs²/(2 N_eff) term from the basic Farr (2019)
expansion; the extra factor of 3 arises from the next-order term in the
Taylor expansion of log μ around its mean.

Reliability criterion
~~~~~~~~~~~~~~~~~~~~~
Farr (2019) recommends discarding proposals where N_eff < 4 N_obs
(equivalently returning -inf).  We use the slightly more conservative
threshold of 5 N_obs following Vitale et al. (2022).

References
~~~~~~~~~~
- Farr, W.M. (2019). arXiv:1904.10879
- Thrane & Talbot (2019). PASA 36, e010
- Talbot & Golomb (2023). arXiv:2209.02209
- Vitale et al. (2022). arXiv:2007.05579
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.utils.utils import logdiffexp
from darksirens.utils.containers import GWEvent, EMCatalog


# ============================================================
# Core estimators (testable in isolation)
# ============================================================

def _lse_to_log_mu_neff(
    lse: jnp.ndarray,
    lse2: jnp.ndarray,
    Ndraw: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert logsumexp aggregates to (log_mu, N_eff).

    Parameters
    ----------
    lse  : logsumexp(log_weights)        — sum of weights in log space
    lse2 : logsumexp(2 * log_weights)    — sum of squared weights in log space
    Ndraw : total number of generated injections

    Returns
    -------
    log_mu : log of the selection integral estimate μ
    Neff   : effective sample size (scalar)
    """
    log_Ndraw  = jnp.log(Ndraw)
    log_mu     = lse  - log_Ndraw
    log_s2     = lse2 - 2.0 * log_Ndraw

    # Var(μ) = (Σ w_i²)/N_draw² − μ²/N_draw
    # In log space: logdiffexp(log_s2, 2*log_mu - log_Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - log_Ndraw)
    Neff       = jnp.exp(2.0 * log_mu - log_sigma2)

    return log_mu, Neff


def selection_log_correction(
    log_mu: jnp.ndarray,
    Neff: jnp.ndarray,
    nEvents: int,
) -> jnp.ndarray:
    """
    Log selection correction term (Farr 2019 / Talbot & Golomb 2023).

    Returns ``-inf`` when N_eff < 5 * N_obs (Vitale et al. 2022 criterion),
    indicating the injection set is too sparse for a reliable estimate.

    The correction is:

        -N_obs * log μ  +  N_obs * (N_obs + 3) / (2 * N_eff)

    The first term is the standard Poisson selection factor.  The second
    is the leading uncertainty correction from Taylor-expanding log μ.

    Parameters
    ----------
    log_mu : log of the selection integral estimate
    Neff   : effective sample size of the selection integral
    nEvents : number of observed GW events

    Returns
    -------
    Scalar log-likelihood contribution from the selection term.
    """
    too_sparse = Neff <= 5 * nEvents
    correction = (
        -nEvents * log_mu
        + nEvents * (3 + nEvents) / (2.0 * Neff)
    )
    return jnp.where(too_sparse, -jnp.inf, correction)


# ============================================================
# Full selection term (batched or unbatched)
# ============================================================

def compute_selection_term(
    gw_sel: GWEvent,
    em_catalog_sel: EMCatalog,
    log_weight_fn,
    Ndraw: float,
    nEvents: int,
    sel_batch_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimate log μ and N_eff from the injection set.

    Parameters
    ----------
    gw_sel : GWEvent
        Injection samples (detected).  If ``sel_batch_size`` is set,
        the length must already be padded to a multiple of that value
        (see ``events.pad_gw_event_to_multiple``).
    em_catalog_sel : EMCatalog
        EM catalog sliced to the injection sky positions.
    log_weight_fn : callable(m1det, q, dL, chieff, pix, prior_wt, catalog) → array
        Per-sample log importance weight.  Must broadcast over the batch
        dimension.  Typically a closure from ``likelihood.py`` that captures
        cosmo, survey, pop_params, and the finite-guard for log_prior_z.
    Ndraw : float
        Total number of generated injections (detected + missed).
    nEvents : int
        Number of observed GW events (for the N_eff reliability check).
    sel_batch_size : int or None
        If not None, process injections in chunks via ``lax.scan`` to
        limit peak GPU memory.  The injection array must be pre-padded.

    Returns
    -------
    log_mu : scalar — log of the selection integral estimate
    Neff   : scalar — effective sample size
    """
    def _batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        # --- Unbatched: process all injections at once ---
        lse, lse2 = _batch_lse(
            gw_sel.dL,
            gw_sel.m1det,
            gw_sel.q,
            gw_sel.chieff,
            gw_sel.pixels,
            gw_sel.prior_wt,
        )
    else:
        # --- Batched via lax.scan ---
        # Peak GPU memory: O(sel_batch_size × N_grid) instead of O(N_sel × N_grid).
        # Requires N_sel % sel_batch_size == 0 (pad beforehand).
        N_sel     = gw_sel.dL.shape[0]
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            lse_b, lse2_b = _batch_lse(
                sl(gw_sel.dL),
                sl(gw_sel.m1det),
                sl(gw_sel.q),
                sl(gw_sel.chieff),
                sl(gw_sel.pixels),
                sl(gw_sel.prior_wt),
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(
            _scan_fn, None, jnp.arange(N_batches)
        )
        # Combine per-batch logsumexp values: logsumexp is additive
        # across disjoint index sets.
        lse  = logsumexp(lse_all)
        lse2 = logsumexp(lse2_all)

    return _lse_to_log_mu_neff(lse, lse2, Ndraw)
