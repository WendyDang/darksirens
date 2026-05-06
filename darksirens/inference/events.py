"""
events.py
---------
Factory for constructing GWEvent containers with proper JAX barrier
wrapping and pre-computed mass ratio.

Why barriers on GW data?
~~~~~~~~~~~~~~~~~~~~~~~~
``lax.optimization_barrier`` tells XLA that an array is an opaque
runtime value, preventing constant-folding from materialising large
intermediate tensors in the HLO graph at compile time.

Without barriers, JAX sees the captured data arrays as compile-time
constants during JIT tracing and may attempt to evaluate operations
on them at compile time.  For O(200k) sample arrays this produces
enormous HLO graphs, slow compilation, and in the worst case an OOM
during the compile step — not the run step — which is notoriously
hard to diagnose.

The *correct* place to apply barriers is here, before the arrays are
captured in any JIT closure.  Applying them inside the likelihood body
is too late: JAX has already ingested the raw values during tracing.

Why q here?
~~~~~~~~~~~
``GWEvent.q = m2det / m1det`` is a derived quantity used in every
likelihood evaluation.  As a NamedTuple property it would be
recomputed on every access; stored explicitly it is computed once,
barrier-wrapped alongside the raw arrays, and reused cheaply.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax

from darksirens.utils.containers import GWEvent


def _barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Wrap a single array with ``lax.optimization_barrier``.

    This is the single canonical definition.  It was previously
    duplicated in ``likelihood.py``; it now lives only here.
    """
    return lax.optimization_barrier(jnp.asarray(arr))


def make_gw_event(
    m1det,
    m2det,
    dL,
    chieff,
    prior_wt,
    pixels,
) -> GWEvent:
    """
    Construct a ``GWEvent`` with barrier-wrapped arrays and pre-computed ``q``.

    Parameters
    ----------
    m1det, m2det : array-like
        Detector-frame masses [M_sun].
    dL : array-like
        Luminosity distance [Mpc].
    chieff : array-like
        Effective inspiral spin.
    prior_wt : array-like
        PE prior weights (normalised per event).
    pixels : array-like of int
        HEALPix pixel indices.

    Returns
    -------
    GWEvent
        All floating-point fields are barrier-wrapped.  ``q`` is computed
        from the barrier-wrapped ``m2det / m1det`` so XLA cannot trace
        back through the division to the raw constants.
    """
    m1det_b   = _barrier(jnp.asarray(m1det,    dtype=jnp.float64))
    m2det_b   = _barrier(jnp.asarray(m2det,    dtype=jnp.float64))
    dL_b      = _barrier(jnp.asarray(dL,       dtype=jnp.float64))
    chieff_b  = _barrier(jnp.asarray(chieff,   dtype=jnp.float64))
    prior_wt_b = _barrier(jnp.asarray(prior_wt, dtype=jnp.float64))
    # Integer pixels: barrier still prevents constant-folding of the
    # ang2pix indexing chain, which can be large.
    pixels_b  = _barrier(jnp.asarray(pixels,   dtype=jnp.int32))
    q_b       = _barrier(m2det_b / m1det_b)

    return GWEvent(
        m1det    = m1det_b,
        m2det    = m2det_b,
        dL       = dL_b,
        chieff   = chieff_b,
        prior_wt = prior_wt_b,
        pixels   = pixels_b,
        q        = q_b,
    )


def pad_gw_event_to_multiple(event: GWEvent, multiple: int, fill_prior_wt: float = 1.0) -> GWEvent:
    """
    Pad a ``GWEvent`` so that its length is a multiple of ``multiple``.

    Used when ``sel_batch_size`` is set: ``lax.scan`` requires the array
    length to divide evenly into batches.  Padding with ``prior_wt = 1.0``
    makes the log weight for padded entries equal to ``log_p_pop - 0``,
    which can be large, so callers should also zero-out the contribution
    from padded entries by design (e.g. the padded injections simply
    inflate ``Ndraw`` if not handled).

    Safer approach: pad ``dL`` to a sentinel that maps to a very high
    redshift so ``log_p_pop → -inf`` for padded entries.

    Parameters
    ----------
    event : GWEvent
    multiple : int
    fill_prior_wt : float
        Fill value for ``prior_wt`` on padded entries.  Default 1.0.

    Returns
    -------
    GWEvent with length rounded up to the nearest ``multiple``.
    int — number of padding entries added.
    """
    N = event.dL.shape[0]
    remainder = N % multiple
    if remainder == 0:
        return event, 0

    pad = multiple - remainder

    def _pad1d(arr, fill=0.0):
        return jnp.concatenate([arr, jnp.full(pad, fill, dtype=arr.dtype)])

    # dL=1e5 Mpc → z≈20 → log_p_pop → -inf for any sensible mass model
    padded = make_gw_event(
        m1det    = _pad1d(event.m1det,    fill=30.0),
        m2det    = _pad1d(event.m2det,    fill=30.0),
        dL       = _pad1d(event.dL,       fill=1e5),
        chieff   = _pad1d(event.chieff,   fill=0.0),
        prior_wt = _pad1d(event.prior_wt, fill=fill_prior_wt),
        pixels   = _pad1d(event.pixels.astype(np.int32), fill=0),
    )
    return padded, pad
