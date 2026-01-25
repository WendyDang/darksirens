import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond
from functools import partial

import astropy
import numpy as np
import healpy as hp

import h5py
import astropy.units as u

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from jax.scipy.stats import norm
from darksirens.gw.populations import pop_model_parser
from darksirens.em.completeness import universe_model_parser
from tqdm import tqdm

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *
from darksirens.utils.utils import *


@partial(
    jax.jit,
    static_argnames=[
        "nEvents", "Ndraw", "nsamp", "apix",
        "batch", "pop_model", "universe_model"
    ],
)
def darksiren_log_likelihood(
    cosmo_params,
    survey_params,
    pop_params,
    m1det, m2det, dL, p_pe, pixels_pe,
    zgals_pe, dzgals_pe, wgals_pe,
    m1detsels, m2detsels, dLsels, p_draw,
    pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
    nEvents, nsamp, Ndraw, apix, batch,
    pop_model, universe_model,
    delta_g_pix_z
):

    log_p_pop = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniverse = universe_model_parser(universe_model=universe_model)

    def logPriorUniverse_safe(z, pix,
                              H0, Om0, n0, z50, w, delta, gamma,
                              apix, zgals, dzgals, wgals,
                              delta_g_pix_z, b_miss, alpha):
        lp = raw_logPriorUniverse(
            z, pix,
            H0, Om0, n0, z50, w, delta, gamma,
            apix, zgals, dzgals, wgals,
            delta_g_pix_z, b_miss, alpha
        )
        # clamp -inf / nan to a large negative finite value
        lp = jnp.where(jnp.isfinite(lp), lp, -1e6)
        return lp

    # unpack parameters
    H0, Om0 = cosmo_params
    log10n0, z50, w, delta, gamma, b_miss, alpha = survey_params
    n0 = 10**log10n0

    # -----------------------------
    # Selection term μ
    # -----------------------------
    zsels = z_of_dL(dLsels, H0, Om0)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)
    qsels = m2sels/m1sels

    log_det_weights = log_p_pop(m1sels, qsels, *pop_params)
    log_det_weights += logPriorUniverse_safe(
        zsels, pixels_sel,
        H0, Om0, n0, z50, w, delta, gamma,
        apix, zgals_sel, dzgals_sel, wgals_sel,
        delta_g_pix_z, b_miss, alpha
    )
    log_det_weights += -jnp.log(ddL_of_z(zsels, dLsels, H0, Om0))
    log_det_weights += -jnp.log(p_draw) - 2*jnp.log1p(zsels)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2*log_mu - log_sigma2)

    ll = jnp.where((Neff <= 5 * nEvents), -jnp.inf, 0)
    ll += -nEvents*log_mu + nEvents*(3+nEvents)/(2*Neff)

    # -----------------------------
    # Event term
    # -----------------------------
    z = z_of_dL(dL, H0, Om0)
    m1 = m1det/(1+z)
    m2 = m2det/(1+z)
    q = m2/m1

    log_weights = log_p_pop(m1, q, *pop_params)
    log_weights += logPriorUniverse_safe(
        z, pixels_pe,
        H0, Om0, n0, z50, w, delta, gamma,
        apix, zgals_pe, dzgals_pe, wgals_pe,
        delta_g_pix_z, b_miss, alpha
    )
    log_weights += -jnp.log(ddL_of_z(z, dL, H0, Om0))
    log_weights += -jnp.log(p_pe) - 2*jnp.log1p(z)

    log_weights = log_weights.reshape((nEvents, nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights, axis=-1))

    return jnp.nan_to_num(ll, nan=-jnp.inf)