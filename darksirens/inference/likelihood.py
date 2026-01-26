# darksirens/inference/likelihood.py

import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp

from darksirens.gw.populations import pop_model_parser
from darksirens.em.completeness import universe_model_parser
from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.utils import logdiffexp


# ---------------------------------------------------------------------
# Low-level JAX kernel (unchanged except for cleanup)
# ---------------------------------------------------------------------
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
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    # Unpack parameters
    H0, Om0 = cosmo_params
    log10n0, z50, w, delta, gamma, b_miss, alpha = survey_params
    n0 = 10.0**log10n0

    # --------------------------------------------------------
    # Selection term μ
    # --------------------------------------------------------
    zsels = z_of_dL(dLsels, H0, Om0)
    m1sels = m1detsels / (1 + zsels)
    m2sels = m2detsels / (1 + zsels)
    qsels = m2sels / m1sels

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

    # --------------------------------------------------------
    # Event term
    # --------------------------------------------------------
    z = z_of_dL(dL, H0, Om0)
    m1 = m1det / (1 + z)
    m2 = m2det / (1 + z)
    q = m2 / m1

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


# ---------------------------------------------------------------------
# High-level wrapper for samplers (Option A)
# ---------------------------------------------------------------------
def make_likelihood(opts, data, delta_g_pix_z, pop_params_fid):
    """
    Returns a function likelihood(coord) that:
      - slices the parameter vector
      - applies fix_population logic
      - calls darksiren_log_likelihood
    """

    # Unpack data once
    m1det = data["m1det"]
    m2det = data["m2det"]
    dL = data["dL"]
    p_pe = data["p_pe"]
    pixels_pe = data["pixels_pe"]
    zgals_pe = data["zgals_pe"]
    dzgals_pe = data["dzgals_pe"]
    wgals_pe = data["wgals_pe"]

    m1detsels = data["m1detsels"]
    m2detsels = data["m2detsels"]
    dLsels = data["dLsels"]
    p_draw = data["p_draw"]
    pixels_sel = data["pixels_sel"]
    zgals_sel = data["zgals_sel"]
    dzgals_sel = data["dzgals_sel"]
    wgals_sel = data["wgals_sel"]

    nEvents = data["nEvents"]
    nsamp = opts.nsamp
    Ndraw = data["Ndraw"]
    apix = data["apix"]
    batch = opts.batch

    pop_model = opts.pop_model
    universe_model = opts.universe_model

    # Build the likelihood function used by samplers
    def likelihood(coord):
        coord = jnp.asarray(coord)

        # Cosmology always first two
        H0, Om0 = coord[:2]

        if opts.fix_population:
            pop_params = pop_params_fid
            survey_params = coord[2:]
        else:
            n_pop = len(pop_params_fid)
            pop_params = coord[2:2+n_pop]
            survey_params = coord[2+n_pop:]

        return darksiren_log_likelihood(
            (H0, Om0),
            survey_params,
            pop_params,
            m1det, m2det, dL, p_pe, pixels_pe,
            zgals_pe, dzgals_pe, wgals_pe,
            m1detsels, m2detsels, dLsels, p_draw,
            pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
            nEvents, nsamp, Ndraw, apix, batch,
            pop_model, universe_model,
            delta_g_pix_z
        )

    return likelihood
