import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em.completeness import universe_model_parser
from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.utils import logdiffexp

from astropy.cosmology import Planck15

H0_fid = float(Planck15.H0.value)
Om0_fid = float(Planck15.Om0)

survey_params_fid = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])

@partial(
    jax.jit,
    static_argnames=[
        "nEvents", "Ndraw", "nsamp", "apix",
        "pop_model", "universe_model"
    ],
)
def darksiren_log_likelihood(
    cosmo_params,
    survey_params,
    pop_params,
    m1det, m2det, dL, chieff, p_pe, pixels_pe,
    zgals_pe, dzgals_pe, wgals_pe,
    m1detsels, m2detsels, dLsels, chieffsels, p_draw,
    pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
    nEvents, nsamp, Ndraw, apix,
    pop_model, universe_model,
    delta_g_pix_z
):
    log_p_pop = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniverse = universe_model_parser(universe_model=universe_model)

    def logPriorUniverse_safe(z, pix,
                              H0, Om0, n0, z50, w, delta,
                              apix, zgals, dzgals, wgals,
                              delta_g_pix_z, b_miss, alpha):
        lp = raw_logPriorUniverse(
            z, pix,
            H0, Om0, n0, z50, w, delta,
            apix, zgals, dzgals, wgals,
            delta_g_pix_z, b_miss, alpha
        )
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    H0, Om0 = cosmo_params
    log10n0, z50, w, delta, b_miss, alpha = survey_params
    n0 = 10.0**log10n0

    # --- Selection term μ ---
    zsels = z_of_dL(dLsels, H0, Om0)
    m1sels = m1detsels / (1 + zsels)
    m2sels = m2detsels / (1 + zsels)
    qsels = m2sels / m1sels

    # Pass chieffsels to population evaluation
    log_det_weights = log_p_pop(m1sels, qsels, zsels, chieffsels, pop_params)
    log_det_weights += logPriorUniverse_safe(
        zsels, pixels_sel,
        H0, Om0, n0, z50, w, delta,
        apix, zgals_sel, dzgals_sel, wgals_sel,
        delta_g_pix_z, b_miss, alpha
    )
    log_det_weights += -jnp.log(ddL_of_z(zsels, dLsels, H0, Om0))
    log_det_weights += -jnp.log(p_draw) - 2*jnp.log1p(zsels)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2*log_mu - log_sigma2)

    ll = jnp.where((Neff <= 5 * nEvents), -jnp.inf, 0.0)
    ll += -nEvents*log_mu + nEvents*(3+nEvents)/(2*Neff)

    # --- Event term ---
    z = z_of_dL(dL, H0, Om0)
    m1 = m1det / (1 + z)
    m2 = m2det / (1 + z)
    q = m2 / m1

    # Pass chieff to population evaluation
    log_weights = log_p_pop(m1, q, z, chieff, pop_params)
    log_weights += logPriorUniverse_safe(
        z, pixels_pe,
        H0, Om0, n0, z50, w, delta,
        apix, zgals_pe, dzgals_pe, wgals_pe,
        delta_g_pix_z, b_miss, alpha
    )
    log_weights += -jnp.log(ddL_of_z(z, dL, H0, Om0))
    log_weights += -jnp.log(p_pe) - 2*jnp.log1p(z)

    log_weights = log_weights.reshape((nEvents, nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights, axis=-1))

    return jnp.nan_to_num(ll, nan=-jnp.inf)


def make_likelihood(opts, data, delta_g_pix_z, pop_params_fid):
    """
    Enhanced wrapper that converts None values into dummy JAX arrays 
    to ensure compatibility with JIT.
    """
    
    # Helper to convert None to dummy array for JAX JIT compatibility
    def to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    # Unpack always-required data
    m1det, m2det, dL = data["m1det"], data["m2det"], data["dL"]
    chieff = data["chieff"]
    p_pe, pixels_pe = data["p_pe"], data["pixels_pe"]
    
    m1detsels, m2detsels, dLsels = data["m1detsels"], data["m2detsels"], data["dLsels"]
    chieffsels = data["chieffsels"]
    p_draw, pixels_sel = data["p_draw"], data["pixels_sel"]
    
    nEvents, nsamp, Ndraw, apix = data["nEvents"], opts.nsamp, data["Ndraw"], data["apix"]
    pop_model, universe_model = opts.pop_model, opts.universe_model

    # Convert survey-optional data to JAX arrays (even if empty/dummy)
    zgals_pe = to_jax("zgals_pe")
    dzgals_pe = to_jax("dzgals_pe")
    wgals_pe = to_jax("wgals_pe")
    zgals_sel = to_jax("zgals_sel")
    dzgals_sel = to_jax("dzgals_sel")
    wgals_sel = to_jax("wgals_sel")

    # Get active parameter counts
    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    true_n_pop = len(pop_labels)

    def likelihood(coord):
        coord = jnp.asarray(coord)
        offset = 0

        if opts.fix_cosmology:
            H0, Om0 = H0_fid, Om0_fid
        else:
            H0, Om0 = coord[offset:offset+2]
            offset += 2

        if opts.fix_population:
            pop_params = pop_params_fid
        else:
            pop_params = coord[offset:offset+true_n_pop]
            offset += true_n_pop

        if opts.fix_survey:
            survey_params = survey_params_fid
        else:
            n_survey = len(survey_params_fid)
            survey_params = coord[offset:offset+n_survey]
            offset += n_survey

        return darksiren_log_likelihood(
            (H0, Om0),
            survey_params,
            pop_params,
            m1det, m2det, dL, chieff, p_pe, pixels_pe,
            zgals_pe, dzgals_pe, wgals_pe,
            m1detsels, m2detsels, dLsels, chieffsels, p_draw,
            pixels_sel, zgals_sel, dzgals_sel, wgals_sel,
            nEvents, nsamp, Ndraw, apix,
            pop_model, universe_model,
            delta_g_pix_z
        )

    return likelihood