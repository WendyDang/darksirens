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
):
    log_p_pop = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniverse = get_redshift_prior(universe_model)

    def logPriorUniverse_safe(z, pix, cosmo, survey, em_catalog):
        lp = raw_logPriorUniverse(z, pix, cosmo, survey, em_catalog)
        return jnp.where(jnp.isfinite(lp), lp, -1e6)

    H0, Om0 = cosmo.H0, cosmo.Om0

    # ==========================================
    # --- Selection term μ ---
    # ==========================================
    zsels = z_of_dL(gw_sel.dL, H0, Om0)
    m1sels = gw_sel.m1det / (1 + zsels)
    # Using the redshift-invariant q directly from our PyTree!
    
    log_det_weights = log_p_pop(m1sels, gw_sel.q, zsels, gw_sel.chieff, pop_params)
    log_det_weights += logPriorUniverse_safe(
        zsels, gw_sel.pixels, cosmo, survey, em_catalog_sel
    )
    log_det_weights += -jnp.log(ddL_of_z(zsels, gw_sel.dL, H0, Om0))
    log_det_weights += -jnp.log(gw_sel.prior_wt) - 2*jnp.log1p(zsels)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2*log_mu - log_sigma2)

    ll = jnp.where((Neff <= 5 * nEvents), -jnp.inf, 0.0)
    ll += -nEvents*log_mu + nEvents*(3+nEvents)/(2*Neff)

    # ==========================================
    # --- Event term ---
    # ==========================================
    z = z_of_dL(gw_pe.dL, H0, Om0)
    m1 = gw_pe.m1det / (1 + z)
    
    log_weights = log_p_pop(m1, gw_pe.q, z, gw_pe.chieff, pop_params)
    log_weights += logPriorUniverse_safe(
        z, gw_pe.pixels, cosmo, survey, em_catalog_pe
    )
    log_weights += -jnp.log(ddL_of_z(z, gw_pe.dL, H0, Om0))
    log_weights += -jnp.log(gw_pe.prior_wt) - 2*jnp.log1p(z)

    log_weights = log_weights.reshape((nEvents, nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights, axis=-1))

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def make_likelihood(opts, data, delta_g_pix_z, pop_params_fid, fixed_parameter_values=None):
    """
    Enhanced wrapper that converts None values into dummy JAX arrays 
    to ensure compatibility with JIT, and dynamically builds PyTrees.
    """
    
    def to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    nEvents, nsamp, Ndraw, apix = data["nEvents"], opts.nsamp, data["Ndraw"], data["apix"]
    pop_model, universe_model = opts.pop_model, opts.universe_model

    # Convert optional survey data to JAX arrays
    zgals_pe = to_jax("zgals_pe")
    dzgals_pe = to_jax("dzgals_pe")
    wgals_pe = to_jax("wgals_pe")
    ngals_pe = to_jax("ngals_pe")
    
    zgals_sel = to_jax("zgals_sel")
    dzgals_sel = to_jax("dzgals_sel")
    wgals_sel = to_jax("wgals_sel")
    ngals_sel = to_jax("ngals_sel")

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
            apix=apix, zgals=zgals_pe, dzgals=dzgals_pe, wgals=wgals_pe, ngals=ngals_pe, delta_g_pix_z=delta_g_pix_z
        )
        em_catalog_sel = EMCatalog(
            apix=apix, zgals=zgals_sel, dzgals=dzgals_sel, wgals=wgals_sel, ngals=ngals_sel, delta_g_pix_z=delta_g_pix_z
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
        )

    return likelihood