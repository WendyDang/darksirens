import numpy as np
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck


def build_parameter_space(pop_model, fix_population, fix_cosmology, fix_survey):
    """
    Construct labels and bounds for cosmology, population, and survey parameters.
    """
    # --- Cosmology ---
    cosmo_labels = ["H0", "Om0"]
    cosmo_lower = [20.0, Om0Planck - 0.1]
    cosmo_upper = [120.0, Om0Planck + 0.1]
    n_cosmo = len(cosmo_labels)

    # --- Population ---
    pop_lower, pop_upper, pop_labels, model_name = pop_model_prior_parser(pop_model)
    n_pop = len(pop_labels)

    # --- Survey ---
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha"]
    survey_lower = [-10.0, 0.0, 0.01, -10.0, 0.0, 0.0]
    survey_upper = [10.0, 5.0, 5.0, 10.0, 5.0, 1.0]
    n_survey = len(survey_labels)

    # --- Assemble full parameter vector ---
    labels = []
    lower = []
    upper = []

    if not fix_cosmology:
        labels += cosmo_labels
        lower += cosmo_lower
        upper += cosmo_upper
        n_cosmo_eff = n_cosmo
    else:
        n_cosmo_eff = 0

    if not fix_population:
        labels += pop_labels
        lower += list(pop_lower)
        upper += list(pop_upper)
        n_pop_eff = n_pop
    else:
        n_pop_eff = 0

    if not fix_survey:
        labels += survey_labels
        lower += survey_lower
        upper += survey_upper
        n_survey_eff = n_survey
    else:
        n_survey_eff = 0

    return (
        labels, np.array(lower), np.array(upper),
        n_pop_eff, pop_labels, survey_labels, cosmo_labels,
        n_cosmo_eff, n_survey_eff, model_name,
    )

def make_prior_transform(lower, upper):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    def prior_transform(u):
        return u * (upper - lower) + lower
    return prior_transform