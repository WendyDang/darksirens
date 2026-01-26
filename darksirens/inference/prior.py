# prior.py
import numpy as np
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck


def get_fixed_population_params(pop_model):
    """
    Fiducial population parameters used when --fix_population=True.
    You can customize this per pop_model.
    """
    # Example for powerlaw+peak
    return jnp.array([2.0, 1.0, 5.0, 80.0, 0.5, 35.0, 5.0, 0.1])


def build_parameter_space(pop_model, fix_population):
    """
    Construct labels and bounds for cosmology, population, and survey parameters.
    Handles fix_population=True cleanly.
    """

    # --- Cosmology ---
    cosmo_labels = ["H0", "Om0"]
    cosmo_lower = [20.0, Om0Planck - 0.1]
    cosmo_upper = [120.0, Om0Planck + 0.1]

    # --- Population ---
    pop_lower, pop_upper, pop_labels = pop_model_prior_parser(pop_model)
    n_pop = len(pop_labels)

    # --- Survey ---
    survey_labels = ["log10n0", "z50", "w", "delta", "gamma", "b_miss", "alpha"]
    survey_lower = [-10.0, 0.0, 0.01, -10.0, -10.0, 0.0, 0.0]
    survey_upper = [10.0, 5.0, 5.0, 10.0, 10.0, 5.0, 1.0]

    # --- Assemble full parameter vector ---
    if fix_population:
        labels = cosmo_labels + survey_labels
        lower = np.array(cosmo_lower + survey_lower)
        upper = np.array(cosmo_upper + survey_upper)
        n_pop_effective = 0
    else:
        labels = cosmo_labels + pop_labels + survey_labels
        lower = np.array(cosmo_lower + list(pop_lower) + survey_lower)
        upper = np.array(cosmo_upper + list(pop_upper) + survey_upper)
        n_pop_effective = n_pop

    return labels, lower, upper, n_pop_effective, pop_labels, survey_labels, cosmo_labels


def make_prior_transform(lower, upper):
    """
    Returns a function mapping unit cube → parameter space.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def prior_transform(u):
        return u * (upper - lower) + lower

    return prior_transform
