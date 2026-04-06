# prior.py
import numpy as np
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck


def get_fixed_population_params(pop_model):
    """
    Fiducial population parameters used when --fix_population=True.
    """

    if pop_model == "powerlaw+peak":
        # [alpha_1, beta, m_min_1, m_max_1, dm_min_1, mu, sigma, f, gamma]
        return jnp.array([
            2.0,   # alpha_1
            1.0,   # beta
            5.0,   # m_min
            80.0,  # m_max
            0.5,   # dm_min
            35.0,  # mu
            5.0,   # sigma
            0.1,   # f
            3.0    # gamma
        ])

    elif pop_model == "brokenpowerlaw+2peaks":
        # [
        #   alpha_1, alpha_2, break_mass,
        #   m_min, dm_min,
        #   m_max, dm_max,
        #   f1, f2,
        #   mu1, sigma1,
        #   mu2, sigma2,
        #   beta,
        #   gamma
        # ]
        return jnp.array([
            2.0,   # alpha_1
            4.0,   # alpha_2
            30.0,  # break_mass
            5.0,   # m_min
            3.0,   # dm_min
            80.0,  # m_max
            10.0,  # dm_max
            0.10,  # f1
            0.05,  # f2
            10.0,  # mu1
            3.0,   # sigma1
            35.0,  # mu2
            5.0,   # sigma2
            1.0,   # beta
            3.0    # gamma
        ])

    elif pop_model == "brokenpowerlaw+3peaks":
        return jnp.array([
            2.0,   # alpha_1
            4.0,   # alpha_2
            30.0,  # break_mass
            5.0,   # m_min
            3.0,   # dm_min
            80.0,  # m_max
            10.0,  # dm_max
            0.10,  # f1
            0.05,  # f2
            0.03,  # f3
            10.0,  # mu1
            3.0,   # sigma1
            35.0,  # mu2
            5.0,   # sigma2
            70.0,  # mu3
            10.0,  # sigma3
            1.0,   # beta
            3.0    # gamma
        ])

    elif pop_model == "twopowerlaws+peak":
        return jnp.array([
            2.0,   # alpha_1
            4.0,   # alpha_2
            5.0,   # m_min
            3.0,   # dm_min
            80.0,  # m_max
            10.0,  # dm_max
            0.20,  # f1
            0.10,  # f2
            0.05,  # f3
            70.0,  # mu3
            10.0,  # sigma3
            1.0,   # beta
            3.0    # gamma
        ])
    
    elif pop_model == "symmetric_powerlaw+peak":
        return jnp.array([
            2.0,   # alpha
            1.0,   # beta
            5.0,   # m_min
            80.0,  # m_max
            0.5,   # dm_min
            35.0,  # mu
            5.0,   # sigma
            0.1,   # f
            3.0    # gamma
        ])

    elif pop_model == "symmetric_brokenpowerlaw+2peaks":
        return jnp.array([
            2.0,   # alpha_1
            4.0,   # alpha_2
            30.0,  # break_mass
            5.0,   # m_min
            3.0,   # dm_min
            80.0,  # m_max
            10.0,  # dm_max
            0.10,  # f1
            0.05,  # f2
            10.0,  # mu1
            3.0,   # sigma1
            35.0,  # mu2
            5.0,   # sigma2
            1.0,   # beta
            3.0    # gamma
        ])

    elif pop_model == "symmetric_brokenpowerlaw+3peaks":
        return jnp.array([
            2.0, 4.0, 30.0,
            5.0, 3.0,
            80.0, 10.0,
            0.10, 0.05, 0.03,
            10.0, 3.0,
            35.0, 5.0,
            70.0, 10.0,
            1.0,
            3.0
        ])

    elif pop_model == "symmetric_twopowerlaws+peak":
        return jnp.array([
            2.0,   # alpha_1
            4.0,   # alpha_2
            5.0,   # m_min
            3.0,   # dm_min
            80.0,  # m_max
            10.0,  # dm_max
            0.20,  # f1
            0.10,  # f2
            0.05,  # f3
            70.0,  # mu3
            10.0,  # sigma3
            1.0,   # beta
            3.0    # gamma
        ])

    else:
        raise ValueError(f"No fixed parameters defined for model '{pop_model}'")


def build_parameter_space(pop_model, fix_population, fix_cosmology, fix_survey):
    """
    Construct labels and bounds for cosmology, population, and survey parameters.
    Handles fix_population, fix_cosmology, and fix_survey cleanly.
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

    # Cosmology block
    if not fix_cosmology:
        labels += cosmo_labels
        lower += cosmo_lower
        upper += cosmo_upper
        n_cosmo_eff = n_cosmo
    else:
        n_cosmo_eff = 0

    # Population block
    if not fix_population:
        labels += pop_labels
        lower += list(pop_lower)
        upper += list(pop_upper)
        n_pop_eff = n_pop
    else:
        n_pop_eff = 0

    # Survey block
    if not fix_survey:
        labels += survey_labels
        lower += survey_lower
        upper += survey_upper
        n_survey_eff = n_survey
    else:
        n_survey_eff = 0

    return (
        labels,
        np.array(lower),
        np.array(upper),
        n_pop_eff,
        pop_labels,
        survey_labels,
        cosmo_labels,
        n_cosmo_eff,
        n_survey_eff,
        model_name,
    )


def make_prior_transform(lower, upper):
    """
    Returns a function mapping unit cube → parameter space.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def prior_transform(u):
        return u * (upper - lower) + lower

    return prior_transform
