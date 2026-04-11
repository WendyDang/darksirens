# prior.py

import numpy as np
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck

def get_fixed_population_params(pop_model):
    """
    Fiducial population parameters used when --fix_population=True.
    These must match the exact dimension and component ordering of the backend parser.
    """

    if pop_model in ["powerlaw+peak", "symmetric_powerlaw+peak"]:
        return jnp.array([
            0.1,   # f1
            2.0,   # alpha
            5.0,   # m_min
            80.0,  # m_max
            0.5,   # dm_min
            10.0,  # dm_max
            1.0,   # beta_PL
            35.0,  # mu
            5.0,   # sigma
            1.0,   # beta_G
            3.0    # gamma
        ])

    elif pop_model in ["brokenpowerlaw+2peaks", "symmetric_brokenpowerlaw+2peaks"]:
        return jnp.array([
            0.10, 0.05,       # f1, f2
            2.0, 4.0, 30.0,   # alpha_1, alpha_2, break_mass
            5.0, 3.0,         # m_min, dm_min
            80.0, 10.0,       # m_max, dm_max
            1.0,              # beta_BPL
            10.0, 3.0, 1.0,   # mu1, sigma1, beta_G1
            35.0, 5.0, 1.0,   # mu2, sigma2, beta_G2
            3.0               # gamma
        ])

    elif pop_model in ["brokenpowerlaw+3peaks", "symmetric_brokenpowerlaw+3peaks"]:
        return jnp.array([
            0.10, 0.05, 0.03, # f1, f2, f3
            2.0, 4.0, 30.0,   # alpha_1, alpha_2, break_mass
            5.0, 3.0,         # m_min, dm_min
            80.0, 10.0,       # m_max, dm_max
            1.0,              # beta_BPL
            10.0, 3.0, 1.0,   # mu1, sigma1, beta_G1
            35.0, 5.0, 1.0,   # mu2, sigma2, beta_G2
            70.0, 10.0, 1.0,  # mu3, sigma3, beta_G3
            3.0               # gamma
        ])

    elif pop_model in ["twopowerlaws+peak", "symmetric_twopowerlaws+peak"]:
        return jnp.array([
            0.20, 0.10,       # f1, f2
            2.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_1, m_min_1, m_max_1, dm_min_1, dm_max_1, beta_PL1
            4.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_2, m_min_2, m_max_2, dm_min_2, dm_max_2, beta_PL2
            35.0, 5.0, 1.0,   # mu, sigma, beta_G
            3.0               # gamma
        ])

    elif pop_model in ["twopowerlaws+2peaks", "symmetric_twopowerlaws+2peaks"]:
        # Total parameters: 22
        return jnp.array([
            0.15, 0.10, 0.05,               # f1, f2, f3 (Component weights)
            2.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_1, m_min_1, m_max_1, dm_min_1, dm_max_1, beta_PL1
            4.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_2, m_min_2, m_max_2, dm_min_2, dm_max_2, beta_PL2
            10.0, 3.0, 1.0,                 # mu1, sigma1, beta_G1 (Peak 1)
            35.0, 5.0, 1.0,                 # mu2, sigma2, beta_G2 (Peak 2)
            3.0                             # gamma (Global redshift evolution)
        ])

    elif pop_model in ["twopowerlaws+3peaks", "symmetric_twopowerlaws+3peaks"]:
        # Total parameters: 26
        return jnp.array([
            0.15, 0.10, 0.05, 0.03,         # f1, f2, f3, f4 (Component weights)
            2.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_1, m_min_1, m_max_1, dm_min_1, dm_max_1, beta_PL1
            4.0, 5.0, 80.0, 3.0, 10.0, 1.0, # alpha_2, m_min_2, m_max_2, dm_min_2, dm_max_2, beta_PL2
            10.0, 3.0, 1.0,                 # mu1, sigma1, beta_G1 (Peak 1)
            35.0, 5.0, 1.0,                 # mu2, sigma2, beta_G2 (Peak 2)
            70.0, 10.0, 1.0,                # mu3, sigma3, beta_G3 (Peak 3)
            3.0                             # gamma (Global redshift evolution)
        ])

    else:
        raise ValueError(f"No fixed parameters defined for model '{pop_model}'")


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