import numpy as np
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck

def get_fixed_population_params(pop_model):
    """
    Fiducial population parameters used when --fix_population=True.
    These must match the exact dimension and component ordering of the backend parser:
    Weights -> Masses -> Betas -> Spins -> Gamma.
    """
    
    # 1. Parse the requested model for shared flags
    base_model = pop_model
    if "_shared_beta_spin" in base_model:
        base_model = base_model.replace("_shared_beta_spin", "")
        shared_beta = True
        shared_spin = True
    else:
        shared_beta = "_shared_beta" in base_model
        shared_spin = "_shared_spin" in base_model
        base_model = base_model.replace("_shared_beta", "").replace("_shared_spin", "")

    # 2. Define Weights and Mass parameters per base model
    if base_model == "powerlaw+peak":
        n_comp = 2
        weights = [0.1]
        masses = [
            2.0, 5.0, 80.0, 0.5, 10.0,  # PL: alpha, m_min, m_max, dm_min, dm_max
            35.0, 5.0                   # G: mu, sigma
        ]
        
    elif base_model == "brokenpowerlaw+2peaks":
        n_comp = 3
        weights = [0.10, 0.05]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL: a1, a2, m_break, m_min, m_max, dm_min, dm_max
            10.0, 3.0,                            # G1: mu1, sigma1
            35.0, 5.0                             # G2: mu2, sigma2
        ]
        
    elif base_model == "brokenpowerlaw+3peaks":
        n_comp = 4
        weights = [0.10, 0.05, 0.03]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL
            10.0, 3.0,                            # G1
            35.0, 5.0,                            # G2
            70.0, 10.0                            # G3: mu3, sigma3
        ]
        
    elif base_model == "twopowerlaws+peak":
        n_comp = 3
        weights = [0.20, 0.10]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            35.0, 5.0                  # G
        ]
        
    elif base_model == "twopowerlaws+2peaks":
        n_comp = 4
        weights = [0.15, 0.10, 0.05]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            10.0, 3.0,                 # G1
            35.0, 5.0                  # G2
        ]
        
    elif base_model == "twopowerlaws+3peaks":
        n_comp = 5
        weights = [0.15, 0.10, 0.05, 0.03]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            10.0, 3.0,                 # G1
            35.0, 5.0,                 # G2
            70.0, 10.0                 # G3
        ]
        
    elif base_model == "brokenpowerlaw+2peaks+powerlaw":
        n_comp = 4
        weights = [0.10, 0.05, 0.05]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL
            10.0, 3.0,                            # G1
            35.0, 5.0,                            # G2
            3.0, 50.0, 100.0, 3.0, 10.0           # PL
        ]
        
    else:
        raise ValueError(f"No fixed parameters defined for model '{pop_model}'")

    # 3. Define Pairing (Beta) Parameters
    betas = [1.0] if shared_beta else [1.0] * n_comp
    
    # 4. Define Spin Parameters (mu_chi=0.0, sigma_chi=0.1)
    spins = [0.0, 0.1] if shared_spin else [0.0, 0.1] * n_comp
    
    # 5. Define Gamma
    gamma = [3.0]

    # Assemble global array adhering to unified strict ordering
    full_params = weights + masses + betas + spins + gamma
    return jnp.array(full_params)


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