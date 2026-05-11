import numpy as np
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.utils.cosmology import Om0Planck

def apply_block_prior_overrides(block_name, labels, lower, upper, overrides):
    """Apply flat per-parameter prior overrides to a parameter block.

    Supported format:
        {"param_name": [low, high], ...}
    """
    if overrides is None:
        return list(lower), list(upper)

    if not isinstance(overrides, dict):
        raise TypeError(
            f"Prior overrides for block '{block_name}' must be a dict, got {type(overrides).__name__}."
        )

    lower_out = list(lower)
    upper_out = list(upper)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for label, bounds in overrides.items():
        if label not in label_to_index:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Override for '{label}' in block '{block_name}' must be [lower, upper]."
            )
        idx = label_to_index[label]
        lower_out[idx] = bounds[0]
        upper_out[idx] = bounds[1]

    return lower_out, upper_out


def apply_block_fixed_values(block_name, labels, lower, upper, fixed_values):
    """Apply per-parameter fixed values by collapsing bounds to [value, value]."""
    if fixed_values is None:
        return list(lower), list(upper)

    if not isinstance(fixed_values, dict):
        raise TypeError(
            f"Fixed values for block '{block_name}' must be a dict, got {type(fixed_values).__name__}."
        )

    lower_out = list(lower)
    upper_out = list(upper)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for label, value in fixed_values.items():
        if label not in label_to_index:
            continue
        idx = label_to_index[label]
        v = float(value)
        lower_out[idx] = v
        upper_out[idx] = v

    return lower_out, upper_out

def build_parameter_space(
    pop_model,
    fix_population,
    fix_cosmology,
    fix_survey,
    prior_overrides=None,
    fixed_parameter_values=None,
):
    """Construct labels and prior bounds for cosmological, population, and survey parameters."""
    if prior_overrides is None:
        prior_overrides = {}
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    # --- Cosmology ---
    cosmo_labels = ["H0", "Om0"]
    cosmo_lower = [20.0, Om0Planck - 0.1]
    cosmo_upper = [120.0, Om0Planck + 0.1]

    # --- Population ---
    pop_lower, pop_upper, pop_labels, model_name = pop_model_prior_parser(pop_model)

    # --- Survey ---
    # ``log10n0`` is log10 of the comoving galaxy density in Mpc^-3,
    # matching dV_of_z [Mpc^3 sr^-1 dz^-1] times the HEALPix pixel area.
    # The redshift grid used by the completion model spans 0 <= z <= 5;
    # these defaults keep the survey rolloff inside that domain while avoiding
    # the formerly ultra-broad density/evolution fits that could force heavy
    # clipping throughout the completion grid.
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha_miss"]
    survey_lower = [-4.0, 0.05, 0.02, -3.0, 0.0, 0.0]
    survey_upper = [-1.0, 4.5, 1.5, 3.0, 3.0, 1.0]

    # Make sure all prior override keys are valid parameter labels
    known_labels = set(cosmo_labels) | set(pop_labels) | set(survey_labels)
    unknown = [k for k in prior_overrides.keys() if k not in known_labels]
    if unknown:
        raise KeyError(
            f"Unknown prior override labels: {unknown}. Valid labels for pop_model='{pop_model}': "
            f"{sorted(known_labels)}"
        )

    unknown_fixed = [k for k in fixed_parameter_values.keys() if k not in known_labels]
    if unknown_fixed:
        raise KeyError(
            f"Unknown fixed parameter labels: {unknown_fixed}. Valid labels for pop_model='{pop_model}': "
            f"{sorted(known_labels)}"
        )

    # Apply block overrides
    cosmo_lower, cosmo_upper = apply_block_prior_overrides(
        "cosmology", cosmo_labels, cosmo_lower, cosmo_upper, prior_overrides
    )
    pop_lower, pop_upper = apply_block_prior_overrides(
        "population", pop_labels, pop_lower, pop_upper, prior_overrides
    )
    survey_lower, survey_upper = apply_block_prior_overrides(
        "survey", survey_labels, survey_lower, survey_upper, prior_overrides
    )

    # Apply fixed-value presets after prior overrides so fixed values always win.
    cosmo_lower, cosmo_upper = apply_block_fixed_values(
        "cosmology", cosmo_labels, cosmo_lower, cosmo_upper, fixed_parameter_values
    )
    pop_lower, pop_upper = apply_block_fixed_values(
        "population", pop_labels, pop_lower, pop_upper, fixed_parameter_values
    )
    survey_lower, survey_upper = apply_block_fixed_values(
        "survey", survey_labels, survey_lower, survey_upper, fixed_parameter_values
    )

    n_cosmo = len(cosmo_labels)
    n_pop = len(pop_labels)
    n_survey = len(survey_labels)

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
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    def prior_transform(u):
        return u * (upper - lower) + lower
    return prior_transform