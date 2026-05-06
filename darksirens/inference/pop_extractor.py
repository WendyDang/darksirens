"""
pop_extractor.py
----------------
make_pop_extractor(settings) — single source of truth for extracting the
population parameter sub-vector from the flat sampled coordinate vector theta.
"""

from __future__ import annotations
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser, get_fixed_population_params


def make_pop_extractor(settings: dict):
    """
    Return a JAX-compatible function  pop_theta = extractor(theta).

    Parameters
    ----------
    settings : dict  (loaded from the run's settings.json)
        Required key: "pop_model"
        Optional keys: "fix_population", "fix_cosmology", "fix_survey",
                       "fixed_parameter_values"

    Returns
    -------
    extractor : callable
        theta (1-D sampled coordinate vector) → pop_params (1-D array).
        Safe inside jax.jit and jax.vmap.
    """
    pop_model_name         = settings["pop_model"]
    fix_population         = bool(settings.get("fix_population", False))
    fix_cosmology          = bool(settings.get("fix_cosmology",  False))
    fix_survey             = bool(settings.get("fix_survey",     False))
    fixed_parameter_values = settings.get("fixed_parameter_values") or {}

    # Fast path: entire population block is fixed.
    if fix_population:
        fixed_array = get_fixed_population_params(pop_model_name)
        def extractor_fixed(theta):
            return fixed_array
        return extractor_fixed

    # General path: use build_parameter_space as the ordering oracle.
    # This is the same function make_likelihood calls, so the two are
    # guaranteed to agree on parameter ordering.
    from darksirens.inference.prior import build_parameter_space

    (
        labels,          # full ordered list of sampled labels
        *_,
        pop_labels,      # population labels in model order
        _survey_labels,
        _cosmo_labels,
        *__,
    ) = build_parameter_space(
        pop_model              = pop_model_name,
        fix_population         = fix_population,
        fix_cosmology          = fix_cosmology,
        fix_survey             = fix_survey,
        fixed_parameter_values = fixed_parameter_values,
    )

    label_to_coord_idx = {label: idx for idx, label in enumerate(labels)}

    pop_coord_indices = []
    pop_fixed_mask    = []
    pop_fixed_values  = []

    for label in pop_labels:
        if label in fixed_parameter_values:
            pop_coord_indices.append(0)          # dummy; masked below
            pop_fixed_mask.append(True)
            pop_fixed_values.append(float(fixed_parameter_values[label]))
        else:
            if label not in label_to_coord_idx:
                raise KeyError(
                    f"Population label '{label}' not found in sampled coordinate "
                    f"labels — check fixed_parameter_values / fix_* flags."
                )
            pop_coord_indices.append(label_to_coord_idx[label])
            pop_fixed_mask.append(False)
            pop_fixed_values.append(0.0)

    idx_jnp   = jnp.array(pop_coord_indices, dtype=jnp.int32)
    mask_jnp  = jnp.array(pop_fixed_mask,    dtype=bool)
    fixed_jnp = jnp.array(pop_fixed_values,  dtype=jnp.float64)

    def extractor(theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mask_jnp, fixed_jnp, theta[idx_jnp])

    return extractor