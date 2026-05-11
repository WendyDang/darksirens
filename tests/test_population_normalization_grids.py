import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import sys
import types

# Parametric tests do not evaluate GP models, but importing the population
# registry imports optional GP classes. Keep these tests independent of tinygp.
if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub


from darksirens.gw.populations.base import ParamSpec
from darksirens.gw.populations.parametric import (
    BrokenPowerLaw,
    PowerLaw,
    PowerLawPairing,
    TruncatedGaussianSpin,
)
from darksirens.gw.populations.registry import get_fixed_population_params, pop_model_parser
from darksirens.gw.populations.utils import (
    configure_normalization_grids,
    get_chi_grid,
    get_mass_grid,
    get_q_grid,
    normalization_grid_settings,
)


def teardown_function():
    configure_normalization_grids(n_mass=500, n_q=200, n_chi=200)


def _relative_difference(coarse, reference):
    coarse = float(coarse)
    reference = float(reference)
    return abs(coarse - reference) / max(abs(reference), 1.0e-300)


def test_normalization_grid_settings_are_configurable_and_cached():
    configure_normalization_grids(n_mass=321, n_q=123, n_chi=77)

    settings = normalization_grid_settings()
    assert settings.n_mass == 321
    assert settings.n_q == 123
    assert settings.n_chi == 77
    assert get_mass_grid().shape == (321,)
    assert get_q_grid().shape == (123,)
    assert get_chi_grid().shape == (77,)


def test_parametric_mass_norms_converge_near_minimum_smoothing_widths():
    pl = PowerLaw(
        ParamSpec("alpha", -4.0, 6.0),
        ParamSpec("mmin", 2.0, 10.0),
        ParamSpec("mmax", 50.0, 100.0),
        ParamSpec("dmmin", 0.01, 10.0),
        ParamSpec("dmmax", 0.01, 20.0),
    )
    bpl = BrokenPowerLaw(
        ParamSpec("alpha1", 0.0, 6.0),
        ParamSpec("alpha2", 0.0, 6.0),
        ParamSpec("mb", 20.0, 50.0),
        ParamSpec("mmin", 2.0, 10.0),
        ParamSpec("mmax", 50.0, 100.0),
        ParamSpec("dmmin", 0.01, 10.0),
        ParamSpec("dmmax", 0.01, 20.0),
    )
    cases = [
        (pl, jnp.array([2.3, 2.0, 50.0, 0.01, 0.01])),
        (bpl, jnp.array([1.6, 3.8, 35.0, 2.0, 50.0, 0.01, 0.01])),
    ]

    configure_normalization_grids(n_mass=500)
    coarse = [component._norm(theta) for component, theta in cases]

    configure_normalization_grids(n_mass=20000)
    reference = [component._norm(theta) for component, theta in cases]

    for coarse_norm, reference_norm in zip(coarse, reference):
        assert _relative_difference(coarse_norm, reference_norm) < 3.0e-3


def test_pairing_and_spin_norms_converge_near_narrow_features():
    pairing = PowerLawPairing(ParamSpec("beta", -2.0, 7.0))
    spin = TruncatedGaussianSpin(
        ParamSpec("mu_chi", -1.0, 1.0),
        ParamSpec("sigma_chi", 0.01, 1.0),
    )
    m1 = jnp.array([3.0, 10.0, 80.0])
    q = jnp.array([0.8, 0.5, 0.3])
    pairing_theta = jnp.array([2.0])
    spin_theta = jnp.array([0.0, 0.01])

    configure_normalization_grids(n_q=200, n_chi=200)
    coarse_pair = pairing(m1, q, 2.0, 0.01, pairing_theta)
    coarse_spin_norm = spin._norm(spin_theta)

    configure_normalization_grids(n_q=10000, n_chi=10000)
    ref_pair = pairing(m1, q, 2.0, 0.01, pairing_theta)
    ref_spin_norm = spin._norm(spin_theta)

    np.testing.assert_allclose(coarse_pair, ref_pair, rtol=1.5e-2, atol=0.0)
    assert _relative_difference(coarse_spin_norm, ref_spin_norm) < 1.0e-3


def test_representative_population_model_is_stable_at_production_grid_sizes():
    theta = get_fixed_population_params("powerlaw+peak")
    log_p_pop = pop_model_parser("powerlaw+peak")
    m1 = jnp.array([8.0, 20.0, 45.0, 70.0])
    q = jnp.array([0.4, 0.7, 0.9, 0.5])
    z = jnp.array([0.05, 0.2, 0.6, 1.0])
    chi = jnp.array([-0.1, 0.0, 0.2, 0.5])

    configure_normalization_grids(n_mass=500, n_q=200, n_chi=200)
    coarse = log_p_pop(m1, q, z, chi, theta)

    configure_normalization_grids(n_mass=3000, n_q=2000, n_chi=2000)
    reference = log_p_pop(m1, q, z, chi, theta)

    np.testing.assert_allclose(coarse, reference, rtol=2.0e-2, atol=2.0e-2)
