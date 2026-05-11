import sys
import types

if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        pass

    class _KernelsStub:
        class Matern52:
            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub

from darksirens.inference.prior import build_parameter_space


def test_survey_default_priors_are_physical_and_overridable():
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}

    assert bounds["log10n0"] == (-4.0, -1.0)
    assert bounds["z50"] == (0.05, 4.5)
    assert bounds["w"] == (0.02, 1.5)
    assert bounds["delta"] == (-3.0, 3.0)
    assert bounds["b_miss"] == (0.0, 3.0)
    assert bounds["alpha_miss"] == (0.0, 1.0)

    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={"log10n0": [-6.0, -2.0]},
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}
    assert bounds["log10n0"] == (-6.0, -2.0)
