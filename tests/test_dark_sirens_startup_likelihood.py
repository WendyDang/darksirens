from types import SimpleNamespace
import sys
import types

import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np

# The lightweight likelihood startup fixture uses a parametric population model,
# but importing the registry also imports optional GP model classes.  Keep this
# regression test independent of the optional tinygp dependency.
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


from darksirens.em import zgrid
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.likelihood import make_likelihood


def test_dark_sirens_likelihood_evaluates_once_before_sampling():
    """Small startup regression: one dark-sirens likelihood call must not fail."""
    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    n_events = 1
    nsamp = 2
    n_sel = 16

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)

    pixels_pe = jnp.array([7, 7], dtype=jnp.int32)
    pixels_sel = jnp.array([2, 7, *([2] * (n_sel - 2))], dtype=jnp.int32)

    data = {
        "nEvents": n_events,
        "nsamp": nsamp,
        "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside),
        "nside": nside,
        "n_pix_catalog": n_pix_catalog,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((n_pix_catalog, len(zgrid))),
        "sigma_kernel": 0.02,
        "m1det": jnp.array([36.0, 38.0]),
        "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]),
        "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp),
        "pixels_pe": pixels_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": pixels_sel,
    }
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    likelihood = make_likelihood(
        opts,
        data,
        get_fixed_population_params(opts.pop_model),
    )

    value = likelihood(jnp.array([]))
    assert value.shape == ()
    assert not bool(jnp.isnan(value))
