import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.inference.events import make_gw_event
from darksirens.inference.selection import compute_selection_term
from darksirens.utils.containers import EMCatalog


def test_selection_batching_matches_unbatched_for_non_divisible_length():
    """Regression: final incomplete selection batch must not be dropped."""
    n_sel = 10
    gw_sel = make_gw_event(
        m1det=jnp.linspace(30.0, 40.0, n_sel),
        m2det=jnp.linspace(24.0, 32.0, n_sel),
        dL=jnp.linspace(400.0, 600.0, n_sel),
        chieff=jnp.zeros(n_sel),
        prior_wt=jnp.ones(n_sel),
        pixels=jnp.zeros(n_sel, dtype=jnp.int32),
    )
    catalog = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.zeros((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones(1, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        sigma_kernel=0.1,
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )

    def constant_log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        # Deliberately ignore prior_wt. The batching implementation must still
        # use the explicit prior_wt == 0 padding sentinel as a structural mask;
        # otherwise the two padded rows in a size-4 scan would contribute weight.
        return jnp.zeros_like(dL)

    unbatched = compute_selection_term(
        gw_sel,
        catalog,
        constant_log_weight,
        Ndraw=float(n_sel),
        nEvents=1,
        sel_batch_size=None,
    )
    batched = compute_selection_term(
        gw_sel,
        catalog,
        constant_log_weight,
        Ndraw=float(n_sel),
        nEvents=1,
        sel_batch_size=4,
    )

    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(unbatched[0]))
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(unbatched[1]))
