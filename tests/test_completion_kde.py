import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.em.completion import _kde_dndz_obs, build_pixel_kde_cache
from darksirens.em import zgrid


def test_kde_masks_empty_pixel_with_wgals_indicator():
    """An empty padded pixel must not contribute a fake z=0 galaxy."""
    zgals = jnp.zeros((1, 4))
    wgals = jnp.zeros((1, 4))

    dndz = _kde_dndz_obs(0, zgals, wgals=wgals)

    np.testing.assert_allclose(np.asarray(dndz), 0.0, atol=1e-14)


def test_kde_masks_partially_padded_pixel_with_ngals_indicator():
    """Padded zeros after the real entries must not create a low-z spike."""
    zgals = jnp.array([[0.5, 0.0, 0.0, 0.0]])
    ngals = jnp.array([1], dtype=jnp.int32)

    dndz = _kde_dndz_obs(0, zgals, ngals=ngals)
    low_z_value = float(dndz[0])
    real_gal_idx = int(jnp.argmin(jnp.abs(zgrid - 0.5)))
    real_gal_value = float(dndz[real_gal_idx])

    assert low_z_value < 1e-5
    assert real_gal_value > 0.5


def test_build_pixel_kde_cache_masks_empty_and_partially_padded_pixels():
    """Cached KDEs use the same real-galaxy masks as the uncached path."""
    zgals = jnp.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    ngals = jnp.array([0, 1], dtype=jnp.int32)

    dN_obs_kde, pixel_to_cache_idx = build_pixel_kde_cache(
        unique_pixels=np.array([0, 1], dtype=np.int32),
        zgals=zgals,
        n_pix_catalog=2,
        ngals=ngals,
    )

    np.testing.assert_allclose(
        np.asarray(dN_obs_kde[pixel_to_cache_idx[0]]), 0.0, atol=1e-14
    )
    assert float(dN_obs_kde[pixel_to_cache_idx[1], 0]) < 1e-5
