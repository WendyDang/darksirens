"""Catalog compaction and cache setup helpers for dark-siren inference."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import jax.numpy as jnp
from jax import lax
import numpy as np

from darksirens.em.completion import build_pixel_kde_cache

DARK_SIREN_CACHE_MODELS = {"dark_sirens"}


def barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """Apply the pre-JIT optimization barrier used by likelihood closures."""
    return lax.optimization_barrier(jnp.asarray(arr))


@dataclass(frozen=True)
class CatalogViews:
    """Barrier-wrapped PE/selection catalog views captured by the closure."""

    zgals_pe_catalog: jnp.ndarray
    dzgals_pe_catalog: jnp.ndarray
    wgals_pe_catalog: jnp.ndarray
    ngals_pe_catalog: jnp.ndarray | None
    zgals_sel_catalog: jnp.ndarray
    dzgals_sel_catalog: jnp.ndarray
    wgals_sel_catalog: jnp.ndarray
    ngals_sel_catalog: jnp.ndarray | None
    unique_pixels_pe: jnp.ndarray | None
    unique_pixels_sel: jnp.ndarray | None
    sample_to_unique_pe: jnp.ndarray
    sample_to_unique_sel: jnp.ndarray
    delta_g_pix_z: jnp.ndarray
    sigma_kernel: float
    dN_obs_kde_pe: jnp.ndarray | None
    dN_obs_kde_sel: jnp.ndarray | None
    pixel_to_cache_idx_pe: jnp.ndarray | None
    pixel_to_cache_idx_sel: jnp.ndarray | None


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


def _catalog_key(data: dict, compact_key: str, full_key: str) -> str:
    return compact_key if data.get(compact_key) is not None else full_key


def unique_inference_pixels(pixels_pe, pixels_sel, required_pixels=None) -> np.ndarray:
    """Return the sorted union of unique PE and selection HEALPix pixels."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    parts = [unique_pe, unique_sel]
    if required_pixels is not None:
        parts.append(np.asarray(required_pixels, dtype=np.int32).reshape(-1))
    return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)


def _full_catalog_arrays(data: dict):
    full_z = data.get("zgals_catalog") if data.get("zgals_catalog") is not None else data.get("zgals")
    full_dz = data.get("dzgals_catalog") if data.get("dzgals_catalog") is not None else data.get("dzgals")
    full_w = data.get("wgals_catalog") if data.get("wgals_catalog") is not None else data.get("wgals")
    full_n = data.get("ngals_catalog")
    return full_z, full_dz, full_w, full_n


def _ensure_compact(data: dict, prefix: str, pixels_key: str) -> bool:
    """Synthesize compact views for callers that still provide only full arrays."""
    if data.get(f"zgals_{prefix}") is not None:
        return True
    full_z, full_dz, full_w, full_n = _full_catalog_arrays(data)
    pixels = data.get(pixels_key)
    if any(value is None for value in (full_z, full_dz, full_w, full_n, pixels)):
        return False
    unique_pixels, sample_to_unique_idx = np.unique(
        np.asarray(pixels, dtype=np.int32), return_inverse=True
    )
    unique_pixels = unique_pixels.astype(np.int32, copy=False)
    data[f"unique_pixels_{prefix}"] = unique_pixels
    data[f"sample_to_unique_{prefix}"] = sample_to_unique_idx.astype(np.int32, copy=False)
    data[f"zgals_{prefix}"] = full_z[unique_pixels]
    data[f"dzgals_{prefix}"] = full_dz[unique_pixels]
    data[f"wgals_{prefix}"] = full_w[unique_pixels]
    data["ngals_pe" if prefix == "pe" else "ngals_sel"] = full_n[unique_pixels]
    return True


def prepare_catalog_views(
    opts,
    data: dict,
    universe_model: str,
    counterpart_pixel: int | None,
    cache_builder=build_pixel_kde_cache,
) -> CatalogViews:
    """Compact catalogs, build sample-to-unique maps, and prebuild KDE caches."""
    pe_uses_compact = data.get("zgals_pe") is not None
    sel_uses_compact = data.get("zgals_sel") is not None

    pe_uses_compact = pe_uses_compact or _ensure_compact(data, "pe", "pixels_pe")
    sel_uses_compact = sel_uses_compact or _ensure_compact(data, "sel", "pixels_sel")

    zgals_pe_catalog = barrier(_to_jax(data, _catalog_key(data, "zgals_pe", "zgals_catalog")))
    dzgals_pe_catalog = barrier(_to_jax(data, _catalog_key(data, "dzgals_pe", "dzgals_catalog")))
    wgals_pe_catalog = barrier(_to_jax(data, _catalog_key(data, "wgals_pe", "wgals_catalog")))
    ngals_pe_raw = data.get("ngals_pe") if pe_uses_compact else data.get("ngals_catalog")
    ngals_pe_catalog = (
        barrier(jnp.asarray(ngals_pe_raw, dtype=jnp.int32))
        if ngals_pe_raw is not None
        else None
    )

    zgals_sel_catalog = barrier(_to_jax(data, _catalog_key(data, "zgals_sel", "zgals_catalog")))
    dzgals_sel_catalog = barrier(_to_jax(data, _catalog_key(data, "dzgals_sel", "dzgals_catalog")))
    wgals_sel_catalog = barrier(_to_jax(data, _catalog_key(data, "wgals_sel", "wgals_catalog")))
    ngals_sel_raw = data.get("ngals_sel") if sel_uses_compact else data.get("ngals_catalog")
    ngals_sel_catalog = (
        barrier(jnp.asarray(ngals_sel_raw, dtype=jnp.int32))
        if ngals_sel_raw is not None
        else None
    )

    unique_pixels_pe_raw = data.get("unique_pixels_pe") if pe_uses_compact else None
    unique_pixels_sel_raw = data.get("unique_pixels_sel") if sel_uses_compact else None
    unique_pixels_pe = (
        barrier(jnp.asarray(unique_pixels_pe_raw, dtype=jnp.int32))
        if unique_pixels_pe_raw is not None
        else None
    )
    unique_pixels_sel = (
        barrier(jnp.asarray(unique_pixels_sel_raw, dtype=jnp.int32))
        if unique_pixels_sel_raw is not None
        else None
    )
    sample_to_unique_pe_raw = (
        data.get("sample_to_unique_pe") if pe_uses_compact else data.get("pixels_pe")
    )
    sample_to_unique_sel_raw = (
        data.get("sample_to_unique_sel") if sel_uses_compact else data.get("pixels_sel")
    )
    sample_to_unique_pe = barrier(jnp.asarray(sample_to_unique_pe_raw, dtype=jnp.int32))
    sample_to_unique_sel = barrier(jnp.asarray(sample_to_unique_sel_raw, dtype=jnp.int32))

    full_z, full_dz, full_w, full_n = _full_catalog_arrays(data)
    union_unique_pixels = None
    if all(
        value is not None
        for value in (full_z, full_dz, full_w, full_n, data.get("pixels_pe"), data.get("pixels_sel"))
    ):
        required_pixels = (
            [counterpart_pixel]
            if universe_model == "bright_sirens" and counterpart_pixel is not None
            else None
        )
        union_unique_pixels = unique_inference_pixels(
            data["pixels_pe"], data["pixels_sel"], required_pixels=required_pixels
        )
        pe_global_pixels = np.asarray(data["pixels_pe"], dtype=np.int32)
        sel_global_pixels = np.asarray(data["pixels_sel"], dtype=np.int32)
        sample_to_union_pe_raw = np.searchsorted(
            union_unique_pixels, pe_global_pixels
        ).astype(np.int32, copy=False)
        sample_to_union_sel_raw = np.searchsorted(
            union_unique_pixels, sel_global_pixels
        ).astype(np.int32, copy=False)

        zgals_union_catalog = barrier(jnp.asarray(full_z[union_unique_pixels]))
        dzgals_union_catalog = barrier(jnp.asarray(full_dz[union_unique_pixels]))
        wgals_union_catalog = barrier(jnp.asarray(full_w[union_unique_pixels]))
        ngals_union_catalog = barrier(jnp.asarray(full_n[union_unique_pixels], dtype=jnp.int32))
        unique_pixels_union = barrier(jnp.asarray(union_unique_pixels, dtype=jnp.int32))
        sample_to_unique_pe = barrier(jnp.asarray(sample_to_union_pe_raw, dtype=jnp.int32))
        sample_to_unique_sel = barrier(jnp.asarray(sample_to_union_sel_raw, dtype=jnp.int32))

        zgals_pe_catalog = zgals_sel_catalog = zgals_union_catalog
        dzgals_pe_catalog = dzgals_sel_catalog = dzgals_union_catalog
        wgals_pe_catalog = wgals_sel_catalog = wgals_union_catalog
        ngals_pe_catalog = ngals_sel_catalog = ngals_union_catalog
        unique_pixels_pe = unique_pixels_sel = unique_pixels_union

    delta_g_pix_z = barrier(_to_jax(data, "delta_g_pix_z"))
    sigma_kernel = data["sigma_kernel"]

    dN_obs_kde_pe = dN_obs_kde_sel = None
    pixel_to_cache_idx_pe = pixel_to_cache_idx_sel = None
    cache_required = universe_model in DARK_SIREN_CACHE_MODELS

    if cache_required:
        if union_unique_pixels is not None:
            dN_obs_kde_pe, pixel_to_cache_idx_pe = cache_builder(
                unique_pixels=union_unique_pixels,
                zgals=full_z,
                n_pix_catalog=int(data.get("n_pix_catalog", np.asarray(full_z).shape[0])),
                wgals=full_w,
                ngals=full_n,
            )
            dN_obs_kde_sel = dN_obs_kde_pe
            pixel_to_cache_idx_sel = pixel_to_cache_idx_pe
        else:
            missing_cache_inputs = [
                name
                for name, value in (
                    ("PE compact galaxy redshifts", data.get("zgals_pe")),
                    ("selection compact galaxy redshifts", data.get("zgals_sel")),
                    ("PE sample-to-unique map", data.get("sample_to_unique_pe")),
                    ("selection sample-to-unique map", data.get("sample_to_unique_sel")),
                    (
                        "PE galaxy mask (wgals or ngals)",
                        data.get("wgals_pe") if data.get("wgals_pe") is not None else data.get("ngals_pe"),
                    ),
                    (
                        "selection galaxy mask (wgals or ngals)",
                        data.get("wgals_sel") if data.get("wgals_sel") is not None else data.get("ngals_sel"),
                    ),
                )
                if value is None
            ]
            if missing_cache_inputs:
                message = (
                    "Dark-siren inference requires the per-pixel KDE cache; "
                    f"cannot build it because these inputs are missing: {', '.join(missing_cache_inputs)}."
                )
                if getattr(opts, "allow_uncached_dark_sirens", False):
                    warnings.warn(
                        message
                        + " Falling back to uncached completion for tests/backward compatibility.",
                        RuntimeWarning,
                    )
                else:
                    raise RuntimeError(message)
            else:
                n_pe_rows = int(np.asarray(data["zgals_pe"]).shape[0])
                n_sel_rows = int(np.asarray(data["zgals_sel"]).shape[0])
                dN_obs_kde_pe, pixel_to_cache_idx_pe = cache_builder(
                    unique_pixels=np.arange(n_pe_rows, dtype=np.int32),
                    zgals=data["zgals_pe"],
                    n_pix_catalog=n_pe_rows,
                    wgals=data.get("wgals_pe"),
                    ngals=data.get("ngals_pe"),
                )
                dN_obs_kde_sel, pixel_to_cache_idx_sel = cache_builder(
                    unique_pixels=np.arange(n_sel_rows, dtype=np.int32),
                    zgals=data["zgals_sel"],
                    n_pix_catalog=n_sel_rows,
                    wgals=data.get("wgals_sel"),
                    ngals=data.get("ngals_sel"),
                )

    dN_obs_kde_pe = barrier(dN_obs_kde_pe) if dN_obs_kde_pe is not None else None
    dN_obs_kde_sel = barrier(dN_obs_kde_sel) if dN_obs_kde_sel is not None else None
    pixel_to_cache_idx_pe = (
        barrier(jnp.asarray(pixel_to_cache_idx_pe, dtype=jnp.int32))
        if pixel_to_cache_idx_pe is not None
        else None
    )
    pixel_to_cache_idx_sel = (
        barrier(jnp.asarray(pixel_to_cache_idx_sel, dtype=jnp.int32))
        if pixel_to_cache_idx_sel is not None
        else None
    )

    return CatalogViews(
        zgals_pe_catalog=zgals_pe_catalog,
        dzgals_pe_catalog=dzgals_pe_catalog,
        wgals_pe_catalog=wgals_pe_catalog,
        ngals_pe_catalog=ngals_pe_catalog,
        zgals_sel_catalog=zgals_sel_catalog,
        dzgals_sel_catalog=dzgals_sel_catalog,
        wgals_sel_catalog=wgals_sel_catalog,
        ngals_sel_catalog=ngals_sel_catalog,
        unique_pixels_pe=unique_pixels_pe,
        unique_pixels_sel=unique_pixels_sel,
        sample_to_unique_pe=sample_to_unique_pe,
        sample_to_unique_sel=sample_to_unique_sel,
        delta_g_pix_z=delta_g_pix_z,
        sigma_kernel=sigma_kernel,
        dN_obs_kde_pe=dN_obs_kde_pe,
        dN_obs_kde_sel=dN_obs_kde_sel,
        pixel_to_cache_idx_pe=pixel_to_cache_idx_pe,
        pixel_to_cache_idx_sel=pixel_to_cache_idx_sel,
    )
