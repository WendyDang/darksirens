# data.py

import jax.numpy as jnp
import healpy as hp
import numpy as np

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

from darksirens.em import zgrid, compute_lss_overdensity

GALAXY_AWARE_MODELS = ["dark_sirens", "dark_sirens_complete"]
BRIGHT_SIREN_MODELS = ["bright_sirens"]


def _compact_catalog_for_pixels(pixels, zgals, dzgals, wgals, ngals):
    """Return compact catalog rows and sample→row lookup for pixels."""
    pixels = np.asarray(pixels, dtype=np.int32)
    unique_pixels, sample_to_unique_idx = np.unique(pixels, return_inverse=True)
    unique_pixels = unique_pixels.astype(np.int32, copy=False)
    sample_to_unique_idx = sample_to_unique_idx.astype(np.int32, copy=False)
    return (
        unique_pixels,
        sample_to_unique_idx,
        zgals[unique_pixels],
        dzgals[unique_pixels],
        wgals[unique_pixels],
        ngals[unique_pixels],
    )


def _catalog_memory_diagnostics(zgals, dzgals, wgals, pixels_pe, pixels_sel, ngals_pe, ngals_sel):
    """Summarise memory saved by compact unique-pixel catalog views."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    row_bytes = sum(arr.dtype.itemsize * arr.shape[1] for arr in (zgals, dzgals, wgals))
    duplicated_pe = max(0, np.asarray(pixels_pe).size - unique_pe.size) * row_bytes
    duplicated_sel = max(0, np.asarray(pixels_sel).size - unique_sel.size) * row_bytes
    max_gals = 0
    if ngals_pe is not None and ngals_pe.size:
        max_gals = max(max_gals, int(np.max(ngals_pe)))
    if ngals_sel is not None and ngals_sel.size:
        max_gals = max(max_gals, int(np.max(ngals_sel)))
    return {
        "unique_pe_pixels": int(unique_pe.size),
        "unique_sel_pixels": int(unique_sel.size),
        "duplicated_catalog_bytes_avoided": int(duplicated_pe + duplicated_sel),
        "max_galaxies_per_unique_pixel": max_gals,
    }


def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data. 
    Handles cases where survey_path might be None (non-dark sirens models).
    """

    # 1. Initialize survey variables as None/defaults
    nside = None
    npix = None
    zgals = dzgals = wgals = None
    zgals_pe = dzgals_pe = wgals_pe = None
    zgals_sel = dzgals_sel = wgals_sel = None
    unique_pixels_pe = unique_pixels_sel = None
    sample_to_unique_pe = sample_to_unique_sel = None
    ngals = ngals_pe = ngals_sel = None
    catalog_memory = None
    apix = 0.0
    sigma_kernel = 0.0

    # 2. Load survey data, or build the synthetic one-object catalog used by
    # bright sirens.  The counterpart is not a survey hyperparameter: it is
    # fixed event metadata supplied through the inference CLI.
    if opts.universe_model in BRIGHT_SIREN_MODELS:
        if opts.counterpart is None:
            raise ValueError("bright_sirens requires opts.counterpart=(ra, dec, z).")
        ra_cp, dec_cp, z_cp = opts.counterpart
        nside = int(opts.counterpart_nside)
        npix = hp.nside2npix(nside)
        cp_pix = int(hp.ang2pix(nside, np.pi / 2.0 - dec_cp, ra_cp))

        zgals = np.zeros((npix, 1), dtype=float)
        dzgals = np.ones((npix, 1), dtype=float) * float(opts.counterpart_dz)
        wgals = np.zeros((npix, 1), dtype=float)
        ngals = np.zeros(npix, dtype=np.int32)

        zgals[cp_pix, 0] = z_cp
        wgals[cp_pix, 0] = 1.0
        ngals[cp_pix] = 1

        apix = hp.nside2pixarea(nside)
        sigma_kernel = opts.sigma_kernel
        print(
            "Using bright-siren counterpart catalog: "
            f"ra={ra_cp}, dec={dec_cp}, z={z_cp}, pixel={cp_pix}, nside={nside}"
        )
    elif opts.survey_path is not None:
        nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)
        npix = hp.nside2npix(nside)
        apix = hp.nside2pixarea(nside)
        sigma_kernel = opts.sigma_kernel
        print("Using a smoothing kernel of sigma: " + str(sigma_kernel))
    else:
        # If no survey, we might still need a default nside for 
        # pixelization logic in other parts of the code
        nside = 1
        npix = hp.nside2npix(nside)

    # 3. Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
        opts.gw_path
    )

    # 4. Load Selection samples (Always required)
    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = load_selection_samples(opts.gwselection_path)

    # 5. Pixel indexing and Galaxy lookups
    # Only perform these if a survey was actually loaded
    pixels_pe = hp.ang2pix(nside, jnp.pi/2 - dec, ra)
    pixels_sel = hp.ang2pix(nside, jnp.pi/2 - decsels, rasels)

    if zgals is not None:
        (
            unique_pixels_pe, sample_to_unique_pe,
            zgals_pe, dzgals_pe, wgals_pe, ngals_pe,
        ) = _compact_catalog_for_pixels(pixels_pe, zgals, dzgals, wgals, ngals)
        (
            unique_pixels_sel, sample_to_unique_sel,
            zgals_sel, dzgals_sel, wgals_sel, ngals_sel,
        ) = _compact_catalog_for_pixels(pixels_sel, zgals, dzgals, wgals, ngals)

        catalog_memory = _catalog_memory_diagnostics(
            zgals, dzgals, wgals, pixels_pe, pixels_sel, ngals_pe, ngals_sel
        )
        print("samples" + str(ngals_pe[sample_to_unique_pe].sum()))
        print("selection" + str(ngals_sel[sample_to_unique_sel].sum()))
        print(
            "    - Compact catalog rows: "
            f"PE {catalog_memory['unique_pe_pixels']:,}, "
            f"selection {catalog_memory['unique_sel_pixels']:,}"
        )
        print(
            "    - Duplicated catalog bytes avoided: "
            f"{catalog_memory['duplicated_catalog_bytes_avoided'] / 1e9:.4f} GB"
        )
        print(
            "    - Max galaxies per unique inference pixel: "
            f"{catalog_memory['max_galaxies_per_unique_pixel']:,}"
        )

    # 6. Pack into dictionary
    data = dict(
        # GW PE samples
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        p_pe=p_pe,
        pixels_pe=jnp.asarray(pixels_pe),
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,
        ngals=ngals_pe,
        unique_pixels_pe=unique_pixels_pe,
        sample_to_unique_pe=sample_to_unique_pe,

        # Selection samples
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        p_draw=p_draw,
        pixels_sel=jnp.asarray(pixels_sel),
        zgals_sel=zgals_sel,
        dzgals_sel=dzgals_sel,
        wgals_sel=wgals_sel,
        ngals_sel=ngals_sel,
        unique_pixels_sel=unique_pixels_sel,
        sample_to_unique_sel=sample_to_unique_sel,

        # Survey metadata and full catalog arrays.  Full pixel indexing is kept
        # only for operations that need global HEALPix rows, such as LSS
        # overdensity construction and startup cache generation.
        nEvents=nEvents,
        Ndraw=Ndraw,
        nsamp=nsamp,
        apix=apix,
        nside=nside,
        n_pix_catalog=npix,
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
        ngals_catalog=ngals,
        zgals_catalog=zgals,
        dzgals_catalog=dzgals,
        wgals_catalog=wgals,
        catalog_memory=catalog_memory,
        sigma_kernel=sigma_kernel
    )

    nEvents_check = data.get("nEvents", "Unknown")
    nside_check = data.get("nside", "N/A")
    print(f"    - Data loaded. Found {nEvents_check} GW events.")
    print(f"    - HEALPix nside detected: {nside_check}")

    # --------------------------------------------------------
    # LSS overdensity field (Handle memory carefully)
    # --------------------------------------------------------
    print(f"[*] Preparing LSS/Overdensity Field...")
    if opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS:
        print(f"    - Calculating high-resolution overdensity grid...")
        delta_g_pix_z = compute_lss_overdensity(
            data["zgals"],
            nside_check,
            wgals=data.get("wgals"),
            ngals=data.get("ngals_catalog"),
        )
    else:
        print(f"    - Non-LSS run. Creating memory-efficient dummy (1, {len(zgrid)}) grid.")
        # We use shape (1, nz) to satisfy JAX broadcasting without 93GB allocations
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))

    mem_usage = delta_g_pix_z.nbytes / 1e9
    print(f"    - Overdensity array shape: {delta_g_pix_z.shape} ({mem_usage:.4f} GB)")

    # Append the LSS overdensity field to the returned dictionary
    data["delta_g_pix_z"] = delta_g_pix_z

    return data