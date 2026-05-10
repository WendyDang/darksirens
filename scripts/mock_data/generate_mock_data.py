#!/usr/bin/env python3
"""Generate end-to-end mock data for the dark-sirens pipeline.

The mock is intentionally simple and transparent:

* galaxies are isotropic on the sky and uniform in comoving volume;
* GW hosts are drawn from the complete catalog, before EM incompleteness;
* BBH masses/spins use a POWER LAW + PEAK model with one shared beta and one
  shared truncated-Gaussian chi_eff spin distribution;
* GW detectability is a semi-analytic network-SNR threshold;
* the observed EM survey is produced by applying a footprint, redshift/magnitude
  limits, and a smooth redshift-dependent completeness curve.

The HDF5 files are written in the formats consumed by ``darksirens_inference``
and ``darksirens_pixelate``/``load_survey``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit

C_KM_S = 299_792.458


@dataclass(frozen=True)
class PopulationConfig:
    """Fiducial POWER LAW + PEAK, shared beta, shared spin parameters."""

    alpha: float = 3.4
    mmin: float = 5.0
    mmax: float = 85.0
    peak_fraction: float = 0.10
    peak_mu: float = 35.0
    peak_sigma: float = 4.0
    beta: float = 1.3
    chi_mu: float = 0.0
    chi_sigma: float = 0.15
    gamma: float = 0.0


@dataclass(frozen=True)
class SurveyConfig:
    """Simple EM selection model for catalog incompleteness."""

    footprint_dec_min_deg: float = -40.0
    footprint_dec_max_deg: float = 80.0
    z_hard_max: float = 1.2
    magnitude_limit: float = 24.0
    z50: float = 0.75
    width: float = 0.12
    absolute_mag_mean: float = -21.0
    absolute_mag_sigma: float = 1.0
    redshift_error_floor: float = 0.0005
    redshift_error_slope: float = 0.0015


def _build_cosmology(h0: float, om0: float) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=h0 * u.km / u.s / u.Mpc, Om0=om0)


def _cosmology_grids(cosmo: FlatLambdaCDM, zmax: float, ngrid: int = 20_000) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, zmax, ngrid)
    dc = cosmo.comoving_distance(z).to_value(u.Mpc)
    dl = cosmo.luminosity_distance(z).to_value(u.Mpc)
    ez = np.sqrt(cosmo.Om0 * (1.0 + z) ** 3 + (1.0 - cosmo.Om0))
    dvc_dz = 4.0 * np.pi * (C_KM_S / cosmo.H0.value) * dc**2 / ez
    vc_cdf = cumulative_trapezoid(dvc_dz, z, initial=0.0)
    vc_cdf /= vc_cdf[-1]
    return {"z": z, "dc": dc, "dl": dl, "dvc_dz": dvc_dz, "vc_cdf": vc_cdf}


def _sample_uniform_comoving_z(rng: np.random.Generator, grids: dict[str, np.ndarray], n: int) -> np.ndarray:
    return np.interp(rng.uniform(size=n), grids["vc_cdf"], grids["z"])


def _interp_dl(z: np.ndarray, grids: dict[str, np.ndarray]) -> np.ndarray:
    return np.interp(z, grids["z"], grids["dl"])


def _sample_sky(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    ra = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_dec = rng.uniform(-1.0, 1.0, n)
    dec = np.arcsin(sin_dec)
    return ra, dec


def _powerlaw_pdf(m: np.ndarray, alpha: float, mmin: float, mmax: float) -> np.ndarray:
    out = np.zeros_like(m, dtype=float)
    mask = (m >= mmin) & (m <= mmax)
    if np.isclose(alpha, 1.0):
        norm = np.log(mmax / mmin)
    else:
        norm = (mmax ** (1.0 - alpha) - mmin ** (1.0 - alpha)) / (1.0 - alpha)
    out[mask] = m[mask] ** (-alpha) / norm
    return out


def _truncnorm_pdf(x: np.ndarray, mu: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    from scipy.stats import norm as normal_dist

    z_norm = sigma * (normal_dist.cdf((hi - mu) / sigma) - normal_dist.cdf((lo - mu) / sigma))
    out = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * z_norm)
    return np.where((x >= lo) & (x <= hi), out, 0.0)


def _sample_powerlaw(rng: np.random.Generator, n: int, alpha: float, mmin: float, mmax: float) -> np.ndarray:
    u = rng.uniform(size=n)
    if np.isclose(alpha, 1.0):
        return mmin * (mmax / mmin) ** u
    a = 1.0 - alpha
    return (u * (mmax**a - mmin**a) + mmin**a) ** (1.0 / a)


def _sample_powerlaw_peak_m1(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    use_peak = rng.uniform(size=n) < pop.peak_fraction
    m1 = _sample_powerlaw(rng, n, pop.alpha, pop.mmin, pop.mmax)
    n_peak = int(use_peak.sum())
    if n_peak:
        draws = []
        while sum(map(len, draws)) < n_peak:
            cand = rng.normal(pop.peak_mu, pop.peak_sigma, n_peak)
            cand = cand[(cand >= pop.mmin) & (cand <= pop.mmax)]
            draws.append(cand)
        m1[use_peak] = np.concatenate(draws)[:n_peak]
    return m1


def _sample_q(rng: np.random.Generator, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    qmin = np.clip(pop.mmin / m1, 1.0e-3, 1.0)
    u = rng.uniform(size=len(m1))
    b = pop.beta
    if np.isclose(b, -1.0):
        return qmin * (1.0 / qmin) ** u
    bp1 = b + 1.0
    return (u * (1.0 - qmin**bp1) + qmin**bp1) ** (1.0 / bp1)


def _q_pdf(q: np.ndarray, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    qmin = np.clip(pop.mmin / m1, 1.0e-3, 1.0)
    out = np.zeros_like(q, dtype=float)
    mask = (q >= qmin) & (q <= 1.0)
    if np.isclose(pop.beta, -1.0):
        norm = np.log(1.0 / qmin)
    else:
        norm = (1.0 - qmin ** (pop.beta + 1.0)) / (pop.beta + 1.0)
    out[mask] = q[mask] ** pop.beta / norm[mask]
    return out


def _sample_chieff(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    vals = []
    while sum(map(len, vals)) < n:
        cand = rng.normal(pop.chi_mu, pop.chi_sigma, n)
        vals.append(cand[(cand >= -1.0) & (cand <= 1.0)])
    return np.concatenate(vals)[:n]


def _mass_spin_pdf(m1: np.ndarray, q: np.ndarray, chi: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    p_pl = _powerlaw_pdf(m1, pop.alpha, pop.mmin, pop.mmax)
    p_pk = _truncnorm_pdf(m1, pop.peak_mu, pop.peak_sigma, pop.mmin, pop.mmax)
    p_m1 = (1.0 - pop.peak_fraction) * p_pl + pop.peak_fraction * p_pk
    p_chi = _truncnorm_pdf(chi, pop.chi_mu, pop.chi_sigma, -1.0, 1.0)
    return p_m1 * _q_pdf(q, m1, pop) * p_chi


def _network_snr(m1: np.ndarray, m2: np.ndarray, z: np.ndarray, dl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mchirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mchirp_det = mchirp * (1.0 + z)
    projection = rng.beta(2.0, 5.0, size=len(np.atleast_1d(m1))) ** 0.5
    rho_ref = 11.5
    return rho_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl) * projection


def _generate_complete_catalog(
    rng: np.random.Generator,
    n_galaxies: int,
    grids: dict[str, np.ndarray],
    survey: SurveyConfig,
) -> dict[str, np.ndarray]:
    z = _sample_uniform_comoving_z(rng, grids, n_galaxies)
    ra, dec = _sample_sky(rng, n_galaxies)
    abs_mag = rng.normal(survey.absolute_mag_mean, survey.absolute_mag_sigma, n_galaxies)
    dl_pc = _interp_dl(z, grids) * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    return {"ra": ra, "dec": dec, "z": z, "abs_mag": abs_mag, "app_mag": app_mag}


def _apply_survey_selection(
    rng: np.random.Generator,
    catalog: dict[str, np.ndarray],
    survey: SurveyConfig,
) -> np.ndarray:
    dec_deg = np.rad2deg(catalog["dec"])
    footprint = (dec_deg >= survey.footprint_dec_min_deg) & (dec_deg <= survey.footprint_dec_max_deg)
    depth = (catalog["z"] <= survey.z_hard_max) & (catalog["app_mag"] <= survey.magnitude_limit)
    completeness = expit((survey.z50 - catalog["z"]) / survey.width)
    return footprint & depth & (rng.uniform(size=len(catalog["z"])) < completeness)


def _pixelate_catalog(ra: np.ndarray, dec: np.ndarray, z: np.ndarray, dz: np.ndarray, w: np.ndarray, nside: int) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    zgals = np.full((npix, max_gals), 100.0)
    dzgals = np.full((npix, max_gals), 1.0)
    wgals = np.zeros((npix, max_gals))
    offsets = np.zeros(npix, dtype=np.int32)
    for i, p in enumerate(pix):
        j = offsets[p]
        zgals[p, j] = z[i]
        dzgals[p, j] = dz[i]
        wgals[p, j] = w[i]
        offsets[p] += 1
    return {"zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals": counts}


def _draw_events_until_detected(
    rng: np.random.Generator,
    nobs: int,
    catalog: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
) -> dict[str, np.ndarray]:
    kept: list[dict[str, np.ndarray]] = []
    while sum(len(x["z"]) for x in kept) < nobs:
        ntry = max(4 * nobs, 256)
        host_idx = rng.integers(0, len(catalog["z"]), ntry)
        z = catalog["z"][host_idx]
        ra = catalog["ra"][host_idx]
        dec = catalog["dec"][host_idx]
        dl = _interp_dl(z, grids)
        m1 = _sample_powerlaw_peak_m1(rng, ntry, pop)
        q = _sample_q(rng, m1, pop)
        m2 = q * m1
        chi = _sample_chieff(rng, ntry, pop)
        snr = _network_snr(m1, m2, z, dl, rng)
        det = snr >= snr_threshold
        if np.any(det):
            kept.append({k: v[det] for k, v in dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr).items()})
    out = {k: np.concatenate([x[k] for x in kept])[:nobs] for k in kept[0]}
    return out


def _posterior_samples(rng: np.random.Generator, truth: dict[str, np.ndarray], nsamp: int) -> dict[str, np.ndarray]:
    nobs = len(truth["z"])
    arrays = {"ra": [], "dec": [], "dL": [], "m1det": [], "m2det": [], "chieff": [], "p_pe": []}
    for i in range(nobs):
        rho = truth["snr"][i]
        frac_dl = np.clip(1.8 / rho, 0.08, 0.35)
        dl = rng.lognormal(np.log(truth["dl"][i]) - 0.5 * frac_dl**2, frac_dl, nsamp)
        sigma_ang = np.deg2rad(np.clip(35.0 / rho, 1.0, 12.0))
        dra = rng.normal(0.0, sigma_ang / max(np.cos(truth["dec"][i]), 0.1), nsamp)
        ddec = rng.normal(0.0, sigma_ang, nsamp)
        arrays["ra"].append((truth["ra"][i] + dra) % (2.0 * np.pi))
        arrays["dec"].append(np.clip(truth["dec"][i] + ddec, -0.5 * np.pi, 0.5 * np.pi))
        m1det = truth["m1"][i] * (1.0 + truth["z"][i])
        m2det = truth["m2"][i] * (1.0 + truth["z"][i])
        arrays["m1det"].append(np.clip(rng.normal(m1det, 0.08 * m1det, nsamp), 2.0, None))
        arrays["m2det"].append(np.clip(rng.normal(m2det, 0.10 * m2det, nsamp), 1.0, None))
        arrays["chieff"].append(np.clip(rng.normal(truth["chi"][i], 0.08, nsamp), -1.0, 1.0))
        arrays["dL"].append(dl)
        arrays["p_pe"].append(np.ones(nsamp))
    return {k: np.concatenate(v) for k, v in arrays.items()}


def _selection_injections(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
) -> dict[str, np.ndarray | int]:
    z = _sample_uniform_comoving_z(rng, grids, ndraw)
    ra, dec = _sample_sky(rng, ndraw)
    dl = _interp_dl(z, grids)
    m1 = _sample_powerlaw_peak_m1(rng, ndraw, pop)
    q = _sample_q(rng, m1, pop)
    m2 = q * m1
    chi = _sample_chieff(rng, ndraw, pop)
    snr = _network_snr(m1, m2, z, dl, rng)
    det = snr >= snr_threshold

    pz = np.interp(z, grids["z"], grids["dvc_dz"]) / np.trapz(grids["dvc_dz"], grids["z"])
    ddldz = np.gradient(grids["dl"], grids["z"])
    jac = np.interp(z, grids["z"], ddldz) * (1.0 + z) ** 2
    p_draw = _mass_spin_pdf(m1, q, chi, pop) * pz / np.maximum(jac, 1.0e-300) / (4.0 * np.pi)
    p_draw = np.maximum(p_draw, 1.0e-300)

    return {
        "m1detsels": m1[det] * (1.0 + z[det]),
        "m2detsels": m2[det] * (1.0 + z[det]),
        "dLsels": dl[det],
        "chieffsels": chi[det],
        "rasels": ra[det],
        "decsels": dec[det],
        "p_draw": p_draw[det],
        "Ndraw": ndraw,
        "n_detected": int(det.sum()),
    }


def write_mock_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    pop = PopulationConfig()
    survey = SurveyConfig()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cosmo = _build_cosmology(args.H0, args.Om0)
    zmax = float(args.zmax)
    grids = _cosmology_grids(cosmo, zmax)

    complete = _generate_complete_catalog(rng, args.n_galaxies, grids, survey)
    observed = _apply_survey_selection(rng, complete, survey)
    zerr = survey.redshift_error_floor + survey.redshift_error_slope * (1.0 + complete["z"])
    weights = np.ones(observed.sum())
    pixelated = _pixelate_catalog(
        complete["ra"][observed], complete["dec"][observed], complete["z"][observed], zerr[observed], weights, args.nside
    )

    truth = _draw_events_until_detected(rng, args.nobs, complete, grids, pop, args.snr_threshold)
    post = _posterior_samples(rng, truth, args.nsamp)
    sel = _selection_injections(rng, args.ndraw, grids, pop, args.snr_threshold)

    metadata = {
        "seed": args.seed,
        "cosmology": {"H0": args.H0, "Om0": args.Om0},
        "population": asdict(pop),
        "survey": asdict(survey),
        "snr_threshold": args.snr_threshold,
        "pop_model_for_inference": "powerlaw+peak_shared_beta_spin",
    }

    complete_path = out / "mock_galaxy_catalog_complete.h5"
    with h5py.File(complete_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Complete isotropic, uniform-in-comoving-volume mock galaxy catalog before EM incompleteness."
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in complete.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    raw_path = out / "mock_survey_raw.h5"
    with h5py.File(raw_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Observed mock survey after footprint, magnitude, redshift, and completeness cuts."
        f.attrs["metadata_json"] = json.dumps(metadata)
        f.create_dataset("TARGET_RA", data=np.rad2deg(complete["ra"][observed]), compression="gzip", shuffle=True)
        f.create_dataset("TARGET_DEC", data=np.rad2deg(complete["dec"][observed]), compression="gzip", shuffle=True)
        f.create_dataset("Z", data=complete["z"][observed], compression="gzip", shuffle=True)
        f.create_dataset("ZERR", data=zerr[observed], compression="gzip", shuffle=True)
        f.create_dataset("WEIGHT", data=weights, compression="gzip", shuffle=True)

    pixel_path = out / f"catalog_pixelated_nside_{args.nside}.h5"
    with h5py.File(pixel_path, "w") as f:
        f.attrs["nside"] = int(args.nside)
        f.attrs["mock_data"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in pixelated.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    gw_path = out / "mock_gw_events.h5"
    with h5py.File(gw_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(args.nobs)
        f.attrs["nsamp"] = int(args.nsamp)
        f.attrs["pop_model"] = "powerlaw+peak_shared_beta_spin"
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in post.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)
        truth_group = f.create_group("truth")
        for key, val in truth.items():
            truth_group.create_dataset(key, data=val)

    sel_path = out / "mock_gw_selection.h5"
    with h5py.File(sel_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["Ndraw"] = int(sel["Ndraw"])
        f.attrs["pop_model"] = "powerlaw+peak_shared_beta_spin"
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key in ["m1detsels", "m2detsels", "dLsels", "chieffsels", "rasels", "decsels", "p_draw"]:
            f.create_dataset(key, data=sel[key], compression="gzip", shuffle=True)

    print("Mock dark-sirens data written:")
    print(f"  complete catalog : {complete_path} ({args.n_galaxies:,} galaxies)")
    print(f"  observed survey  : {raw_path} ({observed.sum():,} galaxies retained)")
    print(f"  pixelated survey : {pixel_path} (nside={args.nside})")
    print(f"  GW posteriors    : {gw_path} ({args.nobs} events x {args.nsamp} samples)")
    print(f"  GW selection     : {sel_path} ({sel['n_detected']:,}/{args.ndraw:,} detected injections)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/mock_dark_sirens", help="Output directory for HDF5 products.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-galaxies", type=int, default=50_000)
    parser.add_argument("--nobs", type=int, default=8)
    parser.add_argument("--nsamp", type=int, default=512)
    parser.add_argument("--ndraw", type=int, default=80_000)
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--zmax", type=float, default=1.5)
    parser.add_argument("--H0", type=float, default=67.74)
    parser.add_argument("--Om0", type=float, default=0.3075)
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    write_mock_data(parse_args())
