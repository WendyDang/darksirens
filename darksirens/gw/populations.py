import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
from jax import random, jit
import jax.numpy as jnp

import numpy as np

from jax.scipy.stats import norm
from darksirens.utils.cosmology import *

# ------------------------------------------------------------
# LaTeX-ready model names
# ------------------------------------------------------------
MODEL_NAME_LATEX = {
    "powerlaw+peak":                     "PL+G",
    "brokenpowerlaw+2peaks":             "BPL+2G",
    "brokenpowerlaw+3peaks":             "BPL+3G",
    "twopowerlaws+peak":                 "2PL+G",

    "symmetric_powerlaw+peak":           "Sym PL+G",
    "symmetric_brokenpowerlaw+2peaks":   "Sym BPL+2G",
    "symmetric_brokenpowerlaw+3peaks":   "Sym BPL+3G",
    "symmetric_twopowerlaws+peak":       "Sym 2PL+G",

    "mock_data":                         r"\text{Mock}"
}


def get_model_name_latex(model_key: str) -> str:
    try:
        return MODEL_NAME_LATEX[model_key]
    except KeyError:
        return model_key  # fallback


# Grids (if you actually use them elsewhere in this module)
mass = jnp.linspace(1.0, 250.0, 2000)
mass_ratio = jnp.linspace(0.0, 1.0, 2000)
chieff_grid = jnp.linspace(-1.0, 1.0, 2000)


# ------------------------------------------------------------
# Filters and basic pieces
# ------------------------------------------------------------
def Sfilter_low(m, m_min, dm_min):
    """
    Smoothed low-mass filter.

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm, deltaMM):
        return jnp.exp(deltaMM / mm + deltaMM / (mm - deltaMM))

    S_filter = 1.0 / (f(m - m_min, dm_min) + 1.0)
    S_filter = jnp.where(m < m_min + dm_min, S_filter, 1.0)
    S_filter = jnp.where(m > m_min, S_filter, 0.0)
    return S_filter


def Sfilter_high(m, m_max, dm_max):
    """
    Smoothed high-mass filter.

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm, deltaMM):
        return jnp.exp(deltaMM / mm + deltaMM / (mm - deltaMM))

    S_filter = 1.0 / (f(m - m_max, -dm_max) + 1.0)
    S_filter = jnp.where(m > m_max - dm_max, S_filter, 1.0)
    S_filter = jnp.where(m < m_max, S_filter, 0.0)
    return S_filter


def logpchieff(chieff, mu_chieff, sigma_chieff):
    pchieff = jnp.exp(-(chieff - mu_chieff) ** 2 / (2.0 * sigma_chieff ** 2)) / jnp.sqrt(
        2.0 * jnp.pi * sigma_chieff**2
    )
    return jnp.log(pchieff)


# ------------------------------------------------------------
# Powerlaw + peak in m1, powerlaw in q
# ------------------------------------------------------------
@jit
def logpm1m2_plpeak_massratio(
    m1,
    q,
    m_min_1,
    m_max_1,
    alpha_1,
    dm_min_1,
    beta,
    mu,
    sigma,
    f,
):
    """
    log p(m1, q) for a power-law + peak in m1 and power-law in q.
    """
    # Power-law exponent convention
    alpha_pl = -alpha_1

    # --- p(m1): Power-law component ---
    norm_pl = m_max_1 ** (1.0 + alpha_pl) - m_min_1 ** (1.0 + alpha_pl)
    p_m1_pl = (1.0 + alpha_pl) * m1**alpha_pl / norm_pl

    # Mask out-of-range m1
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    # --- p(m1): Peak component ---
    p_m1_peak = jnp.exp(-0.5 * (m1 - mu) ** 2 / sigma**2) / jnp.sqrt(
        2.0 * jnp.pi * sigma**2
    )

    # Mixture with low-mass smoothing
    p_m1 = Sfilter_low(m1, m_min_1, dm_min_1) * (f * p_m1_peak + (1.0 - f) * p_m1_pl)

    # --- p(q | m1): mass-ratio power law ---
    q_min = m_min_1 / m1
    denom = 1.0 - q_min ** (1.0 + beta)
    p_q = Sfilter_low(q * m1, m_min_1, dm_min_1) * (1.0 + beta) * q**beta / denom

    # Enforce m2 >= m_min_1
    p_q = jnp.where(q * m1 < m_min_1, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q)


@jit
def logpm1m2_brokenpowerlaw_2peaks_massratio(
    m1, q,
    # broken power law
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    # peaks
    f1, f2,
    mu1, sigma1,
    mu2, sigma2,
    # mass ratio
    beta
):
    """
    log p(m1, q) for a broken power law + two Gaussian peaks + mass-ratio power law.
    """

    # ------------------------------------------------------------
    # Broken power law in m1
    # ------------------------------------------------------------
    # Low-mass smoothing
    S_low = Sfilter_low(m1, m_min, dm_min)
    # High-mass smoothing
    S_high = Sfilter_high(m1, m_max, dm_max)

    # Two power-law branches
    pl1 = m1 ** (-alpha_1)
    pl2 = m1 ** (-alpha_2)

    p_pl = jnp.where(m1 < break_mass, pl1, pl2)

    # ------------------------------------------------------------
    # Gaussian peaks
    # ------------------------------------------------------------
    peak1 = jnp.exp(-0.5 * (m1 - mu1)**2 / sigma1**2) / jnp.sqrt(2*jnp.pi*sigma1**2)
    peak2 = jnp.exp(-0.5 * (m1 - mu2)**2 / sigma2**2) / jnp.sqrt(2*jnp.pi*sigma2**2)

    # Mixture weights: remaining weight goes to power law
    f_pl = 1.0 - f1 - f2
    p_m1 = S_low * S_high * (f_pl * p_pl + f1 * peak1 + f2 * peak2)

    # ------------------------------------------------------------
    # Mass ratio distribution p(q | m1)
    # ------------------------------------------------------------
    q_min = m_min / m1
    expo = 1.0 + beta
    denom = 1.0 - q_min**expo

    p_q = (1.0 + beta) * q**beta / denom

    # enforce m2 >= m_min
    p_q = jnp.where(q * m1 < m_min, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q)


@jit
def logpm1m2_brokenpowerlaw_3peaks_massratio(
    m1, q,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu1, sigma1,
    mu2, sigma2,
    mu3, sigma3,
    beta
):
    """
    log p(m1, q) for a broken power law + 3 Gaussian peaks + mass-ratio power law.
    """

    # ------------------------------------------------------------
    # Broken power law
    # ------------------------------------------------------------
    S_low  = Sfilter_low(m1,  m_min, dm_min)
    S_high = Sfilter_high(m1, m_max, dm_max)

    pl1 = m1 ** (-alpha_1)
    pl2 = m1 ** (-alpha_2)
    p_pl = jnp.where(m1 < break_mass, pl1, pl2)

    # ------------------------------------------------------------
    # Gaussian peaks
    # ------------------------------------------------------------
    peak1 = jnp.exp(-0.5 * (m1 - mu1)**2 / sigma1**2) / jnp.sqrt(2*jnp.pi*sigma1**2)
    peak2 = jnp.exp(-0.5 * (m1 - mu2)**2 / sigma2**2) / jnp.sqrt(2*jnp.pi*sigma2**2)
    peak3 = jnp.exp(-0.5 * (m1 - mu3)**2 / sigma3**2) / jnp.sqrt(2*jnp.pi*sigma3**2)

    # Remaining weight goes to power law
    f_pl = 1.0 - f1 - f2 - f3

    p_m1 = S_low * S_high * (
        f_pl * p_pl +
        f1 * peak1 +
        f2 * peak2 +
        f3 * peak3
    )

    # ------------------------------------------------------------
    # Mass ratio distribution p(q | m1)
    # ------------------------------------------------------------
    q_min = m_min / m1
    expo = 1.0 + beta
    denom = 1.0 - q_min**expo

    p_q = (1.0 + beta) * q**beta / denom
    p_q = jnp.where(q * m1 < m_min, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q)


@jit
def logpm1m2_twopowerlaws_peak_massratio(
    m1, q,
    alpha_1, alpha_2,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu3, sigma3,
    beta
):
    """
    log p(m1, q) for two power laws + one high-mass Gaussian peak.
    """

    # ------------------------------------------------------------
    # Smoothing filters
    # ------------------------------------------------------------
    S_low  = Sfilter_low(m1,  m_min, dm_min)
    S_high = Sfilter_high(m1, m_max, dm_max)

    # ------------------------------------------------------------
    # Two power laws (ordered in mass)
    # ------------------------------------------------------------
    pl1 = m1 ** (-alpha_1)
    pl2 = m1 ** (-alpha_2)

    # enforce ordering: pl1 applies below pl2
    p_pl = jnp.where(m1 < (m_min + m_max) / 2.0, pl1, pl2)

    # ------------------------------------------------------------
    # High-mass Gaussian peak
    # ------------------------------------------------------------
    peak3 = jnp.exp(-0.5 * (m1 - mu3)**2 / sigma3**2) / jnp.sqrt(
        2 * jnp.pi * sigma3**2
    )

    # ------------------------------------------------------------
    # Mixture
    # ------------------------------------------------------------
    f_pl = 1.0 - f1 - f2 - f3
    p_m1 = S_low * S_high * (
        f_pl * p_pl +
        f1 * pl1 +
        f2 * pl2 +
        f3 * peak3
    )

    # ------------------------------------------------------------
    # Mass ratio distribution p(q | m1)
    # ------------------------------------------------------------
    q_min = m_min / m1
    expo = 1.0 + beta
    denom = 1.0 - q_min**expo

    p_q = (1.0 + beta) * q**beta / denom
    p_q = jnp.where(q * m1 < m_min, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q)


@jit
def pm_plpeak(m, m_min, m_max, alpha, dm_min, mu, sigma, f):
    alpha_pl = -alpha

    # Power-law
    norm_pl = m_max**(1 + alpha_pl) - m_min**(1 + alpha_pl)
    p_pl = (1 + alpha_pl) * m**alpha_pl / norm_pl
    p_pl = jnp.where((m < m_min) | (m > m_max), 0.0, p_pl)

    # Peak
    p_peak = jnp.exp(-0.5 * (m - mu)**2 / sigma**2) / jnp.sqrt(
        2 * jnp.pi * sigma**2
    )

    # Mixture with smoothing
    return Sfilter_low(m, m_min, dm_min) * (f * p_peak + (1 - f) * p_pl)


@jit
def logpm1m2_symmetric_plpeak_massratio(
    m1, m2,
    m_min, m_max,
    alpha, dm_min,
    mu, sigma, f,
    beta
):
    """
    Symmetric model:
        p(m1, m2) = p(m1) * p(m2) * (m2/m1)^beta
    where p(m) is powerlaw+peak.
    """

    # p(m1) and p(m2)
    p1 = pm_plpeak(m1, m_min, m_max, alpha, dm_min, mu, sigma, f)
    p2 = pm_plpeak(m2, m_min, m_max, alpha, dm_min, mu, sigma, f)

    # pairing kernel q^beta
    q = m2 / m1
    p_pair = q**beta

    # enforce m2 >= m_min
    p_pair = jnp.where(m2 < m_min, 0.0, p_pair)

    return jnp.log(p1) + jnp.log(p2) + jnp.log(p_pair)


@jit
def pm_brokenpowerlaw_2peaks(
    m,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2,
    mu1, sigma1,
    mu2, sigma2
):
    # Smoothing
    S_low  = Sfilter_low(m,  m_min, dm_min)
    S_high = Sfilter_high(m, m_max, dm_max)

    # Broken power law
    pl1 = m**(-alpha_1)
    pl2 = m**(-alpha_2)
    p_pl = jnp.where(m < break_mass, pl1, pl2)

    # Gaussian peaks
    peak1 = jnp.exp(-0.5 * (m - mu1)**2 / sigma1**2) / jnp.sqrt(2*jnp.pi*sigma1**2)
    peak2 = jnp.exp(-0.5 * (m - mu2)**2 / sigma2**2) / jnp.sqrt(2*jnp.pi*sigma2**2)

    # Mixture
    f_pl = 1.0 - f1 - f2
    p_m = S_low * S_high * (f_pl * p_pl + f1 * peak1 + f2 * peak2)

    return p_m


@jit
def logpm1m2_symmetric_brokenpowerlaw_2peaks_massratio(
    m1, m2,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2,
    mu1, sigma1,
    mu2, sigma2,
    beta
):
    # p(m1) and p(m2)
    p1 = pm_brokenpowerlaw_2peaks(
        m1,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2,
        mu1, sigma1,
        mu2, sigma2
    )

    p2 = pm_brokenpowerlaw_2peaks(
        m2,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2,
        mu1, sigma1,
        mu2, sigma2
    )

    # Pairing kernel q^beta
    q = m2 / m1
    p_pair = q**beta
    p_pair = jnp.where(m2 < m_min, 0.0, p_pair)

    return jnp.log(p1) + jnp.log(p2) + jnp.log(p_pair)


@jit
def pm_brokenpowerlaw_3peaks(
    m,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu1, sigma1,
    mu2, sigma2,
    mu3, sigma3,
):
    S_low  = Sfilter_low(m,  m_min, dm_min)
    S_high = Sfilter_high(m, m_max, dm_max)

    pl1 = m**(-alpha_1)
    pl2 = m**(-alpha_2)
    p_pl = jnp.where(m < break_mass, pl1, pl2)

    peak1 = jnp.exp(-0.5 * (m - mu1)**2 / sigma1**2) / jnp.sqrt(2*jnp.pi*sigma1**2)
    peak2 = jnp.exp(-0.5 * (m - mu2)**2 / sigma2**2) / jnp.sqrt(2*jnp.pi*sigma2**2)
    peak3 = jnp.exp(-0.5 * (m - mu3)**2 / sigma3**2) / jnp.sqrt(2*jnp.pi*sigma3**2)

    f_pl = 1.0 - f1 - f2 - f3

    p_m = S_low * S_high * (
        f_pl * p_pl +
        f1 * peak1 +
        f2 * peak2 +
        f3 * peak3
    )
    return p_m


@jit
def logpm1m2_symmetric_brokenpowerlaw_3peaks_massratio(
    m1, m2,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu1, sigma1,
    mu2, sigma2,
    mu3, sigma3,
    beta
):
    p1 = pm_brokenpowerlaw_3peaks(
        m1,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
    )
    p2 = pm_brokenpowerlaw_3peaks(
        m2,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
    )

    q = m2 / m1
    p_pair = q**beta
    p_pair = jnp.where(m2 < m_min, 0.0, p_pair)

    return jnp.log(p1) + jnp.log(p2) + jnp.log(p_pair)


@jit
def pm_symmetric_twopowerlaws_peak(
    m,
    alpha_1, alpha_2,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu3, sigma3,
    beta
):
    S_low  = Sfilter_low(m,  m_min, dm_min)
    S_high = Sfilter_high(m, m_max, dm_max)

    pl1 = m ** (-alpha_1)
    pl2 = m ** (-alpha_2)

    # enforce ordering: pl1 applies below pl2
    p_pl = jnp.where(m < (m_min + m_max) / 2.0, pl1, pl2)

    peak3 = jnp.exp(-0.5 * (m - mu3)**2 / sigma3**2) / jnp.sqrt(
        2 * jnp.pi * sigma3**2
    )

    f_pl = 1.0 - f1 - f2 - f3
    p_m = S_low * S_high * (
        f_pl * p_pl +
        f1 * pl1 +
        f2 * pl2 +
        f3 * peak3
    )

    return p_m

@jit
def logpm1m2_symmetric_symmetric_twopowerlaws_peak_massratio(
    m1, m2,
    alpha_1, alpha_2,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2, f3,
    mu3, sigma3,
    beta
):
    p1 = pm_symmetric_twopowerlaws_peak(
        m1,
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta
    )
    p2 = pm_symmetric_twopowerlaws_peak(
        m2,
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta
    )

    q = m2 / m1
    p_pair = q**beta
    p_pair = jnp.where(m2 < m_min, 0.0, p_pair)

    return jnp.log(p1) + jnp.log(p2) + jnp.log(p_pair)


# ------------------------------------------------------------
# Redshift evolution
# ------------------------------------------------------------
@jit
def logpowerlaw_redshift(z, gamma):
    return gamma * jnp.log1p(z)


@jit
def log_p_pop_powerlaw_peak(
    m1,
    q,
    z,
    alpha_1,
    beta,
    m_min_1,
    m_max_1,
    dm_min_1,
    mu,
    sigma,
    f,
    gamma,
):
    """
    Full population log-density for the powerlaw+peak model:
        p(m1, q, z) ∝ p(m1, q) * p(z) * p(spins)
    """
    log_dNdm1dq = logpm1m2_plpeak_massratio(
        m1,
        q,
        m_min_1,
        m_max_1,
        alpha_1,
        dm_min_1,
        beta,
        mu,
        sigma,
        f,
    )

    # Placeholder spin prior (if you later include chi_eff, etc.)
    log_p_sz = jnp.log(0.25)  # e.g. 1/2 per spin dimension

    log_p_z = logpowerlaw_redshift(z, gamma)
    return log_p_sz + log_dNdm1dq + log_p_z

# ------------------------------------------------------------
# log_p_pops
# ------------------------------------------------------------
@jit
def log_p_pop_brokenpowerlaw_2peaks(
    m1, q, z,
    alpha_1, alpha_2, break_mass,
    m_min, dm_min,
    m_max, dm_max,
    f1, f2,
    mu1, sigma1,
    mu2, sigma2,
    beta,
    gamma
):
    """
    Full population model:
        p(m1, q, z) = p(m1, q) * p(z) * p(spins)
    """

    log_m1q = logpm1m2_brokenpowerlaw_2peaks_massratio(
        m1, q,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2,
        mu1, sigma1,
        mu2, sigma2,
        beta
    )

    # spin prior placeholder
    log_p_spin = jnp.log(0.25)

    # redshift evolution
    log_p_z = gamma * jnp.log1p(z)

    return log_m1q + log_p_spin + log_p_z


@jit
def log_p_pop_brokenpowerlaw_3peaks(
        m1, q, z, 
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
        beta,
        gamma):

    log_m1q = logpm1m2_brokenpowerlaw_3peaks_massratio(
        m1, q,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1q + log_p_z + log_p_spin


@jit
def log_p_pop_twopowerlaws_peak(
        m1, q, z, 
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta,
        gamma):

    log_m1q = logpm1m2_twopowerlaws_peak_massratio(
        m1, q,
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1q + log_p_z + log_p_spin


@jit
def log_p_pop_symmetric_plpeak(
        m1, q, z, 
        alpha,
        beta,
        m_min,
        m_max,
        dm_min,
        mu,
        sigma,
        f,
        gamma):

    # convert q → m2
    m2 = q * m1

    log_m1m2 = logpm1m2_symmetric_plpeak_massratio(
        m1, m2,
        m_min, m_max,
        alpha, dm_min,
        mu, sigma, f,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1m2 + log_p_z + log_p_spin


@jit
def log_p_pop_symmetric_brokenpowerlaw_2peaks(
        m1, q, z, 
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2,
        mu1, sigma1,
        mu2, sigma2,
        beta,
        gamma):

    m2 = q * m1

    log_m1m2 = logpm1m2_symmetric_brokenpowerlaw_2peaks_massratio(
        m1, m2,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2,
        mu1, sigma1,
        mu2, sigma2,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1m2 + log_p_z + log_p_spin


@jit
def log_p_pop_symmetric_brokenpowerlaw_3peaks(
        m1, q, z,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
        beta,
        gamma):

    m2 = q * m1

    log_m1m2 = logpm1m2_symmetric_brokenpowerlaw_3peaks_massratio(
        m1, m2,
        alpha_1, alpha_2, break_mass,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu1, sigma1,
        mu2, sigma2,
        mu3, sigma3,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1m2 + log_p_z + log_p_spin


@jit
def log_p_pop_symmetric_twopowerlaws_peak(
        m1, q, z, 
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta,
        gamma):

    m2 = q * m1

    log_m1m2 = logpm1m2_symmetric_symmetric_twopowerlaws_peak_massratio(
        m1, m2,
        alpha_1, alpha_2,
        m_min, dm_min,
        m_max, dm_max,
        f1, f2, f3,
        mu3, sigma3,
        beta
    )

    log_p_z = gamma * jnp.log1p(z)
    log_p_spin = jnp.log(0.25)

    return log_m1m2 + log_p_z + log_p_spin


# ------------------------------------------------------------
# Model parser
# ------------------------------------------------------------
def pop_model_parser(pop_model):
    if pop_model == "powerlaw+peak":
        return log_p_pop_powerlaw_peak
    elif pop_model == "brokenpowerlaw+2peaks":
        return log_p_pop_brokenpowerlaw_2peaks
    elif pop_model == "brokenpowerlaw+3peaks":
        return log_p_pop_brokenpowerlaw_3peaks
    elif pop_model == "twopowerlaws+peak":
        return log_p_pop_twopowerlaws_peak
    elif pop_model == "symmetric_powerlaw+peak":
        return log_p_pop_symmetric_plpeak
    elif pop_model == "symmetric_brokenpowerlaw+2peaks":
        return log_p_pop_symmetric_brokenpowerlaw_2peaks
    elif pop_model == "symmetric_brokenpowerlaw+3peaks":
        return log_p_pop_symmetric_brokenpowerlaw_3peaks
    elif pop_model == "symmetric_twopowerlaws+peak":
        return log_p_pop_symmetric_twopowerlaws_peak
    elif pop_model == "mock_data":
        return log_p_pop_mock_data
    else:
        raise ValueError(f"Unknown population model: {pop_model}")


# ------------------------------------------------------------
# Prior parser
# ------------------------------------------------------------
def pop_model_prior_parser(pop_model):
    if pop_model == "powerlaw+peak":
        model_name = get_model_name_latex(pop_model)

        # Only keep what you actually use for bounds
        m_min_1_low, m_min_1_high = 2.0, 10.0
        m_max_1_low, m_max_1_high = 50.0, 100.0

        alpha_1_low, alpha_1_high = -4.0, 6.0
        dm_min_1_low, dm_min_1_high = 0.0, 10.0

        beta_low, beta_high = -2.0, 7.0

        mu_low, mu_high = 20.0, 50.0
        sigma_low, sigma_high = 1.0, 10.0

        f_low, f_high = 0.0, 0.15

        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low,
            beta_low,
            m_min_1_low,
            m_max_1_low,
            dm_min_1_low,
            mu_low,
            sigma_low,
            f_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high,
            beta_high,
            m_min_1_high,
            m_max_1_high,
            dm_min_1_high,
            mu_high,
            sigma_high,
            f_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha$",
            r"$\beta$",
            r"$m_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm min}$",
            r"$\mu$",
            r"$\sigma$",
            r"$f$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name

    elif pop_model == "brokenpowerlaw+2peaks":
        model_name = get_model_name_latex(pop_model)

        # -----------------------------
        # Parameter bounds (cleaned)
        # -----------------------------
        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        break_mass_lo, break_mass_hi = 20.0, 50.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 100.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 100.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0

        mu1_low, mu1_high = 5.0, 20.0
        sigma1_low, sigma1_high = 1.0, 10.0

        mu2_low, mu2_high = 25.0, 40.0
        sigma2_low, sigma2_high = 1.0, 10.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        # -----------------------------
        # Final theta ordering
        # -----------------------------
        lower_bound = [
            alpha_1_low,
            alpha_2_low,
            break_mass_lo,
            m_min_low,
            dm_min_low,
            m_max_low,
            dm_max_low,
            f1_low,
            f2_low,
            mu1_low,
            sigma1_low,
            mu2_low,
            sigma2_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high,
            alpha_2_high,
            break_mass_hi,
            m_min_high,
            dm_min_high,
            m_max_high,
            dm_max_high,
            f1_high,
            f2_high,
            mu1_high,
            sigma1_high,
            mu2_high,
            sigma2_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm break}$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$\mu_1$",
            r"$\sigma_1$",
            r"$\mu_2$",
            r"$\sigma_2$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name
    
    elif pop_model == "brokenpowerlaw+3peaks":
        model_name = get_model_name_latex(pop_model)

        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        break_mass_lo, break_mass_hi = 20.0, 50.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 100.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 100.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0
        f3_low, f3_high = 0.0, 1.0

        mu1_low, mu1_high = 5.0, 20.0
        sigma1_low, sigma1_high = 1.0, 10.0

        mu2_low, mu2_high = 25.0, 40.0
        sigma2_low, sigma2_high = 1.0, 10.0

        # NEW: third peak
        mu3_low, mu3_high = 50.0, 100.0
        sigma3_low, sigma3_high = 1.0, 20.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low, alpha_2_low,
            break_mass_lo,
            m_min_low, dm_min_low,
            m_max_low, dm_max_low,
            f1_low, f2_low, f3_low,
            mu1_low, sigma1_low,
            mu2_low, sigma2_low,
            mu3_low, sigma3_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high, alpha_2_high,
            break_mass_hi,
            m_min_high, dm_min_high,
            m_max_high, dm_max_high,
            f1_high, f2_high, f3_high,
            mu1_high, sigma1_high,
            mu2_high, sigma2_high,
            mu3_high, sigma3_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm break}$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$f_3$",
            r"$\mu_1$",
            r"$\sigma_1$",
            r"$\mu_2$",
            r"$\sigma_2$",
            r"$\mu_3$",
            r"$\sigma_3$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name

    elif pop_model == "twopowerlaws+peak":
        model_name = get_model_name_latex(pop_model)

        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 50.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 50.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0
        f3_low, f3_high = 0.0, 1.0

        mu3_low, mu3_high = 50.0, 100.0
        sigma3_low, sigma3_high = 1.0, 20.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low, alpha_2_low,
            m_min_low, dm_min_low,
            m_max_low, dm_max_low,
            f1_low, f2_low, f3_low,
            mu3_low, sigma3_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high, alpha_2_high,
            m_min_high, dm_min_high,
            m_max_high, dm_max_high,
            f1_high, f2_high, f3_high,
            mu3_high, sigma3_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$f_3$",
            r"$\mu_3$",
            r"$\sigma_3$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name
    
    elif pop_model == "symmetric_powerlaw+peak":
        model_name = get_model_name_latex(pop_model)

        alpha_low, alpha_high = -4.0, 6.0
        beta_low, beta_high = -2.0, 7.0
        m_min_low, m_min_high = 2.0, 10.0
        m_max_low, m_max_high = 50.0, 100.0
        dm_min_low, dm_min_high = 0.0, 10.0
        mu_low, mu_high = 20.0, 50.0
        sigma_low, sigma_high = 1.0, 10.0
        f_low, f_high = 0.0, 0.15
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_low, beta_low,
            m_min_low, m_max_low, dm_min_low,
            mu_low, sigma_low, f_low,
            gamma_low
        ]

        upper_bound = [
            alpha_high, beta_high,
            m_min_high, m_max_high, dm_min_high,
            mu_high, sigma_high, f_high,
            gamma_high
        ]

        labels = [
            r"$\alpha$",
            r"$\beta$",
            r"$m_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm min}$",
            r"$\mu$",
            r"$\sigma$",
            r"$f$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name

    elif pop_model == "symmetric_brokenpowerlaw+2peaks":
        model_name = get_model_name_latex(pop_model)

        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        break_mass_lo, break_mass_hi = 20.0, 50.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 100.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 100.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0

        mu1_low, mu1_high = 5.0, 20.0
        sigma1_low, sigma1_high = 1.0, 10.0

        mu2_low, mu2_high = 25.0, 40.0
        sigma2_low, sigma2_high = 1.0, 10.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low, alpha_2_low,
            break_mass_lo,
            m_min_low, dm_min_low,
            m_max_low, dm_max_low,
            f1_low, f2_low,
            mu1_low, sigma1_low,
            mu2_low, sigma2_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high, alpha_2_high,
            break_mass_hi,
            m_min_high, dm_min_high,
            m_max_high, dm_max_high,
            f1_high, f2_high,
            mu1_high, sigma1_high,
            mu2_high, sigma2_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm break}$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$\mu_1$",
            r"$\sigma_1$",
            r"$\mu_2$",
            r"$\sigma_2$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name
    
    elif pop_model == "symmetric_brokenpowerlaw+3peaks":
        model_name = get_model_name_latex(pop_model)

        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        break_mass_lo, break_mass_hi = 20.0, 50.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 100.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 100.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0
        f3_low, f3_high = 0.0, 1.0

        mu1_low, mu1_high = 5.0, 20.0
        sigma1_low, sigma1_high = 1.0, 10.0

        mu2_low, mu2_high = 25.0, 40.0
        sigma2_low, sigma2_high = 1.0, 10.0

        mu3_low, mu3_high = 50.0, 100.0
        sigma3_low, sigma3_high = 1.0, 20.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low, alpha_2_low,
            break_mass_lo,
            m_min_low, dm_min_low,
            m_max_low, dm_max_low,
            f1_low, f2_low, f3_low,
            mu1_low, sigma1_low,
            mu2_low, sigma2_low,
            mu3_low, sigma3_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high, alpha_2_high,
            break_mass_hi,
            m_min_high, dm_min_high,
            m_max_high, dm_max_high,
            f1_high, f2_high, f3_high,
            mu1_high, sigma1_high,
            mu2_high, sigma2_high,
            mu3_high, sigma3_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm break}$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$f_3$",
            r"$\mu_1$",
            r"$\sigma_1$",
            r"$\mu_2$",
            r"$\sigma_2$",
            r"$\mu_3$",
            r"$\sigma_3$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name
    
    elif pop_model == "symmetric_twopowerlaws+peak":
        model_name = get_model_name_latex(pop_model)

        alpha_1_low, alpha_1_high = 0.0, 6.0
        alpha_2_low, alpha_2_high = 0.0, 6.0

        m_min_low, m_min_high = 2.0, 10.0
        dm_min_low, dm_min_high = 1.0, 50.0

        m_max_low, m_max_high = 40.0, 200.0
        dm_max_low, dm_max_high = 1.0, 50.0

        f1_low, f1_high = 0.0, 1.0
        f2_low, f2_high = 0.0, 1.0
        f3_low, f3_high = 0.0, 1.0

        mu3_low, mu3_high = 50.0, 100.0
        sigma3_low, sigma3_high = 1.0, 20.0

        beta_low, beta_high = 0.0, 6.0
        gamma_low, gamma_high = -10.0, 10.0

        lower_bound = [
            alpha_1_low, alpha_2_low,
            m_min_low, dm_min_low,
            m_max_low, dm_max_low,
            f1_low, f2_low, f3_low,
            mu3_low, sigma3_low,
            beta_low,
            gamma_low,
        ]

        upper_bound = [
            alpha_1_high, alpha_2_high,
            m_min_high, dm_min_high,
            m_max_high, dm_max_high,
            f1_high, f2_high, f3_high,
            mu3_high, sigma3_high,
            beta_high,
            gamma_high,
        ]

        labels = [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$m_{\rm min}$",
            r"$dm_{\rm min}$",
            r"$m_{\rm max}$",
            r"$dm_{\rm max}$",
            r"$f_1$",
            r"$f_2$",
            r"$f_3$",
            r"$\mu_3$",
            r"$\sigma_3$",
            r"$\beta$",
            r"$\gamma$",
        ]

        return lower_bound, upper_bound, labels, model_name

    elif pop_model == "mock_data":
        # If you have a dedicated prior for mock_data, define it here
        raise NotImplementedError("Prior for 'mock_data' not implemented yet.")

    else:
        raise ValueError(f"Unknown population model: {pop_model}")
