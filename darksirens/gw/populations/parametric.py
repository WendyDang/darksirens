from dataclasses import dataclass
import jax.numpy as jnp

from .base import MassComponent, PairingModel, SpinModel, ParamSpec
from .utils import sfilter_low, sfilter_high

@dataclass
class PowerLaw(MassComponent):
    alpha_spec: ParamSpec
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.alpha_spec, self.m_min_spec, self.m_max_spec, self.dm_min_spec, self.dm_max_spec]

    def _eval_unnorm(self, m, t):
        a, mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3], t[4]
        S = sfilter_low(m, mmin, dmmin) * sfilter_high(m, mmax, dmmax)
        return S * m ** (-a)

@dataclass
class BrokenPowerLaw(MassComponent):
    alpha1_spec: ParamSpec
    alpha2_spec: ParamSpec
    m_break_spec: ParamSpec
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.alpha1_spec, self.alpha2_spec, self.m_break_spec, self.m_min_spec, self.m_max_spec, self.dm_min_spec, self.dm_max_spec]

    def _eval_unnorm(self, m, t):
        a1, a2, mb, mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
        S = sfilter_low(m, mmin, dmmin) * sfilter_high(m, mmax, dmmax)
        join = mb ** (a2 - a1)
        p = jnp.where(m < mb, m ** (-a1), join * m ** (-a2))
        return S * p

@dataclass
class Gaussian(MassComponent):
    mu_spec: ParamSpec
    sigma_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.mu_spec, self.sigma_spec]

    def _eval_unnorm(self, m, t):
        mu, sig = t[0], t[1]
        return jnp.exp(-0.5 * ((m - mu) / sig) ** 2)

@dataclass
class PowerLawPairing(PairingModel):
    beta_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.beta_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        beta = t[0]
        m2 = q * m1
        p = q ** beta
        p = sfilter_low(m2, m_min, dm_min) * p
        return jnp.where(m2 < m_min, 0.0, p)

@dataclass
class GaussianPairing(PairingModel):
    mu_q_spec: ParamSpec
    sigma_q_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.mu_q_spec, self.sigma_q_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        mu, sig = t[0], t[1]
        m2 = q * m1
        p = jnp.exp(-0.5 * ((q - mu) / sig) ** 2)
        p = sfilter_low(m2, m_min, dm_min) * p
        return jnp.where(m2 < m_min, 0.0, p)

@dataclass
class TruncatedGaussianSpin(SpinModel):
    mu_chi_spec: ParamSpec
    sigma_chi_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.mu_chi_spec, self.sigma_chi_spec]

    def _eval_unnorm(self, chieff, t):
        mu, sig = t[0], t[1]
        p = jnp.exp(-0.5 * ((chieff - mu) / sig) ** 2)
        return jnp.where((chieff >= -1.0) & (chieff <= 1.0), p, 0.0)