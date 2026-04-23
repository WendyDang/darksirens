from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp
from jax import jit


# ======================================================================
# 0. Fixed grids
# ======================================================================

M_LO: float = 1.0
M_HI: float = 200.0
N_MASS: int = 500
N_Q: int = 200
N_CHI: int = 200

MASS_GRID = jnp.linspace(M_LO, M_HI, N_MASS)
Q_GRID    = jnp.linspace(0.0, 1.0, N_Q)
CHI_GRID  = jnp.linspace(-1.0, 1.0, N_CHI)

M1_MESH, Q_MESH = jnp.meshgrid(MASS_GRID, Q_GRID, indexing="ij")


# ======================================================================
# 1. Prior specification
# ======================================================================

@dataclass(frozen=True)
class ParamSpec:
    label: str
    low: float
    high: float


def pack_specs(*specs: ParamSpec):
    return (
        [s.low for s in specs],
        [s.high for s in specs],
        [s.label for s in specs],
    )


# ======================================================================
# 2. Smooth edge filters
# ======================================================================

@jit
def sfilter_low(m, m_min, dm):
    delta = m - m_min
    safe_d = jnp.where(delta > 0, delta, 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / (safe_d - safe_dm),
        -500.0, 500.0
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m <= m_min, 0.0, S)
    S = jnp.where(m >= m_min + dm, 1.0, S)
    return S


@jit
def sfilter_high(m, m_max, dm):
    delta = m_max - m
    safe_d = jnp.where(delta > 0, delta, 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / (safe_d - safe_dm),
        -500.0, 500.0
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m >= m_max, 0.0, S)
    S = jnp.where(m <= m_max - dm, 1.0, S)
    return S


# ======================================================================
# 3. Mass components (normalized on MASS_GRID)
# ======================================================================

class MassComponent(ABC):

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        ...

    @property
    def n_params(self) -> int:
        return len(self.param_specs)

    @abstractmethod
    def _eval_unnorm(self, m, theta):
        ...

    def __call__(self, m, theta):
        p = self._eval_unnorm(m, theta)
        p_grid = self._eval_unnorm(MASS_GRID, theta)
        n = jnp.trapezoid(p_grid, MASS_GRID)
        return p / jnp.where(n > 0, n, 1.0)


@dataclass
class PowerLaw(MassComponent):
    alpha_spec: ParamSpec
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.alpha_spec,
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
        ]

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
        return [
            self.alpha1_spec,
            self.alpha2_spec,
            self.m_break_spec,
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
        ]

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


# ======================================================================
# 4. Pairing models (normalized on Q_GRID | m1)
# ======================================================================

class PairingModel(ABC):

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        ...

    @property
    def n_params(self):
        return len(self.param_specs)

    @abstractmethod
    def _eval_unnorm(self, m1, q, m_min, dm_min, theta):
        ...

    def __call__(self, m1, q, m_min, dm_min, theta):
        p = self._eval_unnorm(m1, q, m_min, dm_min, theta)

        m1_arr = jnp.atleast_1d(m1)
        m1_expanded = jnp.expand_dims(m1_arr, axis=-1)
        
        p_grid = self._eval_unnorm(
            m1_expanded,
            Q_GRID,
            m_min,
            dm_min,
            theta,
        )
        
        n = jnp.trapezoid(p_grid, Q_GRID, axis=-1)
        n = n.reshape(jnp.shape(m1))

        return p / jnp.where(n > 0, n, 1.0)


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


# ======================================================================
# 4.5 Spin models (normalized on CHI_GRID)
# ======================================================================

class SpinModel(ABC):

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        ...

    @property
    def n_params(self):
        return len(self.param_specs)

    @abstractmethod
    def _eval_unnorm(self, chieff, theta):
        ...

    def __call__(self, chieff, theta):
        p = self._eval_unnorm(chieff, theta)
        p_grid = self._eval_unnorm(CHI_GRID, theta)
        n = jnp.trapezoid(p_grid, CHI_GRID)
        return p / jnp.where(n > 0, n, 1.0)


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


# ======================================================================
# 5. Unified MixtureModel 
# ======================================================================

@dataclass
class MixtureModel:
    mass_components: list[MassComponent]
    pairing_components: list[PairingModel]
    spin_components: list[SpinModel]

    def __post_init__(self):
        self.k = len(self.mass_components)
        self.shared_pairing = len(self.pairing_components) == 1
        self.shared_spin = len(self.spin_components) == 1

        if not self.shared_pairing and len(self.pairing_components) != self.k:
            raise ValueError(f"Expected {self.k} or 1 pairing components, got {len(self.pairing_components)}")
        if not self.shared_spin and len(self.spin_components) != self.k:
            raise ValueError(f"Expected {self.k} or 1 spin components, got {len(self.spin_components)}")

    @property
    def n_weight_params(self):
        return max(self.k - 1, 0)

    @property
    def param_specs(self):
        # Enforce global ordering: Weights -> Masses -> Pairings -> Spins
        specs = [
            ParamSpec(rf"$f_{i+1}$", 0.0, 1.0)
            for i in range(self.n_weight_params)
        ]
        for c in self.mass_components:
            specs.extend(c.param_specs)
        for c in self.pairing_components:
            specs.extend(c.param_specs)
        for c in self.spin_components:
            specs.extend(c.param_specs)
        return specs

    @property
    def n_params(self):
        return len(self.param_specs)

    def __call__(self, m1, q, chieff, theta):
        n_w = self.n_weight_params
        w_raw = theta[:n_w]
        
        if n_w > 0:
            w_last = 1.0 - jnp.sum(w_raw)
            w = jnp.concatenate([w_raw, jnp.atleast_1d(w_last)])
        else:
            w = jnp.array([1.0])

        idx = n_w

        # Extract parameter blocks
        tm_list = []
        for c in self.mass_components:
            tm_list.append(theta[idx : idx + c.n_params])
            idx += c.n_params

        tp_list = []
        for c in self.pairing_components:
            tp_list.append(theta[idx : idx + c.n_params])
            idx += c.n_params

        ts_list = []
        for c in self.spin_components:
            ts_list.append(theta[idx : idx + c.n_params])
            idx += c.n_params

        out = 0.0
        for i in range(self.k):
            c_m = self.mass_components[i]
            c_p = self.pairing_components[0] if self.shared_pairing else self.pairing_components[i]
            c_s = self.spin_components[0] if self.shared_spin else self.spin_components[i]

            tm = tm_list[i]
            tp = tp_list[0] if self.shared_pairing else tp_list[i]
            ts = ts_list[0] if self.shared_spin else ts_list[i]

            mmin = M_LO
            dmmin = 0.01

            if hasattr(c_m, "m_min_spec"):
                mmin_idx = c_m.param_specs.index(c_m.m_min_spec)
                mmin = tm[mmin_idx]
                
            if hasattr(c_m, "dm_min_spec"):
                dmmin_idx = c_m.param_specs.index(c_m.dm_min_spec)
                dmmin = tm[dmmin_idx]

            out = out + w[i] * c_m(m1, tm) * c_p(m1, q, mmin, dmmin, tp) * c_s(chieff, ts)

        return out


# ======================================================================
# 6. PopulationModel
# ======================================================================

@dataclass
class PopulationModel:
    mixture: MixtureModel

    @property
    def param_specs(self):
        # Gamma strictly appears at the very end
        return [*self.mixture.param_specs, ParamSpec(r"$\gamma$", -10.0, 10.0)]

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_p_pop(self, m1, q, z, chieff, theta):
        tm = theta[:-1]
        gamma = theta[-1]
        p = self.mixture(m1, q, chieff, tm)
        return jnp.where(p > 0, jnp.log(p), -1e10) + gamma * jnp.log1p(z)


# ======================================================================
# 7. Factories
# ======================================================================

def _pl(
    alpha_label=r"$\alpha$",
    mmin_label=r"$m_{\min}$",
    mmax_label=r"$m_{\max}$",
    dmmin_label=r"$dm_{\min}$",
    dmmax_label=r"$dm_{\max}$",
    alpha_lo=-4.0, alpha_hi=6.0,
    mmin_lo=2.0,  mmin_hi=10.0,
    mmax_lo=50.0, mmax_hi=100.0,
    dmmin_lo=0.01, dmmin_hi=10.0,
    dmmax_lo=0.01, dmmax_hi=20.0,
):
    return PowerLaw(
        alpha_spec = ParamSpec(alpha_label, alpha_lo, alpha_hi),
        m_min_spec = ParamSpec(mmin_label,  mmin_lo,  mmin_hi),
        m_max_spec = ParamSpec(mmax_label,  mmax_lo,  mmax_hi),
        dm_min_spec= ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        dm_max_spec= ParamSpec(dmmax_label, dmmax_lo, dmmax_hi),
    )

def _bpl(
    alpha1_label=r"$\alpha_1$",
    alpha2_label=r"$\alpha_2$",
    break_label=r"$m_{\rm break}$",
    mmin_label=r"$m_{\min}$",
    mmax_label=r"$m_{\max}$",
    dmmin_label=r"$dm_{\min}$",
    dmmax_label=r"$dm_{\max}$",
    a1_lo=0.0, a1_hi=6.0,
    a2_lo=0.0, a2_hi=6.0,
    brk_lo=20.0, brk_hi=50.0,
    mmin_lo=2.0,  mmin_hi=10.0,
    mmax_lo=40.0, mmax_hi=200.0,
    dmmin_lo=0.01, dmmin_hi=100.0,
    dmmax_lo=0.01, dmmax_hi=100.0,
):
    return BrokenPowerLaw(
        alpha1_spec = ParamSpec(alpha1_label, a1_lo, a1_hi),
        alpha2_spec = ParamSpec(alpha2_label, a2_lo, a2_hi),
        m_break_spec= ParamSpec(break_label, brk_lo, brk_hi),
        m_min_spec  = ParamSpec(mmin_label,  mmin_lo,  mmin_hi),
        m_max_spec  = ParamSpec(mmax_label,  mmax_lo,  mmax_hi),
        dm_min_spec = ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        dm_max_spec = ParamSpec(dmmax_label, dmmax_lo, dmmax_hi),
    )

def _gauss(
    mu_lo, mu_hi,
    sig_lo=1.0, sig_hi=10.0,
    mu_label=r"$\mu$",
    sig_label=r"$\sigma$",
):
    return Gaussian(
        mu_spec   = ParamSpec(mu_label,  mu_lo,  mu_hi),
        sigma_spec= ParamSpec(sig_label, sig_lo, sig_hi),
    )

def _plpairing(beta_label=r"$\beta$", beta_lo=-2.0, beta_hi=7.0):
    return PowerLawPairing(
        beta_spec = ParamSpec(beta_label, beta_lo, beta_hi)
    )

def _spin(
    mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$", 
    mu_lo=-1.0, mu_hi=1.0, sig_lo=0.01, sig_hi=1.0
):
    return TruncatedGaussianSpin(
        mu_chi_spec=ParamSpec(mu_label, mu_lo, mu_hi),
        sigma_chi_spec=ParamSpec(sig_label, sig_lo, sig_hi)
    )


# ----------------------------------------------------------------------
# Mix Factories
# ----------------------------------------------------------------------

def _mixture_plpeak(shared_beta=False, shared_spin=False):
    masses = [_pl(), _gauss(20, 50)]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_{\rm PL}$"), _plpairing(beta_label=r"$\beta_{\rm G}$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,\rm PL}$", sig_label=r"$\sigma_{\chi,\rm PL}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G}$",  sig_label=r"$\sigma_{\chi,\rm G}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_bpl2peaks(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20,  mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_{\rm BPL}$"),
                _plpairing(beta_label=r"$\beta_{\rm G1}$"),
                _plpairing(beta_label=r"$\beta_{\rm G2}$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G1}$",  sig_label=r"$\sigma_{\chi,\rm G1}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G2}$",  sig_label=r"$\sigma_{\chi,\rm G2}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_bpl3peaks(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20,   mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40,  mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$"),
        _gauss(50, 100, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", sig_hi=20)
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_{\rm BPL}$"),
                _plpairing(beta_label=r"$\beta_{\rm G1}$"),
                _plpairing(beta_label=r"$\beta_{\rm G2}$"),
                _plpairing(beta_label=r"$\beta_{\rm G3}$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G1}$",  sig_label=r"$\sigma_{\chi,\rm G1}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G2}$",  sig_label=r"$\sigma_{\chi,\rm G2}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G3}$",  sig_label=r"$\sigma_{\chi,\rm G3}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_2pl1peak(shared_beta=False, shared_spin=False):
    masses = [
        _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50),
        _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100),
        _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$")
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_1$"), _plpairing(beta_label=r"$\beta_2$"), _plpairing(beta_label=r"$\beta_3$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
             _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
             _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_2pl2peaks(shared_beta=False, shared_spin=False):
    masses = [
        _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50),
        _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100),
        _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$"),
        _gauss(20, 50, sig_hi=15, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$")
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_1$"), _plpairing(beta_label=r"$\beta_2$"), _plpairing(beta_label=r"$\beta_3$"), _plpairing(beta_label=r"$\beta_4$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
             _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
             _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$"),
             _spin(mu_label=r"$\mu_{\chi,4}$", sig_label=r"$\sigma_{\chi,4}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_2pl3peaks(shared_beta=False, shared_spin=False):
    masses = [
        _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50),
        _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100),
        _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$"),
        _gauss(20, 50, sig_hi=10, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$"),
        _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_5$", sig_label=r"$\sigma_5$")
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_1$"), _plpairing(beta_label=r"$\beta_2$"), _plpairing(beta_label=r"$\beta_3$"), _plpairing(beta_label=r"$\beta_4$"), _plpairing(beta_label=r"$\beta_5$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
             _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
             _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$"),
             _spin(mu_label=r"$\mu_{\chi,4}$", sig_label=r"$\sigma_{\chi,4}$"),
             _spin(mu_label=r"$\mu_{\chi,5}$", sig_label=r"$\sigma_{\chi,5}$")]

    return MixtureModel(masses, pairings, spins)


def _mixture_bpl2peaks1pl(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20,   mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40,  mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$"),
        _pl(alpha_label=r"$\alpha_{\rm PL}$", mmin_label=r"$m_{\min,\rm PL}$", mmax_label=r"$m_{\max,\rm PL}$", dmmin_label=r"$dm_{\min,\rm PL}$", dmmax_label=r"$dm_{\max,\rm PL}$", alpha_lo=0, alpha_hi=6, mmin_lo=40, mmin_hi=60, mmax_lo=80, mmax_hi=120)
    ]

    pairings = [_plpairing(beta_label=r"$\beta$")] if shared_beta else \
               [_plpairing(beta_label=r"$\beta_{\rm BPL}$"),
                _plpairing(beta_label=r"$\beta_{\rm G1}$"),
                _plpairing(beta_label=r"$\beta_{\rm G2}$"),
                _plpairing(beta_label=r"$\beta_{\rm PL}$")]

    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else \
            [_spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G1}$",  sig_label=r"$\sigma_{\chi,\rm G1}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm G2}$",  sig_label=r"$\sigma_{\chi,\rm G2}$"),
             _spin(mu_label=r"$\mu_{\chi,\rm PL}$",  sig_label=r"$\sigma_{\chi,\rm PL}$")]

    return MixtureModel(masses, pairings, spins)


# ----------------------------------------------------------------------
# 8. Registry and Parsers
# ----------------------------------------------------------------------

def _make(mix_fn, shared_beta=False, shared_spin=False):
    return PopulationModel(mixture=mix_fn(shared_beta=shared_beta, shared_spin=shared_spin))


_RAW_MODELS = {
    "powerlaw+peak":                   (_mixture_plpeak,        "PL+G"),
    "brokenpowerlaw+2peaks":           (_mixture_bpl2peaks,     "BPL+2G"),
    "brokenpowerlaw+3peaks":           (_mixture_bpl3peaks,     "BPL+3G"),
    "brokenpowerlaw+2peaks+powerlaw":  (_mixture_bpl2peaks1pl,  "BPL+2G+PL"),
    "twopowerlaws+peak":               (_mixture_2pl1peak,      "2PL+G"),
    "twopowerlaws+2peaks":             (_mixture_2pl2peaks,     "2PL+2G"),
    "twopowerlaws+3peaks":             (_mixture_2pl3peaks,     "2PL+3G"),
}

# Construct the full registry dynamically handling all combinations
_MODEL_REGISTRY: dict[str, PopulationModel] = {}
MODEL_NAME_LATEX: dict[str, str] = {"mock_data": r"\text{Mock}"}

for name, (mix_fn, latex_name) in _RAW_MODELS.items():
    # Closure helper
    def bind_fn(f=mix_fn, sb=False, ss=False):
        return lambda: _make(f, shared_beta=sb, shared_spin=ss)

    # 1. Independent Beta, Independent Spin
    _MODEL_REGISTRY[name] = bind_fn(sb=False, ss=False)()
    MODEL_NAME_LATEX[name] = latex_name

    # 2. Shared Beta, Independent Spin
    name_sb = f"{name}_shared_beta"
    _MODEL_REGISTRY[name_sb] = bind_fn(sb=True, ss=False)()
    MODEL_NAME_LATEX[name_sb] = f"{latex_name} (Shared $\\beta$)"

    # 3. Independent Beta, Shared Spin
    name_ss = f"{name}_shared_spin"
    _MODEL_REGISTRY[name_ss] = bind_fn(sb=False, ss=True)()
    MODEL_NAME_LATEX[name_ss] = f"{latex_name} (Shared Spin)"

    # 4. Shared Beta, Shared Spin
    name_both = f"{name}_shared_beta_spin"
    _MODEL_REGISTRY[name_both] = bind_fn(sb=True, ss=True)()
    MODEL_NAME_LATEX[name_both] = f"{latex_name} (Shared $\\beta$, Spin)"


def get_model(pop_model: str) -> PopulationModel:
    try:
        return _MODEL_REGISTRY[pop_model]
    except KeyError:
        raise ValueError(
            f"Unknown model {pop_model!r}. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )

def pop_model_parser(pop_model: str):
    return get_model(pop_model).log_p_pop

def pop_model_prior_parser(pop_model: str) -> tuple[list, list, list, str]:
    model = get_model(pop_model)
    lows, highs, labels = model.prior_bounds()
    return lows, highs, labels, MODEL_NAME_LATEX.get(pop_model, pop_model)