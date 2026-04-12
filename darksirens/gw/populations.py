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

MASS_GRID = jnp.linspace(M_LO, M_HI, N_MASS)
Q_GRID    = jnp.linspace(0.0, 1.0, N_Q)

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

        # Normalize on fixed grid, expanding m1 dimensions safely
        # to ensure it works whether m1 is a scalar or a 1D array
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
        # Reshape normalization factor back to original shape of m1 to prevent array broadcasting errors
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
# 5. JointComponent
# ======================================================================

@dataclass
class JointComponent:
    mass: MassComponent
    pairing: PairingModel

    @property
    def n_params(self):
        return self.mass.n_params + self.pairing.n_params

    @property
    def param_specs(self):
        return [*self.mass.param_specs, *self.pairing.param_specs]

    def __call__(self, m1, q, theta):
        tm = theta[:self.mass.n_params]
        tp = theta[self.mass.n_params:]

        # Dynamically extract mmin and dmmin self-consistently if the components have them.
        # This replaces the hardcoded `isinstance` index unpacking which breaks for arbitrary factories.
        mmin = M_LO
        dmmin = 0.01

        if hasattr(self.mass, "m_min_spec"):
            mmin_idx = self.mass.param_specs.index(self.mass.m_min_spec)
            mmin = tm[mmin_idx]
            
        if hasattr(self.mass, "dm_min_spec"):
            dmmin_idx = self.mass.param_specs.index(self.mass.dm_min_spec)
            dmmin = tm[dmmin_idx]

        return self.mass(m1, tm) * self.pairing(m1, q, mmin, dmmin, tp)


# ======================================================================
# 6. MixtureModel
# ======================================================================

@dataclass
class MixtureModel:
    components: list[JointComponent]

    @property
    def n_weight_params(self):
        return max(len(self.components) - 1, 0)

    @property
    def param_specs(self):
        specs = [
            ParamSpec(rf"$f_{i+1}$", 0.0, 1.0)
            for i in range(self.n_weight_params)
        ]
        for c in self.components:
            specs.extend(c.param_specs)
        return specs

    @property
    def n_params(self):
        return self.n_weight_params + sum(c.n_params for c in self.components)

    def __call__(self, m1, q, theta):
        n_w = self.n_weight_params
        w_raw = theta[:n_w]
        flat = theta[n_w:]

        if n_w > 0:
            w_last = 1.0 - jnp.sum(w_raw)
            w = jnp.concatenate([w_raw, jnp.atleast_1d(w_last)])
        else:
            w = jnp.array([1.0])

        out = 0.0
        off = 0
        for wi, ci in zip(w, self.components):
            sl = flat[off:off + ci.n_params]
            out = out + wi * ci(m1, q, sl)
            off += ci.n_params

        return out

@dataclass
class GlobalPairingMixtureModel:
    mass_components: list[MassComponent]
    pairing: PairingModel

    @property
    def n_weight_params(self):
        return max(len(self.mass_components) - 1, 0)

    @property
    def param_specs(self):
        specs = [
            ParamSpec(rf"$f_{i+1}$", 0.0, 1.0)
            for i in range(self.n_weight_params)
        ]
        for c in self.mass_components:
            specs.extend(c.param_specs)
        # Add the single shared pairing parameter(s) at the very end
        specs.extend(self.pairing.param_specs)
        return specs

    @property
    def n_params(self):
        return self.n_weight_params + sum(c.n_params for c in self.mass_components) + self.pairing.n_params

    def __call__(self, m1, q, theta):
        n_w = self.n_weight_params
        w_raw = theta[:n_w]
        
        # Extract pairing params from the end, flat mass params from the middle
        n_p = self.pairing.n_params
        tp = theta[-n_p:]
        flat = theta[n_w : -n_p] if n_p > 0 else theta[n_w:]

        if n_w > 0:
            w_last = 1.0 - jnp.sum(w_raw)
            w = jnp.concatenate([w_raw, jnp.atleast_1d(w_last)])
        else:
            w = jnp.array([1.0])

        out = 0.0
        off = 0
        for wi, ci in zip(w, self.mass_components):
            tm = flat[off:off + ci.n_params]

            # Dynamically extract mmin and dmmin for the pairing filter
            mmin = M_LO
            dmmin = 0.01

            if hasattr(ci, "m_min_spec"):
                mmin_idx = ci.param_specs.index(ci.m_min_spec)
                mmin = tm[mmin_idx]
                
            if hasattr(ci, "dm_min_spec"):
                dmmin_idx = ci.param_specs.index(ci.dm_min_spec)
                dmmin = tm[dmmin_idx]

            # Multiply component mass evaluate by the global pairing evaluate
            out = out + wi * ci(m1, tm) * self.pairing(m1, q, mmin, dmmin, tp)
            off += ci.n_params

        return out
    

# ======================================================================
# 7. PopulationModel
# ======================================================================

_LOG_P_SPIN = float(jnp.log(0.25))

@dataclass
class PopulationModel:
    mixture: MixtureModel

    @property
    def param_specs(self):
        return [*self.mixture.param_specs, ParamSpec(r"$\gamma$", -10.0, 10.0)]

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_p_pop(self, m1, q, z, theta):
        tm = theta[:-1]
        gamma = theta[-1]
        p = self.mixture(m1, q, tm)
        return jnp.where(p > 0, jnp.log(p), -1e10) + gamma * jnp.log1p(z) + _LOG_P_SPIN


# ======================================================================
# 8. Factories: build components, then general K-mixture
# ======================================================================

# ----------------------------------------------------------------------
# Mass factories
# ----------------------------------------------------------------------

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
    """Returns a PowerLaw mass component with specified prior ranges."""
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
    """Returns a BrokenPowerLaw mass component with specified prior ranges."""
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
    """Returns a Gaussian mass component with specified prior ranges."""
    return Gaussian(
        mu_spec   = ParamSpec(mu_label,  mu_lo,  mu_hi),
        sigma_spec= ParamSpec(sig_label, sig_lo, sig_hi),
    )


# ----------------------------------------------------------------------
# Pairing factory
# ----------------------------------------------------------------------

def _plpairing(beta_label=r"$\beta$", beta_lo=-2.0, beta_hi=7.0):
    """Returns a PowerLaw pairing model (p(q|m1))."""
    return PowerLawPairing(
        beta_spec = ParamSpec(beta_label, beta_lo, beta_hi)
    )


# ----------------------------------------------------------------------
# Joint component helper
# ----------------------------------------------------------------------

def _joint(mass, pairing=None):
    """
    Helper to bundle a mass component and a pairing model.
    Defaults to PowerLawPairing if none is provided.
    """
    if pairing is None:
        pairing = _plpairing()
    return JointComponent(mass, pairing)


# ----------------------------------------------------------------------
# General K-mixtures built from components
# ----------------------------------------------------------------------

def _mixture_plpeak(shared_beta=False):
    """2-component mixture: PowerLaw + Gaussian"""
    pl = _pl()
    g  = _gauss(20, 50)

    if shared_beta:
        return GlobalPairingMixtureModel(
            mass_components=[pl, g],
            pairing=_plpairing(beta_label=r"$\beta$")
        )
    return MixtureModel([
        _joint(pl, _plpairing(beta_label=r"$\beta_{\rm PL}$")),
        _joint(g,  _plpairing(beta_label=r"$\beta_{\rm G}$")),
    ])


def _mixture_bpl2peaks(shared_beta=False):
    """3-component mixture: BrokenPowerLaw + 2 Gaussians"""
    bpl = _bpl()
    g1  = _gauss(5, 20,  mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$")
    g2  = _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")

    if shared_beta:
        return GlobalPairingMixtureModel(
            mass_components=[bpl, g1, g2],
            pairing=_plpairing(beta_label=r"$\beta$")
        )
    return MixtureModel([
        _joint(bpl, _plpairing(beta_label=r"$\beta_{\rm BPL}$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_{\rm G1}$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_{\rm G2}$")),
    ])


def _mixture_bpl3peaks(shared_beta=False):
    """4-component mixture: BrokenPowerLaw + 3 Gaussians"""
    bpl = _bpl()
    g1  = _gauss(5, 20,   mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$")
    g2  = _gauss(25, 40,  mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")
    g3  = _gauss(50, 100, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", sig_hi=20)

    if shared_beta:
        return GlobalPairingMixtureModel(
            mass_components=[bpl, g1, g2, g3],
            pairing=_plpairing(beta_label=r"$\beta$")
        )
    return MixtureModel([
        _joint(bpl, _plpairing(beta_label=r"$\beta_{\rm BPL}$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_{\rm G1}$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_{\rm G2}$")),
        _joint(g3,  _plpairing(beta_label=r"$\beta_{\rm G3}$")),
    ])


def _mixture_2pl1peak(shared_beta=False):
    """3-component mixture: 2 PowerLaws + 1 Gaussian"""
    pl1 = _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", 
              dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50)
    pl2 = _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", 
              dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100)
    g   = _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$")

    if shared_beta:
        return GlobalPairingMixtureModel([pl1, pl2, g], _plpairing(beta_label=r"$\beta$"))
    return MixtureModel([
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g,   _plpairing(beta_label=r"$\beta_3$")),
    ])


def _mixture_2pl2peaks(shared_beta=False):
    """4-component mixture: 2 PowerLaws + 2 Gaussians"""
    pl1 = _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", 
              dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50)
    pl2 = _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", 
              dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100)
    g1  = _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$")
    g2  = _gauss(20, 50, sig_hi=15, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$")

    if shared_beta:
        return GlobalPairingMixtureModel([pl1, pl2, g1, g2], _plpairing(beta_label=r"$\beta$"))
    return MixtureModel([
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_3$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_4$")),
    ])


def _mixture_2pl3peaks(shared_beta=False):
    """5-component mixture: 2 PowerLaws + 3 Gaussians"""
    pl1 = _pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$", 
              dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50)
    pl2 = _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$", 
              dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$", 
              alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100)
    g1  = _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$")
    g2  = _gauss(20, 50, sig_hi=10, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$")
    g3  = _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_5$", sig_label=r"$\sigma_5$")

    if shared_beta:
        return GlobalPairingMixtureModel([pl1, pl2, g1, g2, g3], _plpairing(beta_label=r"$\beta$"))
    return MixtureModel([
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_3$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_4$")),
        _joint(g3,  _plpairing(beta_label=r"$\beta_5$")),
    ])


# ----------------------------------------------------------------------
# Registry and Parsers
# ----------------------------------------------------------------------

def _make(mix_fn, shared_beta=False):
    """Wraps a mixture-generator into a full PopulationModel."""
    return PopulationModel(mixture=mix_fn(shared_beta=shared_beta))


_RAW_MODELS = {
    "powerlaw+peak":           (_mixture_plpeak,    "PL+G"),
    "brokenpowerlaw+2peaks":   (_mixture_bpl2peaks, "BPL+2G"),
    "brokenpowerlaw+3peaks":   (_mixture_bpl3peaks, "BPL+3G"),
    "twopowerlaws+peak":       (_mixture_2pl1peak,  "2PL+G"),
    "twopowerlaws+2peaks":     (_mixture_2pl2peaks, "2PL+2G"),
    "twopowerlaws+3peaks":     (_mixture_2pl3peaks, "2PL+3G"),
}

# Construct the full registry (independent betas, shared betas)
_MODEL_REGISTRY: dict[str, PopulationModel] = {}
MODEL_NAME_LATEX: dict[str, str] = {"mock_data": r"\text{Mock}"}

for name, (mix_fn, latex_name) in _RAW_MODELS.items():
    # Helper functions prevent late-binding closure bugs in the loop
    def bind_fn(f=mix_fn, shared=False):
        return lambda: _make(f, shared_beta=shared)

    # 1. Independent Betas
    model_indep = bind_fn(shared=False)()
    _MODEL_REGISTRY[name] = model_indep
    MODEL_NAME_LATEX[name] = latex_name

    # 2. Shared Beta
    name_shared = f"{name}_shared_beta"
    model_shared = bind_fn(shared=True)()
    _MODEL_REGISTRY[name_shared] = model_shared
    MODEL_NAME_LATEX[name_shared] = f"{latex_name} (Shared $\\beta$)"
    

def get_model(pop_model: str) -> PopulationModel:
    """Retrieves the PopulationModel instance by name."""
    try:
        return _MODEL_REGISTRY[pop_model]
    except KeyError:
        raise ValueError(
            f"Unknown model {pop_model!r}. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )


def pop_model_parser(pop_model: str):
    """Returns the log_p_pop function for the requested model name."""
    return get_model(pop_model).log_p_pop


def pop_model_prior_parser(pop_model: str) -> tuple[list, list, list, str]:
    """Returns prior bounds and labels for the requested model name."""
    model = get_model(pop_model)
    lows, highs, labels = model.prior_bounds()
    return lows, highs, labels, MODEL_NAME_LATEX.get(pop_model, pop_model)