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

def _mixture_plpeak():
    """2-component mixture: PowerLaw + Gaussian"""
    pl = _pl()
    g  = _gauss(20, 50)

    components = [
        _joint(pl, _plpairing(beta_label=r"$\beta_{\rm PL}$")),
        _joint(g,  _plpairing(beta_label=r"$\beta_{\rm G}$")),
    ]
    return MixtureModel(components)


def _mixture_bpl2peaks():
    """3-component mixture: BrokenPowerLaw + 2 Gaussians"""
    bpl = _bpl()
    g1  = _gauss(5, 20,  mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$")
    g2  = _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")

    components = [
        _joint(bpl, _plpairing(beta_label=r"$\beta_{\rm BPL}$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_{\rm G1}$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_{\rm G2}$")),
    ]
    return MixtureModel(components)


def _mixture_bpl3peaks():
    """4-component mixture: BrokenPowerLaw + 3 Gaussians"""
    bpl = _bpl()
    g1  = _gauss(5, 20,   mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$")
    g2  = _gauss(25, 40,  mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")
    g3  = _gauss(50, 100, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", sig_hi=20)

    components = [
        _joint(bpl, _plpairing(beta_label=r"$\beta_{\rm BPL}$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_{\rm G1}$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_{\rm G2}$")),
        _joint(g3,  _plpairing(beta_label=r"$\beta_{\rm G3}$")),
    ]
    return MixtureModel(components)


def _mixture_2pl1peak():
    """3-component mixture: 2 PowerLaws + 1 Gaussian"""
    pl1 = _pl(
        alpha_label=r"$\alpha_1$",
        mmin_label=r"$m_{\min,1}$",
        mmax_label=r"$m_{\max,1}$",
        dmmin_label=r"$dm_{\min,1}$",
        dmmax_label=r"$dm_{\max,1}$",
        alpha_lo=0, alpha_hi=6,
        mmin_lo=2, mmin_hi=10,
        mmax_lo=15, mmax_hi=50,
    )

    pl2 = _pl(
        alpha_label=r"$\alpha_2$",
        mmin_label=r"$m_{\min,2}$",
        mmax_label=r"$m_{\max,2}$",
        dmmin_label=r"$dm_{\min,2}$",
        dmmax_label=r"$dm_{\max,2}$",
        alpha_lo=0, alpha_hi=6,
        mmin_lo=20, mmin_hi=40,
        mmax_lo=50, mmax_hi=100,
    )

    g = _gauss(
        50, 100,
        sig_hi=20,
        mu_label=r"$\mu_3$",
        sig_label=r"$\sigma_3$",
    )

    components = [
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g,   _plpairing(beta_label=r"$\beta_3$")),
    ]
    return MixtureModel(components)

def _mixture_2pl2peaks():
    """4-component mixture: 2 PowerLaws + 2 Gaussians"""
    pl1 = _pl(
        alpha_label=r"$\alpha_1$",
        mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$",
        dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$",
        alpha_lo=0, alpha_hi=6, 
        mmin_lo=2, mmin_hi=10, 
        mmax_lo=15, mmax_hi=50,
    )

    pl2 = _pl(
        alpha_label=r"$\alpha_2$",
        mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$",
        dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$",
        alpha_lo=0, alpha_hi=6, 
        mmin_lo=20, mmin_hi=40, 
        mmax_lo=50, mmax_hi=100,
    )

    g1 = _gauss(
        5, 20, 
        sig_hi=10,
        mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$",
    )

    g2 = _gauss(
        20, 50, 
        sig_hi=15,
        mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$",
    )

    components = [
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_3$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_4$")),
    ]
    return MixtureModel(components)


def _mixture_2pl3peaks():
    """5-component mixture: 2 PowerLaws + 3 Gaussians"""
    pl1 = _pl(
        alpha_label=r"$\alpha_1$",
        mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$",
        dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$",
        alpha_lo=0, alpha_hi=6, 
        mmin_lo=2, mmin_hi=10, 
        mmax_lo=15, mmax_hi=50,
    )

    pl2 = _pl(
        alpha_label=r"$\alpha_2$",
        mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$",
        dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$",
        alpha_lo=0, alpha_hi=6, 
        mmin_lo=20, mmin_hi=40, 
        mmax_lo=50, mmax_hi=100,
    )

    g1 = _gauss(
        5, 20, 
        sig_hi=10,
        mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$",
    )

    g2 = _gauss(
        20, 50, 
        sig_hi=10,
        mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$",
    )

    g3 = _gauss(
        50, 100, 
        sig_hi=20,
        mu_label=r"$\mu_5$", sig_label=r"$\sigma_5$",
    )

    components = [
        _joint(pl1, _plpairing(beta_label=r"$\beta_1$")),
        _joint(pl2, _plpairing(beta_label=r"$\beta_2$")),
        _joint(g1,  _plpairing(beta_label=r"$\beta_3$")),
        _joint(g2,  _plpairing(beta_label=r"$\beta_4$")),
        _joint(g3,  _plpairing(beta_label=r"$\beta_5$")),
    ]
    return MixtureModel(components)


# ----------------------------------------------------------------------
# Registry and Parsers
# ----------------------------------------------------------------------

def _make(mix_fn):
    """Wraps a mixture-generator into a full PopulationModel."""
    return PopulationModel(mixture=mix_fn())


# Define standard models
_BASE_MODELS = {
    "powerlaw+peak":           _make(_mixture_plpeak),
    "brokenpowerlaw+2peaks":   _make(_mixture_bpl2peaks),
    "brokenpowerlaw+3peaks":   _make(_mixture_bpl3peaks),
    "twopowerlaws+peak":       _make(_mixture_2pl1peak),
    "twopowerlaws+2peaks":     _make(_mixture_2pl2peaks),
    "twopowerlaws+3peaks":     _make(_mixture_2pl3peaks),
}

# Construct the full registry (including symmetric variants)
_MODEL_REGISTRY: dict[str, PopulationModel] = {}
for name, model in _BASE_MODELS.items():
    _MODEL_REGISTRY[name] = model
    _MODEL_REGISTRY[f"symmetric_{name}"] = model

# LaTeX labels for plots/diagnostics
MODEL_NAME_LATEX: dict[str, str] = {
    "powerlaw+peak":                   "PL+G",
    "brokenpowerlaw+2peaks":           "BPL+2G",
    "brokenpowerlaw+3peaks":           "BPL+3G",
    "twopowerlaws+peak":               "2PL+G",
    "twopowerlaws+2peaks":             "2PL+2G",
    "twopowerlaws+3peaks":             "2PL+3G",
    "symmetric_powerlaw+peak":         "Sym PL+G",
    "symmetric_brokenpowerlaw+2peaks": "Sym BPL+2G",
    "symmetric_brokenpowerlaw+3peaks": "Sym BPL+3G",
    "symmetric_twopowerlaws+peak":     "Sym 2PL+G",
    "symmetric_twopowerlaws+2peaks":   "Sym 2PL+2G",
    "symmetric_twopowerlaws+3peaks":   "Sym 2PL+3G",
    "mock_data":                       r"\text{Mock}",
}

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