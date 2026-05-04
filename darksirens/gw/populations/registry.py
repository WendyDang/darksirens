"""
registry.py
-----------
Population model registry for gravitational-wave source inference.

Structure
---------
1. Prior bounds constants  — all default prior ranges in one place
2. Component factory functions — each builds one model component
3. Mixture factories — assemble components into MixtureModel objects
4. Model registry — maps string keys to PopulationModel instances
5. Fiducial parameters — default values for --fix_population=True
6. Public API — get_model, pop_model_parser, pop_model_prior_parser

Adding a new model
------------------
(a) Write a mixture factory function _mixture_<name>().
(b) Add it to _RAW_MODELS with a LaTeX label.
(c) Add a fiducial parameter entry to _FIDUCIAL_PARAMS.
That is all — the variant loop and public API require no changes.
"""

from __future__ import annotations
from typing import NamedTuple

import jax.numpy as jnp

from .base import ParamSpec, MixtureModel, PopulationModel
from .parametric import (
    PowerLaw, BrokenPowerLaw, Gaussian,
    PowerLawPairing, TruncatedGaussianSpin,
)
from .gp import (
    GaussianProcessMass1D, GaussianProcessMassRatio1D,
    GaussianProcessPairing2D,
)


# ============================================================
# 1. Prior bounds
# ============================================================
# All default prior ranges live here.  Override them by passing
# explicit lo/hi values to the factory functions below.
#
# Notation: each value is (low, high) for a uniform prior.

class _B(NamedTuple):
    """Compact (low, high) prior-bound pair."""
    lo: float
    hi: float


# -- Mass --
ALPHA        = _B(-4.0,   6.0)   # Power-law slope
ALPHA_BPL    = _B( 0.0,   6.0)   # Broken power-law slopes (both arms)
M_BREAK      = _B(20.0,  50.0)   # Break mass for BPL
M_MIN        = _B( 2.0,  10.0)   # Low-mass cutoff
M_MAX        = _B(50.0, 100.0)   # High-mass cutoff
DM_MIN       = _B( 0.01, 10.0)   # Low-mass smoothing window
DM_MAX       = _B( 0.01, 20.0)   # High-mass smoothing window
GAUSS_MU     = _B( 5.0,  50.0)   # Gaussian peak mean (overridden per peak)
GAUSS_SIGMA  = _B( 1.0,  10.0)   # Gaussian peak width

# -- Pairing --
BETA         = _B(-2.0,   7.0)   # Power-law mass-ratio slope

# -- Spin --
CHI_MU       = _B(-1.0,   1.0)   # Effective spin mean
CHI_SIGMA    = _B( 0.01,  1.0)   # Effective spin width

# -- GP kernel --
GP_AMP       = _B( 0.01,  5.0)   # GP kernel amplitude
GP_LS        = _B( 0.05,  1.0)   # GP length scale (mass space)
GP_LS_M_PAIR = _B( 0.5,   3.0)   # GP length scale (m1 axis of 2D pairing)
GP_LS_Q_PAIR = _B( 0.2,   1.0)   # GP length scale (q axis of 2D pairing)
GP_ALPHA     = _B(-4.0,  12.0)   # GP mean-function power-law slope
GP_BETA_Q    = _B(-4.0,  12.0)   # GP mean-function pairing slope
GP_Y         = _B(-7.0,   7.0)   # GP node values (mass, spin)
GP_Y_PAIR    = _B(-5.0,   5.0)   # GP node values (pairing)


# ============================================================
# 2. Component factory functions
# ============================================================
# Each function returns one model component.  Arguments are labels
# (for the corner plot) and optional bound overrides.  The defaults
# come from the constants above so the call sites stay short.

def _pl(
    alpha_label = r"$\alpha$",
    mmin_label  = r"$m_{\min}$",
    mmax_label  = r"$m_{\max}$",
    dmmin_label = r"$\delta m_{\min}$",
    dmmax_label = r"$\delta m_{\max}$",
    *,                          # keyword-only below
    alpha = ALPHA,
    mmin  = M_MIN,
    mmax  = M_MAX,
    dmmin = DM_MIN,
    dmmax = DM_MAX,
) -> PowerLaw:
    """Single power-law mass component."""
    return PowerLaw(
        ParamSpec(alpha_label, alpha.lo, alpha.hi),
        ParamSpec(mmin_label,  mmin.lo,  mmin.hi),
        ParamSpec(mmax_label,  mmax.lo,  mmax.hi),
        ParamSpec(dmmin_label, dmmin.lo, dmmin.hi),
        ParamSpec(dmmax_label, dmmax.lo, dmmax.hi),
    )


def _bpl(
    alpha1_label = r"$\alpha_1$",
    alpha2_label = r"$\alpha_2$",
    break_label  = r"$m_{\rm break}$",
    mmin_label   = r"$m_{\min}$",
    mmax_label   = r"$m_{\max}$",
    dmmin_label  = r"$\delta m_{\min}$",
    dmmax_label  = r"$\delta m_{\max}$",
    *,
    alpha1 = ALPHA_BPL,
    alpha2 = ALPHA_BPL,
    mbreak = M_BREAK,
    mmin   = M_MIN,
    mmax   = _B(40.0, 200.0),   # BPL typically fits wider mass range
    dmmin  = _B(0.01, 100.0),
    dmmax  = _B(0.01, 100.0),
) -> BrokenPowerLaw:
    """Broken power-law mass component."""
    return BrokenPowerLaw(
        ParamSpec(alpha1_label, alpha1.lo, alpha1.hi),
        ParamSpec(alpha2_label, alpha2.lo, alpha2.hi),
        ParamSpec(break_label,  mbreak.lo, mbreak.hi),
        ParamSpec(mmin_label,   mmin.lo,   mmin.hi),
        ParamSpec(mmax_label,   mmax.lo,   mmax.hi),
        ParamSpec(dmmin_label,  dmmin.lo,  dmmin.hi),
        ParamSpec(dmmax_label,  dmmax.lo,  dmmax.hi),
    )


def _gauss(
    mu_label  = r"$\mu$",
    sig_label = r"$\sigma$",
    *,
    mu  = GAUSS_MU,
    sig = GAUSS_SIGMA,
) -> Gaussian:
    """Gaussian mass peak component."""
    return Gaussian(
        ParamSpec(mu_label,  mu.lo,  mu.hi),
        ParamSpec(sig_label, sig.lo, sig.hi),
    )


def _plpairing(
    beta_label = r"$\beta$",
    *,
    beta = BETA,
) -> PowerLawPairing:
    """Power-law mass-ratio pairing component."""
    return PowerLawPairing(
        ParamSpec(beta_label, beta.lo, beta.hi),
    )


def _spin(
    mu_label  = r"$\mu_\chi$",
    sig_label = r"$\sigma_\chi$",
    *,
    mu  = CHI_MU,
    sig = CHI_SIGMA,
) -> TruncatedGaussianSpin:
    """Truncated Gaussian effective-spin component."""
    return TruncatedGaussianSpin(
        ParamSpec(mu_label,  mu.lo,  mu.hi),
        ParamSpec(sig_label, sig.lo, sig.hi),
    )


def _gp_mass(
    mmin_label  = r"$m_{\min}$",
    mmax_label  = r"$m_{\max}$",
    dmmin_label = r"$\delta m_{\min}$",
    dmmax_label = r"$\delta m_{\max}$",
    alpha_label = r"$\alpha$",
    amp_label   = r"$A$",
    ls_label    = r"$\ell$",
    y_labels    = None,
    *,
    mmin  = M_MIN,
    mmax  = M_MAX,
    dmmin = DM_MIN,
    dmmax = DM_MAX,
    alpha = GP_ALPHA,
    amp   = GP_AMP,
    ls    = GP_LS,
    y     = GP_Y,
) -> GaussianProcessMass1D:
    """
    1D GP mass component with 11 inducing nodes.

    The GP mean function is a power law with slope `alpha`.
    Node values y_0…y_10 are deviations from that mean in log-space.
    Zero nodes → pure power law.
    """
    if y_labels is None:
        y_labels = [rf"$y_{{{i}}}$" for i in range(11)]
    return GaussianProcessMass1D(
        ParamSpec(mmin_label,  mmin.lo,  mmin.hi),
        ParamSpec(mmax_label,  mmax.lo,  mmax.hi),
        ParamSpec(dmmin_label, dmmin.lo, dmmin.hi),
        ParamSpec(dmmax_label, dmmax.lo, dmmax.hi),
        ParamSpec(alpha_label, alpha.lo, alpha.hi),
        ParamSpec(amp_label,   amp.lo,   amp.hi),
        ParamSpec(ls_label,    ls.lo,    ls.hi),
        *[ParamSpec(label, y.lo, y.hi) for label in y_labels],
    )


def _gp_pairing(
    beta_label = r"$\beta_q$",
    amp_label  = r"$A_q$",
    ls_label   = r"$\ell_q$",
    y_labels   = None,
    *,
    beta = GP_BETA_Q,
    amp  = GP_AMP,
    ls   = GP_LS,
    y    = GP_Y_PAIR,
) -> GaussianProcessMassRatio1D:
    """
    1D GP mass-ratio component with 5 inducing nodes.

    The GP mean function is a power law q^beta.
    Zero nodes → pure power law.
    """
    if y_labels is None:
        y_labels = [rf"$y_{{q,{i}}}$" for i in range(5)]
    return GaussianProcessMassRatio1D(
        ParamSpec(beta_label, beta.lo, beta.hi),
        ParamSpec(amp_label,  amp.lo,  amp.hi),
        ParamSpec(ls_label,   ls.lo,   ls.hi),
        *[ParamSpec(label, y.lo, y.hi) for label in y_labels],
    )


def _gp_mass_pairing_2d(
    beta_label  = r"$\beta$",
    amp_label   = r"$A_q$",
    ls_m_label  = r"$\ell_{m,q}$",
    ls_q_label  = r"$\ell_{q,q}$",
    y_labels    = None,
    *,
    beta  = GP_BETA_Q,
    amp   = _B(0.01, 3.0),
    ls_m  = GP_LS_M_PAIR,
    ls_q  = GP_LS_Q_PAIR,
    y     = GP_Y_PAIR,
) -> GaussianProcessPairing2D:
    """
    2D GP pairing component on (log m1, q) with a 4×4 = 16 node grid.

    The anisotropic Matern52 kernel uses separate length scales for the
    m1 and q axes (ls_m, ls_q).  Zero nodes → pure power law q^beta.
    """
    if y_labels is None:
        y_labels = [rf"$y_{{q,{i}}}$" for i in range(16)]
    return GaussianProcessPairing2D(
        ParamSpec(beta_label, beta.lo, beta.hi),
        ParamSpec(amp_label,  amp.lo,  amp.hi),
        ParamSpec(ls_m_label, ls_m.lo, ls_m.hi),
        ParamSpec(ls_q_label, ls_q.lo, ls_q.hi),
        *[ParamSpec(label, y.lo, y.hi) for label in y_labels],
    )


# ============================================================
# 3. Mixture factories
# ============================================================
# Each returns a MixtureModel.  Shared-{beta,spin} flags collapse
# the pairing / spin components to one shared instance.
#
# Convention for labels:
#   Component suffix  _PL, _G, _BPL, _G1/G2/G3 …
#   Spin suffix       _PL, _G, …  matching the mass component

def _make_spins(labels: list[str], shared_spin: bool) -> list:
    if shared_spin:
        return [_spin()]
    return [_spin(mu_label=rf"$\mu_{{\chi,\rm {s}}}$",
                  sig_label=rf"$\sigma_{{\chi,\rm {s}}}$") for s in labels]


def _make_pairings(labels: list[str], shared_beta: bool) -> list:
    if shared_beta:
        return [_plpairing()]
    return [_plpairing(beta_label=rf"$\beta_{{\rm {s}}}$") for s in labels]


def _mixture_plpeak(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Power-law + Gaussian peak.  The standard LVK POWER LAW + PEAK model."""
    component_labels = ["PL", "G"]
    masses = [
        _pl(),
        _gauss(mu_label=r"$\mu_G$", sig_label=r"$\sigma_G$",
               mu=_B(20, 50), sig=GAUSS_SIGMA),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_bpl2peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + two Gaussian peaks."""
    component_labels = ["BPL", "G1", "G2"]
    masses = [
        _bpl(),
        _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$",  mu=_B( 5, 20)),
        _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$",  mu=_B(25, 40)),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_bpl3peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + three Gaussian peaks."""
    component_labels = ["BPL", "G1", "G2", "G3"]
    masses = [
        _bpl(),
        _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$",  mu=_B( 5, 20)),
        _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$",  mu=_B(25, 40)),
        _gauss(mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$",  mu=_B(50,100), sig=_B(1, 20)),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_bpl2peaks1pl(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + two Gaussian peaks + high-mass power-law tail."""
    component_labels = ["BPL", "G1", "G2", "PL"]
    masses = [
        _bpl(),
        _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5, 20)),
        _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(25, 40)),
        _pl(
            alpha_label = r"$\alpha_{\rm PL}$",
            mmin_label  = r"$m_{\min,\rm PL}$",
            mmax_label  = r"$m_{\max,\rm PL}$",
            dmmin_label = r"$\delta m_{\min,\rm PL}$",
            dmmax_label = r"$\delta m_{\max,\rm PL}$",
            alpha = _B(0, 6),
            mmin  = _B(40, 60),
            mmax  = _B(80, 120),
        ),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_2pl1peak(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + one Gaussian peak."""
    component_labels = ["PL1", "PL2", "G"]
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
            mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
            dmmax_label=r"$\delta m_{\max,1}$",
            alpha=_B(0, 6), mmin=_B(2, 10), mmax=_B(15, 50),
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
            mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
            dmmax_label=r"$\delta m_{\max,2}$",
            alpha=_B(0, 6), mmin=_B(20, 40), mmax=_B(50, 100),
        ),
        _gauss(mu_label=r"$\mu_G$", sig_label=r"$\sigma_G$",
               mu=_B(50, 100), sig=_B(1, 20)),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_2pl2peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + two Gaussian peaks."""
    component_labels = ["PL1", "PL2", "G1", "G2"]
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
            mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
            dmmax_label=r"$\delta m_{\max,1}$",
            alpha=_B(0, 6), mmin=_B(2, 10), mmax=_B(15, 50),
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
            mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
            dmmax_label=r"$\delta m_{\max,2}$",
            alpha=_B(0, 6), mmin=_B(20, 40), mmax=_B(50, 100),
        ),
        _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$",
               mu=_B( 5, 20), sig=_B(1, 10)),
        _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$",
               mu=_B(20, 50), sig=_B(1, 15)),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_2pl3peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + three Gaussian peaks."""
    component_labels = ["PL1", "PL2", "G1", "G2", "G3"]
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
            mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
            dmmax_label=r"$\delta m_{\max,1}$",
            alpha=_B(0, 6), mmin=_B(2, 10), mmax=_B(15, 50),
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
            mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
            dmmax_label=r"$\delta m_{\max,2}$",
            alpha=_B(0, 6), mmin=_B(20, 40), mmax=_B(50, 100),
        ),
        _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$",
               mu=_B( 5, 20), sig=_B(1, 10)),
        _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$",
               mu=_B(20, 50), sig=_B(1, 10)),
        _gauss(mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$",
               mu=_B(50,100), sig=_B(1, 20)),
    ]
    return MixtureModel(
        masses,
        _make_pairings(component_labels, shared_beta),
        _make_spins(component_labels, shared_spin),
    )


def _mixture_gp_mass(shared_beta=True, shared_spin=True) -> MixtureModel:
    """Single GP mass component with power-law pairing and Gaussian spin."""
    return MixtureModel(
        [_gp_mass()],
        [_plpairing()],
        [_spin()],
    )


def _mixture_gp_mass_pairing(shared_beta=True, shared_spin=True) -> MixtureModel:
    """GP mass + 1D GP mass-ratio pairing + Gaussian spin."""
    return MixtureModel(
        [_gp_mass()],
        [_gp_pairing()],
        [_spin()],
    )


def _mixture_gp_mass_pairing_joint(shared_beta=True, shared_spin=True) -> MixtureModel:
    """GP mass + 2D GP pairing on (log m1, q) + Gaussian spin."""
    return MixtureModel(
        [_gp_mass()],
        [_gp_mass_pairing_2d()],
        [_spin()],
    )


# ============================================================
# 4. Model registry
# ============================================================
# Maps CLI model name → (mixture_factory, LaTeX label).
# Shared-beta and shared-spin variants are generated automatically.

_RAW_MODELS: dict[str, tuple] = {
    # name                              mixture factory              LaTeX label
    "powerlaw+peak":                   (_mixture_plpeak,            "PL+G"),
    "brokenpowerlaw+2peaks":           (_mixture_bpl2peaks,         "BPL+2G"),
    "brokenpowerlaw+3peaks":           (_mixture_bpl3peaks,         "BPL+3G"),
    "brokenpowerlaw+2peaks+powerlaw":  (_mixture_bpl2peaks1pl,      "BPL+2G+PL"),
    "twopowerlaws+peak":               (_mixture_2pl1peak,          "2PL+G"),
    "twopowerlaws+2peaks":             (_mixture_2pl2peaks,         "2PL+2G"),
    "twopowerlaws+3peaks":             (_mixture_2pl3peaks,         "2PL+3G"),
    "gp_mass":                         (_mixture_gp_mass,           "GP"),
    "gp_mass_pairing":                 (_mixture_gp_mass_pairing,   "GPxGP"),
    "gp_mass_pairing_joint":           (_mixture_gp_mass_pairing_joint, "GP 2D"),
}

_MODEL_REGISTRY: dict[str, PopulationModel] = {}
MODEL_NAME_LATEX: dict[str, str] = {"mock_data": r"\text{Mock}"}

for _name, (_mix_fn, _latex) in _RAW_MODELS.items():
    for _sb, _ss, _suffix, _label_suffix in [
        (False, False, "",                  ""),
        (True,  False, "_shared_beta",      r" (Shared $\beta$)"),
        (False, True,  "_shared_spin",      r" (Shared Spin)"),
        (True,  True,  "_shared_beta_spin", r" (Shared $\beta$, Spin)"),
    ]:
        _key = _name + _suffix
        _MODEL_REGISTRY[_key] = PopulationModel(
            mixture=_mix_fn(shared_beta=_sb, shared_spin=_ss)
        )
        MODEL_NAME_LATEX[_key] = _latex + _label_suffix


# ============================================================
# 5. Fiducial parameters
# ============================================================
# Used when --fix_population=True.  Each entry documents what every
# number is so the mapping back to param_specs is unambiguous.
#
# Ordering must match PopulationModel.param_specs exactly:
#   weights → mass params → pairing params → spin params → gamma
#
# Style: use inline comments for every number.  Omit nothing.

def _fiducial_pl() -> list:
    return [
        # alpha  mmin   mmax  dmmin  dmmax
        2.3,     5.0,  80.0,   3.0,  10.0,
    ]

def _fiducial_bpl() -> list:
    return [
        # alpha1  alpha2  mbreak  mmin  mmax  dmmin  dmmax
        2.0,      4.0,    30.0,   5.0,  80.0,  3.0,  10.0,
    ]

def _fiducial_gauss_low() -> list:
    return [10.0, 3.0]     # mu_low, sigma_low  [solar masses]

def _fiducial_gauss_mid() -> list:
    return [35.0, 5.0]     # mu_mid, sigma_mid

def _fiducial_gauss_high() -> list:
    return [70.0, 10.0]    # mu_high, sigma_high

def _fiducial_pl_pairing(n: int = 1) -> list:
    return [1.0] * n       # beta per component

def _fiducial_spin(n: int = 1) -> list:
    return [0.0, 0.1] * n  # (mu_chi, sigma_chi) per component

def _fiducial_gp_mass() -> list:
    return [
        # mmin   mmax  dmmin  dmmax
          5.0,  80.0,   3.0,  10.0,
        # alpha   amp    ls
          2.3,    1.0,   1.0,
        # y0…y10 (zero = pure power law, no GP deviation)
        *([0.0] * 11),
    ]

def _fiducial_gp_pairing() -> list:
    return [
        # beta    amp   ls
          1.5,    1.0,  0.5,
        # y0…y4 (zero = pure q^beta)
        *([0.0] * 5),
    ]

def _fiducial_gp_pairing_2d() -> list:
    return [
        # beta    amp    ls_m   ls_q
          0.0,    1.0,   1.0,   0.5,
        # y0…y15 (zero = pure q^beta everywhere)
        *([0.0] * 16),
    ]


_FIDUCIAL_PARAMS: dict[str, dict] = {
    # ----------------------------------------------------------------
    # powerlaw+peak
    # ----------------------------------------------------------------
    "powerlaw+peak": dict(
        n_comp  = 2,
        weights = [0.1],           # fraction in power-law component
        masses  = [*_fiducial_pl(), *_fiducial_gauss_mid()],
    ),

    # ----------------------------------------------------------------
    # brokenpowerlaw+2peaks
    # ----------------------------------------------------------------
    "brokenpowerlaw+2peaks": dict(
        n_comp  = 3,
        weights = [0.10, 0.05],
        masses  = [*_fiducial_bpl(), *_fiducial_gauss_low(), *_fiducial_gauss_mid()],
    ),

    # ----------------------------------------------------------------
    # brokenpowerlaw+3peaks
    # ----------------------------------------------------------------
    "brokenpowerlaw+3peaks": dict(
        n_comp  = 4,
        weights = [0.10, 0.05, 0.03],
        masses  = [*_fiducial_bpl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid(), *_fiducial_gauss_high()],
    ),

    # ----------------------------------------------------------------
    # brokenpowerlaw+2peaks+powerlaw
    # ----------------------------------------------------------------
    "brokenpowerlaw+2peaks+powerlaw": dict(
        n_comp  = 4,
        weights = [0.10, 0.05, 0.05],
        masses  = [
            *_fiducial_bpl(),
            *_fiducial_gauss_low(),
            *_fiducial_gauss_mid(),
            # high-mass PL tail: alpha, mmin, mmax, dmmin, dmmax
            3.0, 50.0, 100.0, 3.0, 10.0,
        ],
    ),

    # ----------------------------------------------------------------
    # twopowerlaws+peak
    # ----------------------------------------------------------------
    "twopowerlaws+peak": dict(
        n_comp  = 3,
        weights = [0.20, 0.10],
        masses  = [*_fiducial_pl(), *_fiducial_pl(), *_fiducial_gauss_mid()],
    ),

    # ----------------------------------------------------------------
    # twopowerlaws+2peaks
    # ----------------------------------------------------------------
    "twopowerlaws+2peaks": dict(
        n_comp  = 4,
        weights = [0.15, 0.10, 0.05],
        masses  = [*_fiducial_pl(), *_fiducial_pl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid()],
    ),

    # ----------------------------------------------------------------
    # twopowerlaws+3peaks
    # ----------------------------------------------------------------
    "twopowerlaws+3peaks": dict(
        n_comp  = 5,
        weights = [0.15, 0.10, 0.05, 0.03],
        masses  = [*_fiducial_pl(), *_fiducial_pl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid(), *_fiducial_gauss_high()],
    ),

    # ----------------------------------------------------------------
    # gp_mass   (k=1, no weight params)
    # ----------------------------------------------------------------
    "gp_mass": dict(
        n_comp  = 1,
        weights = [],
        masses  = _fiducial_gp_mass(),
    ),

    # ----------------------------------------------------------------
    # gp_mass_pairing   (k=1, no weight params)
    # ----------------------------------------------------------------
    "gp_mass_pairing": dict(
        n_comp  = 1,
        weights = [],
        masses  = _fiducial_gp_mass(),
        # pairing overridden below — 1D GP pairing instead of power law
        _pairing_override = _fiducial_gp_pairing(),
    ),

    # ----------------------------------------------------------------
    # gp_mass_pairing_joint   (k=1, no weight params)
    # ----------------------------------------------------------------
    "gp_mass_pairing_joint": dict(
        n_comp  = 1,
        weights = [],
        masses  = _fiducial_gp_mass(),
        _pairing_override = _fiducial_gp_pairing_2d(),
    ),
}


def get_fixed_population_params(pop_model: str) -> jnp.ndarray:
    """
    Return fiducial population parameters for --fix_population=True.

    The array ordering exactly matches PopulationModel.param_specs:
        weights → mass params → pairing params → spin params → gamma

    Parameters
    ----------
    pop_model : str
        Model name, optionally with _shared_beta / _shared_spin suffixes.

    Returns
    -------
    jnp.ndarray
        Flat parameter vector of the correct length for the model.
    """
    # --- Parse shared flags from the model name ---
    base_model = pop_model
    shared_beta = False
    shared_spin = False

    if "_shared_beta_spin" in base_model:
        base_model  = base_model.replace("_shared_beta_spin", "")
        shared_beta = True
        shared_spin = True
    else:
        if "_shared_beta" in base_model:
            base_model  = base_model.replace("_shared_beta", "")
            shared_beta = True
        if "_shared_spin" in base_model:
            base_model  = base_model.replace("_shared_spin", "")
            shared_spin = True

    if base_model not in _FIDUCIAL_PARAMS:
        raise ValueError(
            f"No fiducial parameters defined for model '{base_model}'. "
            f"Available: {sorted(_FIDUCIAL_PARAMS.keys())}."
        )

    spec = _FIDUCIAL_PARAMS[base_model]
    n_comp = spec["n_comp"]

    # --- Pairing fiducials ---
    # GP pairing models override the default power-law pairing values.
    if "_pairing_override" in spec:
        betas = spec["_pairing_override"]
    elif shared_beta:
        betas = _fiducial_pl_pairing(n=1)
    else:
        betas = _fiducial_pl_pairing(n=n_comp)

    # --- Spin fiducials ---
    spins = _fiducial_spin(n=1 if shared_spin else n_comp)

    # --- Assemble in required order ---
    full = spec["weights"] + spec["masses"] + betas + spins + [3.0]  # 3.0 = gamma
    return jnp.array(full, dtype=float)


# ============================================================
# 6. Public API
# ============================================================

def get_model(pop_model: str) -> PopulationModel:
    try:
        return _MODEL_REGISTRY[pop_model]
    except KeyError:
        raise ValueError(
            f"Unknown model {pop_model!r}. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )


def pop_model_parser(pop_model: str):
    """Return the log_p_pop function for the requested model."""
    return get_model(pop_model).log_p_pop


def pop_model_prior_parser(pop_model: str) -> tuple[list, list, list, str]:
    """Return (lower_bounds, upper_bounds, labels, latex_name) for the model."""
    model = get_model(pop_model)
    lows, highs, labels = model.prior_bounds()
    return lows, highs, labels, MODEL_NAME_LATEX.get(pop_model, pop_model)