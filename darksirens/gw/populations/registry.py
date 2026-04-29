import jax.numpy as jnp

from .base import ParamSpec, MixtureModel, PopulationModel
from .parametric import PowerLaw, BrokenPowerLaw, Gaussian, PowerLawPairing, TruncatedGaussianSpin
from .gp import GaussianProcessMass1D, GaussianProcessMassRatio1D, GaussianProcessPairing2D


def _pl(
    alpha_label=r"$\alpha$", mmin_label=r"$m_{\min}$", mmax_label=r"$m_{\max}$",
    dmmin_label=r"$dm_{\min}$", dmmax_label=r"$dm_{\max}$",
    alpha_lo=-4.0, alpha_hi=6.0, mmin_lo=2.0, mmin_hi=10.0,
    mmax_lo=50.0, mmax_hi=100.0, dmmin_lo=0.01, dmmin_hi=10.0,
    dmmax_lo=0.01, dmmax_hi=20.0
):
    return PowerLaw(
        ParamSpec(alpha_label, alpha_lo, alpha_hi),
        ParamSpec(mmin_label, mmin_lo, mmin_hi),
        ParamSpec(mmax_label, mmax_lo, mmax_hi),
        ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        ParamSpec(dmmax_label, dmmax_lo, dmmax_hi)
    )

def _bpl(
    alpha1_label=r"$\alpha_1$", alpha2_label=r"$\alpha_2$", break_label=r"$m_{\rm break}$",
    mmin_label=r"$m_{\min}$", mmax_label=r"$m_{\max}$", dmmin_label=r"$dm_{\min}$",
    dmmax_label=r"$dm_{\max}$",
    a1_lo=0.0, a1_hi=6.0, a2_lo=0.0, a2_hi=6.0, brk_lo=20.0, brk_hi=50.0,
    mmin_lo=2.0, mmin_hi=10.0, mmax_lo=40.0, mmax_hi=200.0,
    dmmin_lo=0.01, dmmin_hi=100.0, dmmax_lo=0.01, dmmax_hi=100.0
):
    return BrokenPowerLaw(
        ParamSpec(alpha1_label, a1_lo, a1_hi),
        ParamSpec(alpha2_label, a2_lo, a2_hi),
        ParamSpec(break_label, brk_lo, brk_hi),
        ParamSpec(mmin_label, mmin_lo, mmin_hi),
        ParamSpec(mmax_label, mmax_lo, mmax_hi),
        ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        ParamSpec(dmmax_label, dmmax_lo, dmmax_hi)
    )

def _gauss(
    mu_lo, mu_hi, sig_lo=1.0, sig_hi=10.0,
    mu_label=r"$\mu$", sig_label=r"$\sigma$"
):
    return Gaussian(
        ParamSpec(mu_label, mu_lo, mu_hi),
        ParamSpec(sig_label, sig_lo, sig_hi)
    )

def _plpairing(beta_label=r"$\beta$", beta_lo=-2.0, beta_hi=7.0):
    return PowerLawPairing(
        ParamSpec(beta_label, beta_lo, beta_hi)
    )

def _spin(
    mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$",
    mu_lo=-1.0, mu_hi=1.0, sig_lo=0.01, sig_hi=1.0
):
    return TruncatedGaussianSpin(
        ParamSpec(mu_label, mu_lo, mu_hi),
        ParamSpec(sig_label, sig_lo, sig_hi)
    )

def _gp_mass(
    mmin_label=r"$m_{\min}$", mmax_label=r"$m_{\max}$",
    dmmin_label=r"$\delta m_{\min}$", dmmax_label=r"$\delta m_{\max}$",
    alpha_label=r"$\alpha$", amp_label=r"$A$", ls_label=r"$l$",
    y_labels=[r"$y_{%d}$" % i for i in range(11)],
    mmin_lo=2.0, mmin_hi=10.0, mmax_lo=50.0, mmax_hi=100.0,
    dmmin_lo=0.01, dmmin_hi=10.0, dmmax_lo=0.01, dmmax_hi=20.0,
    alpha_lo=-4.0, alpha_hi=12.0, amp_lo=0.01, amp_hi=5.0,
    ls_lo=0.05, ls_hi=1.0, y_lo=-7.0, y_hi=7.0
):
    return GaussianProcessMass1D(
        ParamSpec(mmin_label, mmin_lo, mmin_hi),
        ParamSpec(mmax_label, mmax_lo, mmax_hi),
        ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        ParamSpec(dmmax_label, dmmax_lo, dmmax_hi),
        ParamSpec(alpha_label, alpha_lo, alpha_hi),
        ParamSpec(amp_label, amp_lo, amp_hi),
        ParamSpec(ls_label, ls_lo, ls_hi),
        ParamSpec(y_labels[0], y_lo, y_hi), ParamSpec(y_labels[1], y_lo, y_hi),
        ParamSpec(y_labels[2], y_lo, y_hi), ParamSpec(y_labels[3], y_lo, y_hi),
        ParamSpec(y_labels[4], y_lo, y_hi), ParamSpec(y_labels[5], y_lo, y_hi),
        ParamSpec(y_labels[6], y_lo, y_hi), ParamSpec(y_labels[7], y_lo, y_hi),
        ParamSpec(y_labels[8], y_lo, y_hi), ParamSpec(y_labels[9], y_lo, y_hi),
        ParamSpec(y_labels[10], y_lo, y_hi)
    )

def _gp_pairing(
    beta_label=r"$\beta_q$", amp_label=r"$A_q$", ls_label=r"$l_q$",
    y_labels=[r"$y_{q,%d}$" % i for i in range(5)],
    beta_lo=-4.0, beta_hi=12.0, amp_lo=0.01, amp_hi=5.0,
    ls_lo=0.05, ls_hi=1.0, y_lo=-7.0, y_hi=7.0
):
    return GaussianProcessMassRatio1D(
        ParamSpec(beta_label, beta_lo, beta_hi),
        ParamSpec(amp_label, amp_lo, amp_hi),
        ParamSpec(ls_label, ls_lo, ls_hi),
        ParamSpec(y_labels[0], y_lo, y_hi),
        ParamSpec(y_labels[1], y_lo, y_hi),
        ParamSpec(y_labels[2], y_lo, y_hi),
        ParamSpec(y_labels[3], y_lo, y_hi),
        ParamSpec(y_labels[4], y_lo, y_hi)
    )

def _gp_mass_pairing(
    beta_label=r"$\beta$", amp_label=r"$A_q$", ls_m_label=r"$l_{m,q}$", ls_q_label=r"$l_{q,q}$",
    y_labels=[r"$y_{q,%d}$" % i for i in range(16)],
    beta_lo=-4.0, beta_hi=12.0, amp_lo=0.01, amp_hi=3.0,
    ls_m_lo=0.5, ls_m_hi=3.0, ls_q_lo=0.2, ls_q_hi=1.0,
    y_lo=-5.0, y_hi=5.0
):
    return GaussianProcessPairing2D(
        ParamSpec(beta_label, beta_lo, beta_hi),
        ParamSpec(amp_label, amp_lo, amp_hi),
        ParamSpec(ls_m_label, ls_m_lo, ls_m_hi),
        ParamSpec(ls_q_label, ls_q_lo, ls_q_hi),
        ParamSpec(y_labels[0], y_lo, y_hi), ParamSpec(y_labels[1], y_lo, y_hi),
        ParamSpec(y_labels[2], y_lo, y_hi), ParamSpec(y_labels[3], y_lo, y_hi),
        ParamSpec(y_labels[4], y_lo, y_hi), ParamSpec(y_labels[5], y_lo, y_hi),
        ParamSpec(y_labels[6], y_lo, y_hi), ParamSpec(y_labels[7], y_lo, y_hi),
        ParamSpec(y_labels[8], y_lo, y_hi), ParamSpec(y_labels[9], y_lo, y_hi),
        ParamSpec(y_labels[10], y_lo, y_hi), ParamSpec(y_labels[11], y_lo, y_hi),
        ParamSpec(y_labels[12], y_lo, y_hi), ParamSpec(y_labels[13], y_lo, y_hi),
        ParamSpec(y_labels[14], y_lo, y_hi), ParamSpec(y_labels[15], y_lo, y_hi)
    )

def _mixture_plpeak(shared_beta=False, shared_spin=False):
    masses = [
        _pl(),
        _gauss(20, 50)
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [_plpairing(beta_label=r"$\beta_{\rm PL}$"), _plpairing(beta_label=r"$\beta_{\rm G}$")]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,\rm PL}$", sig_label=r"$\sigma_{\chi,\rm PL}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G}$", sig_label=r"$\sigma_{\chi,\rm G}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_bpl2peaks(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20, mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$")
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_{\rm BPL}$"),
            _plpairing(beta_label=r"$\beta_{\rm G1}$"),
            _plpairing(beta_label=r"$\beta_{\rm G2}$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G1}$", sig_label=r"$\sigma_{\chi,\rm G1}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G2}$", sig_label=r"$\sigma_{\chi,\rm G2}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_bpl3peaks(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20, mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$"),
        _gauss(50, 100, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", sig_hi=20)
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_{\rm BPL}$"),
            _plpairing(beta_label=r"$\beta_{\rm G1}$"),
            _plpairing(beta_label=r"$\beta_{\rm G2}$"),
            _plpairing(beta_label=r"$\beta_{\rm G3}$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G1}$", sig_label=r"$\sigma_{\chi,\rm G1}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G2}$", sig_label=r"$\sigma_{\chi,\rm G2}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G3}$", sig_label=r"$\sigma_{\chi,\rm G3}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_2pl1peak(shared_beta=False, shared_spin=False):
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$",
            dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$",
            dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100
        ),
        _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$")
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_1$"),
            _plpairing(beta_label=r"$\beta_2$"),
            _plpairing(beta_label=r"$\beta_3$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
            _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
            _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_2pl2peaks(shared_beta=False, shared_spin=False):
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$",
            dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$",
            dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100
        ),
        _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$"),
        _gauss(20, 50, sig_hi=15, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$")
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_1$"),
            _plpairing(beta_label=r"$\beta_2$"),
            _plpairing(beta_label=r"$\beta_3$"),
            _plpairing(beta_label=r"$\beta_4$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
            _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
            _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$"),
            _spin(mu_label=r"$\mu_{\chi,4}$", sig_label=r"$\sigma_{\chi,4}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_2pl3peaks(shared_beta=False, shared_spin=False):
    masses = [
        _pl(
            alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$", mmax_label=r"$m_{\max,1}$",
            dmmin_label=r"$dm_{\min,1}$", dmmax_label=r"$dm_{\max,1}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=2, mmin_hi=10, mmax_lo=15, mmax_hi=50
        ),
        _pl(
            alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$", mmax_label=r"$m_{\max,2}$",
            dmmin_label=r"$dm_{\min,2}$", dmmax_label=r"$dm_{\max,2}$",
            alpha_lo=0, alpha_hi=6, mmin_lo=20, mmin_hi=40, mmax_lo=50, mmax_hi=100
        ),
        _gauss(5, 20, sig_hi=10, mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$"),
        _gauss(20, 50, sig_hi=10, mu_label=r"$\mu_4$", sig_label=r"$\sigma_4$"),
        _gauss(50, 100, sig_hi=20, mu_label=r"$\mu_5$", sig_label=r"$\sigma_5$")
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_1$"),
            _plpairing(beta_label=r"$\beta_2$"),
            _plpairing(beta_label=r"$\beta_3$"),
            _plpairing(beta_label=r"$\beta_4$"),
            _plpairing(beta_label=r"$\beta_5$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,1}$", sig_label=r"$\sigma_{\chi,1}$"),
            _spin(mu_label=r"$\mu_{\chi,2}$", sig_label=r"$\sigma_{\chi,2}$"),
            _spin(mu_label=r"$\mu_{\chi,3}$", sig_label=r"$\sigma_{\chi,3}$"),
            _spin(mu_label=r"$\mu_{\chi,4}$", sig_label=r"$\sigma_{\chi,4}$"),
            _spin(mu_label=r"$\mu_{\chi,5}$", sig_label=r"$\sigma_{\chi,5}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_bpl2peaks1pl(shared_beta=False, shared_spin=False):
    masses = [
        _bpl(),
        _gauss(5, 20, mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$"),
        _gauss(25, 40, mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$"),
        _pl(
            alpha_label=r"$\alpha_{\rm PL}$", mmin_label=r"$m_{\min,\rm PL}$",
            mmax_label=r"$m_{\max,\rm PL}$", dmmin_label=r"$dm_{\min,\rm PL}$",
            dmmax_label=r"$dm_{\max,\rm PL}$", alpha_lo=0, alpha_hi=6,
            mmin_lo=40, mmin_hi=60, mmax_lo=80, mmax_hi=120
        )
    ]
    pairings = (
        [_plpairing(beta_label=r"$\beta$")] if shared_beta else
        [
            _plpairing(beta_label=r"$\beta_{\rm BPL}$"),
            _plpairing(beta_label=r"$\beta_{\rm G1}$"),
            _plpairing(beta_label=r"$\beta_{\rm G2}$"),
            _plpairing(beta_label=r"$\beta_{\rm PL}$")
        ]
    )
    spins = (
        [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")] if shared_spin else
        [
            _spin(mu_label=r"$\mu_{\chi,\rm BPL}$", sig_label=r"$\sigma_{\chi,\rm BPL}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G1}$", sig_label=r"$\sigma_{\chi,\rm G1}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm G2}$", sig_label=r"$\sigma_{\chi,\rm G2}$"),
            _spin(mu_label=r"$\mu_{\chi,\rm PL}$", sig_label=r"$\sigma_{\chi,\rm PL}$")
        ]
    )
    return MixtureModel(masses, pairings, spins)

def _mixture_gp_mass(shared_beta=True, shared_spin=True):
    masses = [_gp_mass()]
    pairings = [_plpairing(beta_label=r"$\beta$")]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    return MixtureModel(masses, pairings, spins)

def _mixture_gp_mass_pairing(shared_beta=True, shared_spin=True):
    masses = [_gp_mass()]
    pairings = [_gp_pairing()]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    return MixtureModel(masses, pairings, spins)

def _mixture_gp_mass_pairing_joint(shared_beta=True, shared_spin=True):
    masses = [_gp_mass()]
    pairings = [_gp_mass_pairing()]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    return MixtureModel(masses, pairings, spins)

def _make(mix_fn, shared_beta=False, shared_spin=False):
    return PopulationModel(mixture=mix_fn(shared_beta=shared_beta, shared_spin=shared_spin))

_RAW_MODELS = {
    "powerlaw+peak":                   (_mixture_plpeak,                  "PL+G"),
    "brokenpowerlaw+2peaks":           (_mixture_bpl2peaks,               "BPL+2G"),
    "brokenpowerlaw+3peaks":           (_mixture_bpl3peaks,               "BPL+3G"),
    "brokenpowerlaw+2peaks+powerlaw":  (_mixture_bpl2peaks1pl,            "BPL+2G+PL"),
    "twopowerlaws+peak":               (_mixture_2pl1peak,                "2PL+G"),
    "twopowerlaws+2peaks":             (_mixture_2pl2peaks,               "2PL+2G"),
    "twopowerlaws+3peaks":             (_mixture_2pl3peaks,               "2PL+3G"),
    "gp_mass":                         (_mixture_gp_mass,                 "GP"),
    "gp_mass_pairing":                 (_mixture_gp_mass_pairing,         "GPxGP"),
    "gp_mass_pairing_joint":           (_mixture_gp_mass_pairing_joint,   "GP 2D"),
}

_MODEL_REGISTRY: dict[str, PopulationModel] = {}
MODEL_NAME_LATEX: dict[str, str] = {"mock_data": r"\text{Mock}"}

for name, (mix_fn, latex_name) in _RAW_MODELS.items():
    def bind_fn(f=mix_fn, sb=False, ss=False):
        return lambda: _make(f, shared_beta=sb, shared_spin=ss)

    _MODEL_REGISTRY[name] = bind_fn(sb=False, ss=False)()
    MODEL_NAME_LATEX[name] = latex_name

    name_sb = f"{name}_shared_beta"
    _MODEL_REGISTRY[name_sb] = bind_fn(sb=True, ss=False)()
    MODEL_NAME_LATEX[name_sb] = f"{latex_name} (Shared $\\beta$)"

    name_ss = f"{name}_shared_spin"
    _MODEL_REGISTRY[name_ss] = bind_fn(sb=False, ss=True)()
    MODEL_NAME_LATEX[name_ss] = f"{latex_name} (Shared Spin)"

    name_both = f"{name}_shared_beta_spin"
    _MODEL_REGISTRY[name_both] = bind_fn(sb=True, ss=True)()
    MODEL_NAME_LATEX[name_both] = f"{latex_name} (Shared $\\beta$, Spin)"

def get_model(pop_model: str) -> PopulationModel:
    try:
        return _MODEL_REGISTRY[pop_model]
    except KeyError:
        raise ValueError(f"Unknown model {pop_model!r}. Available: {sorted(_MODEL_REGISTRY.keys())}")

def pop_model_parser(pop_model: str):
    return get_model(pop_model).log_p_pop

def pop_model_prior_parser(pop_model: str) -> tuple[list, list, list, str]:
    model = get_model(pop_model)
    lows, highs, labels = model.prior_bounds()
    return lows, highs, labels, MODEL_NAME_LATEX.get(pop_model, pop_model)

def get_fixed_population_params(pop_model):
    """
    Fiducial population parameters used when --fix_population=True.
    These must match the exact dimension and component ordering of the backend parser:
    Weights -> Masses -> Betas -> Spins -> Gamma.
    """
    
    # 1. Parse the requested model for shared flags
    base_model = pop_model
    if "_shared_beta_spin" in base_model:
        base_model = base_model.replace("_shared_beta_spin", "")
        shared_beta = True
        shared_spin = True
    else:
        shared_beta = "_shared_beta" in base_model
        shared_spin = "_shared_spin" in base_model
        base_model = base_model.replace("_shared_beta", "").replace("_shared_spin", "")

    # 2. Define Weights and Mass parameters per base model
    if base_model == "powerlaw+peak":
        n_comp = 2
        weights = [0.1]
        masses = [
            2.0, 5.0, 80.0, 0.5, 10.0,  # PL: alpha, m_min, m_max, dm_min, dm_max
            35.0, 5.0                   # G: mu, sigma
        ]
        
    elif base_model == "brokenpowerlaw+2peaks":
        n_comp = 3
        weights = [0.10, 0.05]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL: a1, a2, m_break, m_min, m_max, dm_min, dm_max
            10.0, 3.0,                            # G1: mu1, sigma1
            35.0, 5.0                             # G2: mu2, sigma2
        ]
        
    elif base_model == "brokenpowerlaw+3peaks":
        n_comp = 4
        weights = [0.10, 0.05, 0.03]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL
            10.0, 3.0,                            # G1
            35.0, 5.0,                            # G2
            70.0, 10.0                            # G3: mu3, sigma3
        ]
        
    elif base_model == "twopowerlaws+peak":
        n_comp = 3
        weights = [0.20, 0.10]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            35.0, 5.0                  # G
        ]
        
    elif base_model == "twopowerlaws+2peaks":
        n_comp = 4
        weights = [0.15, 0.10, 0.05]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            10.0, 3.0,                 # G1
            35.0, 5.0                  # G2
        ]
        
    elif base_model == "twopowerlaws+3peaks":
        n_comp = 5
        weights = [0.15, 0.10, 0.05, 0.03]
        masses = [
            2.0, 5.0, 80.0, 3.0, 10.0, # PL1
            4.0, 5.0, 80.0, 3.0, 10.0, # PL2
            10.0, 3.0,                 # G1
            35.0, 5.0,                 # G2
            70.0, 10.0                 # G3
        ]
        
    elif base_model == "brokenpowerlaw+2peaks+powerlaw":
        n_comp = 4
        weights = [0.10, 0.05, 0.05]
        masses = [
            2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0, # BPL
            10.0, 3.0,                            # G1
            35.0, 5.0,                            # G2
            3.0, 50.0, 100.0, 3.0, 10.0           # PL
        ]

    elif base_model == "gp_mass":
        n_comp = 1
        weights = []  # k=1 means no weight parameters
        masses = [
            5.0, 80.0, 3.0, 10.0, # m_min, m_max, dm_min, dm_max
            2.3,                  # alpha (Fiducial power-law slope resembling GWTC-3)
            1.0, 1.0,             # amp, ls (Fiducial kernel variance and length scale)
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0 # y0, y1, y2, y3, y4 (Zero deviations = pure power-law initially)
        ]
        
    elif base_model == "gp_mass_pairing":
        n_comp = 1
        weights = []  # k=1 means no weight parameters
        masses = [
            5.0, 80.0, 3.0, 10.0, # m_min, m_max, dm_min, dm_max
            2.3,                  # alpha (Fiducial power-law slope resembling GWTC-3)
            1.0, 1.0,             # amp_m, ls_m (Fiducial kernel variance and length scale)
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0                   # y0-y10 (Zero deviations = pure power-law initially)
        ]

    elif base_model == "gp_mass_pairing_joint":
        n_comp = 1
        weights = []  # k=1 means no weight parameters
        masses = [
            5.0, 80.0, 3.0, 10.0, # m_min, m_max, dm_min, dm_max
            2.3,                  # alpha (Fiducial power-law slope resembling GWTC-3)
            1.0, 1.0,             # amp, ls (Fiducial kernel variance and length scale)
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0    # y0-y13 (Zero deviations = pure power-law initially)
        ]

    else:
        raise ValueError(f"No fixed parameters defined for model '{pop_model}'")
        
    # Assuming you have an if/elif chain for betas below the masses:
    
    if base_model == "gp_mass_pairing":
        betas = [
            1.5,                  # beta_q (Fiducial slope favoring equal mass)
            1.0, 0.5,             # amp_q, ls_q (ls=0.5 covers half the (0,1] domain)
            0.0, 0.0, 0.0, 0.0, 0.0 # y_q0 - y_q4 (Zero deviations = pure q^beta initially)
        ]
    elif base_model == "gp_mass_pairing_joint":
        betas = [
            0.0,           # beta (baseline flat mass ratio preference)
            1.0,           # amp_q (amplitude of GP deviations)
            1.0, 0.5,      # ls_m, ls_q (correlation lengths for m1 and q)
            # 16 nodes for the 4x4 grid
            0.0, 0.0, 0.0, 0.0, # y0 - y3
            0.0, 0.0, 0.0, 0.0, # y4 - y7
            0.0, 0.0, 0.0, 0.0, # y8 - y11
            0.0, 0.0, 0.0, 0.0  # y12 - y15
        ]

    else:
        # 3. Define Pairing (Beta) Parameters
        betas = [1.0] if shared_beta else [1.0] * n_comp
    
    # 4. Define Spin Parameters (mu_chi=0.0, sigma_chi=0.1)
    spins = [0.0, 0.1] if shared_spin else [0.0, 0.1] * n_comp
    
    # 5. Define Gamma
    gamma = [3.0]

    # Assemble global array adhering to unified strict ordering
    full_params = weights + masses + betas + spins + gamma
    return jnp.array(full_params)