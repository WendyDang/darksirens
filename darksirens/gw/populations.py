from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp
from jax import jit

import tinygp
from tinygp import GaussianProcess, kernels

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


@dataclass
class GaussianProcessMass(MassComponent):
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec
    amp1_spec: ParamSpec
    amp2_spec: ParamSpec
    ls1_spec: ParamSpec
    ls2_spec: ParamSpec
    n_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
            self.amp1_spec,
            self.amp2_spec,
            self.ls1_spec,
            self.ls2_spec,
            self.n_spec,
        ]

    def _eval_unnorm(self, m, t):
        mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3]
        amp1, amp2, ls1, ls2, n = t[4], t[5], t[6], t[7], t[8]

        # Deterministic key from theta parameter
        key = jax.random.PRNGKey(n.astype(jnp.int32))

        # Build Kernel
        kernel_1 = (amp1**2) * kernels.ExpSquared(scale=ls1)
        kernel_2 = (amp2**2) * kernels.Matern52(scale=ls2)
        kernel = kernel_1 + kernel_2 
        
        # Instantiate and sample GP directly on MASS_GRID
        gp = GaussianProcess(kernel=kernel, X=MASS_GRID, mean=0.0, diag=0.001)
        logpm_grid = gp.sample(key, shape=())
        
        # Apply smoothing filters on the grid
        S_grid = sfilter_low(MASS_GRID, mmin, dmmin) * sfilter_high(MASS_GRID, mmax, dmmax)
        pm_grid = S_grid * jnp.exp(logpm_grid)
        
        # Interpolate the fixed grid evaluation to the requested m array
        return jnp.interp(m, MASS_GRID, pm_grid, left=0.0, right=0.0)
    

@dataclass
class GaussianProcessMassExp(MassComponent):
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec
    alpha_spec: ParamSpec
    amp_spec: ParamSpec
    ls_spec: ParamSpec
    # 11 Latent nodes for high-resolution mass features
    y0_spec: ParamSpec
    y1_spec: ParamSpec
    y2_spec: ParamSpec
    y3_spec: ParamSpec
    y4_spec: ParamSpec
    y5_spec: ParamSpec
    y6_spec: ParamSpec
    y7_spec: ParamSpec
    y8_spec: ParamSpec
    y9_spec: ParamSpec
    y10_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
            self.alpha_spec,
            self.amp_spec,
            self.ls_spec,
            self.y0_spec,
            self.y1_spec,
            self.y2_spec,
            self.y3_spec,
            self.y4_spec,
            self.y5_spec,
            self.y6_spec,
            self.y7_spec,
            self.y8_spec,
            self.y9_spec,
            self.y10_spec,
        ]

    def _eval_unnorm(self, m, t):
        """
        Evaluates the unnormalized primary mass distribution.
        t: array of parameters ordered according to param_specs.
        """
        # 1. Unpack smoothing and hyper-parameters
        mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3]
        alpha, amp, ls = t[4], t[5], t[6]
        
        # 2. Extract the 11 latent nodes (y0 through y10)
        y_nodes = jnp.array(t[7:18]) 

        # 3. Setup Log-Space Grid and Nodes
        # We perform the GP math in log-space for fractional stationarity
        log_MASS_GRID = jnp.log(MASS_GRID)
        log_nodes = jnp.linspace(jnp.log(2.0), jnp.log(100.0), 11)

        # 4. Define Mean and Kernel
        # The mean function anchors the GP to a baseline power-law
        def mean_fn(x):
            return -alpha * x

        # Matern-5/2 is twice-differentiable; ideal for "smoothly bumpy" mass functions
        kernel = (amp**2) * kernels.Matern52(scale=ls)

        # 5. Build GP Prior and Condition on the Latent Nodes
        # diag=1e-5 ensures numerical stability during the Cholesky decomposition
        gp_prior = GaussianProcess(kernel=kernel, X=log_nodes, mean=mean_fn, diag=1e-5)
        _, gp_cond = gp_prior.condition(y_nodes, log_MASS_GRID)
        
        # This is our log-probability density across the fixed MASS_GRID
        logpm_grid = gp_cond.loc

        # 6. Apply smoothing filters (Low-mass and High-mass cutoffs)
        S_grid = sfilter_low(MASS_GRID, mmin, dmmin) * sfilter_high(MASS_GRID, mmax, dmmax)
        pm_grid = S_grid * jnp.exp(logpm_grid)

        # 7. Interpolate the grid-calculated density to the specific requested m-values
        return jnp.interp(m, MASS_GRID, pm_grid, left=0.0, right=0.0)
    


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


@dataclass
class GaussianProcessMassRatio1D(PairingModel):
    beta_spec: ParamSpec
    amp_spec: ParamSpec
    ls_spec: ParamSpec
    
    # Reduced to 5 Latent nodes to prevent overfitting 
    # while capturing power-law + Gaussian peak features
    y0_spec: ParamSpec
    y1_spec: ParamSpec
    y2_spec: ParamSpec
    y3_spec: ParamSpec
    y4_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.beta_spec, self.amp_spec, self.ls_spec,
            self.y0_spec, self.y1_spec, self.y2_spec, 
            self.y3_spec, self.y4_spec
        ]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        # 1. Unpack hyper-parameters
        beta, amp, ls = t[0], t[1], t[2]
        
        # Extract the 5 nodes
        y_nodes = jnp.array(t[3:8])

        # 2. Setup Nodes and Evaluation Grid for q
        # 5 nodes evenly spaced across the bulk of the q domain
        q_nodes = jnp.linspace(0.1, 1.0, 5)
        
        q_eval_grid = jnp.linspace(0.01, 1.0, 500)

        # 3. Define Mean Function and Kernel
        def mean_fn(x):
            return beta * jnp.log(x)

        kernel = (amp**2) * kernels.Matern52(scale=ls)

        # 4. Build GP Prior and Condition on Latent Nodes
        gp_prior = GaussianProcess(kernel=kernel, X=q_nodes, mean=mean_fn, diag=1e-4)
        _, gp_cond = gp_prior.condition(y_nodes, q_eval_grid)
        
        # 5. Extract and safe-guard the grid predictions
        log_pq_grid = jnp.nan_to_num(gp_cond.loc, nan=-20.0)
        log_pq_grid = jnp.clip(log_pq_grid, -10.0, 10.0)
        pq_grid = jnp.exp(log_pq_grid)

        # 6. Interpolate to the requested q values
        pq = jnp.interp(q, q_eval_grid, pq_grid, left=0.0, right=0.0)

        # 7. Apply physical cutoffs (m2 > m_min)
        m1_b, q_b = jnp.broadcast_arrays(m1, q)
        m2 = q_b * m1_b
        
        smooth_cut = sfilter_low(m2, m_min, dm_min)
        smooth_cut = jnp.nan_to_num(smooth_cut, nan=0.0)
        
        p = smooth_cut * pq
        
        return jnp.where(m2 < m_min, 1e-20, p)


@dataclass
class GaussianProcessPairing2D(PairingModel):
    beta_spec: ParamSpec
    amp_spec: ParamSpec
    ls_m_spec: ParamSpec
    ls_q_spec: ParamSpec
    
    # 16 Latent nodes for a 4x4 grid across (m1, q)
    y0_spec: ParamSpec;  y1_spec: ParamSpec;  y2_spec: ParamSpec;  y3_spec: ParamSpec
    y4_spec: ParamSpec;  y5_spec: ParamSpec;  y6_spec: ParamSpec;  y7_spec: ParamSpec
    y8_spec: ParamSpec;  y9_spec: ParamSpec;  y10_spec: ParamSpec; y11_spec: ParamSpec
    y12_spec: ParamSpec; y13_spec: ParamSpec; y14_spec: ParamSpec; y15_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.beta_spec, self.amp_spec, self.ls_m_spec, self.ls_q_spec,
            self.y0_spec,  self.y1_spec,  self.y2_spec,  self.y3_spec,
            self.y4_spec,  self.y5_spec,  self.y6_spec,  self.y7_spec,
            self.y8_spec,  self.y9_spec,  self.y10_spec, self.y11_spec,
            self.y12_spec, self.y13_spec, self.y14_spec, self.y15_spec
        ]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        # 1. Unpack parameters
        beta, amp, ls_m, ls_q = t[0], t[1], t[2], t[3]
        y_nodes = jnp.array(t[4:20])

        # 2. Setup the 4x4 Latent Node Grid
        log_m1_nodes = jnp.log(jnp.array([10.0, 20.0, 40.0, 80.0]))
        q_nodes = jnp.array([0.20, 0.40, 0.60, 0.80])
        
        M_node, Q_node = jnp.meshgrid(log_m1_nodes, q_nodes, indexing="ij")
        X_nodes = jnp.stack([M_node.flatten(), Q_node.flatten()], axis=-1)

        # 3. Setup the Target Grid 
        shape = jnp.broadcast_shapes(jnp.shape(m1), jnp.shape(q))
        m1_b, q_b = jnp.broadcast_arrays(m1, q)
        
        X_test = jnp.stack([jnp.log(m1_b), q_b], axis=-1)
        X_test_flat = X_test.reshape(-1, 2)

        # Scale coordinates manually for ARD
        scales = jnp.array([ls_m, ls_q])
        X_nodes_scaled = X_nodes / scales
        X_test_scaled = X_test_flat / scales

        # 4. Define Mean Function and Kernel
        def mean_fn(x_scaled):
            q_true = x_scaled[..., 1] * ls_q
            return beta * jnp.log(q_true + 0.01)

        kernel = (amp**2) * kernels.Matern52()

        # 5. Condition and Predict 
        # FIX 1: Aggressive jitter (1e-2 instead of 1e-4) to prevent Cholesky failure
        gp_prior = GaussianProcess(kernel=kernel, X=X_nodes_scaled, mean=mean_fn, diag=1e-2)
        _, gp_cond = gp_prior.condition(y_nodes, X_test_scaled)
        
        # FIX 2: Intercept any remaining GP NaNs and force them to a highly negative log-prob
        safe_loc = jnp.nan_to_num(gp_cond.loc, nan=-20.0)
        
        log_pq = jnp.clip(safe_loc.reshape(shape), -10.0, 10.0)
        p = jnp.exp(log_pq)

        # 6. Apply physical cutoffs
        m2 = q_b * m1_b
        
        # FIX 3: Intercept division-by-zero NaNs if dm_min hits 0.0
        smooth_cut = sfilter_low(m2, m_min, dm_min)
        smooth_cut = jnp.nan_to_num(smooth_cut, nan=0.0) 
        
        p = smooth_cut * p
        
        return jnp.where(m2 < m_min, 1e-20, p)
    

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


def _gp(
    mmin_label=r"$m_{\min}$", mmax_label=r"$m_{\max}$",
    dmmin_label=r"$dm_{\min}$", dmmax_label=r"$dm_{\max}$",
    amp1_label=r"$A_1$", amp2_label=r"$A_2$",
    ls1_label=r"$l_1$", ls2_label=r"$l_2$",
    n_label=r"$n_{\rm seed}$", # Label clearly for the corner plot
    mmin_lo=2.0, mmin_hi=10.0,
    mmax_lo=50.0, mmax_hi=100.0,
    dmmin_lo=0.01, dmmin_hi=10.0,
    dmmax_lo=0.01, dmmax_hi=20.0,
    amp_lo=0.1, amp_hi=5.0,
    ls_lo=1.0, ls_hi=20.0,
    n_lo=0.0, n_hi=1e6 # Bounded range for the PRNG seed
):
    return GaussianProcessMass(
        m_min_spec  = ParamSpec(mmin_label,  mmin_lo,  mmin_hi),
        m_max_spec  = ParamSpec(mmax_label,  mmax_lo,  mmax_hi),
        dm_min_spec = ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        dm_max_spec = ParamSpec(dmmax_label, dmmax_lo, dmmax_hi),
        amp1_spec   = ParamSpec(amp1_label,  amp_lo,   amp_hi),
        amp2_spec   = ParamSpec(amp2_label,  amp_lo,   amp_hi),
        ls1_spec    = ParamSpec(ls1_label,   ls_lo,    ls_hi),
        ls2_spec    = ParamSpec(ls2_label,   ls_lo,    ls_hi),
        n_spec      = ParamSpec(n_label,     n_lo,     n_hi),
    )


def _gp_experimental(
    mmin_label=r"$m_{\min}$", mmax_label=r"$m_{\max}$",
    dmmin_label=r"$\delta m_{\min}$", dmmax_label=r"$\delta m_{\max}$",
    alpha_label=r"$\alpha$", amp_label=r"$A$", ls_label=r"$l$",
    # 11 Latent Node Labels
    y_labels=[r"$y_{%d}$" % i for i in range(11)],
    
    # Smoothing & Baseline Bounds
    mmin_lo=2.0, mmin_hi=10.0,
    mmax_lo=50.0, mmax_hi=100.0,
    dmmin_lo=0.01, dmmin_hi=10.0,
    dmmax_lo=0.01, dmmax_hi=20.0,
    alpha_lo=-4.0, alpha_hi=12.0, 
    
    # Updated Hyperparameter Bounds
    amp_lo=0.01, amp_hi=5.0,
    ls_lo=0.05, ls_hi=1.0,  # Tightened: prevents features from being forced too wide
    y_lo=-7.0, y_hi=7.0     # Generous vertical range for log-density deviations
):
    return GaussianProcessMassExp(
        m_min_spec  = ParamSpec(mmin_label,  mmin_lo,  mmin_hi),
        m_max_spec  = ParamSpec(mmax_label,  mmax_lo,  mmax_hi),
        dm_min_spec = ParamSpec(dmmin_label, dmmin_lo, dmmin_hi),
        dm_max_spec = ParamSpec(dmmax_label, dmmax_lo, dmmax_hi),
        alpha_spec  = ParamSpec(alpha_label, alpha_lo, alpha_hi),
        amp_spec    = ParamSpec(amp_label,   amp_lo,   amp_hi),
        ls_spec     = ParamSpec(ls_label,    ls_lo,    ls_hi),
        # Unpacking the 11 node specs
        y0_spec     = ParamSpec(y_labels[0],  y_lo, y_hi),
        y1_spec     = ParamSpec(y_labels[1],  y_lo, y_hi),
        y2_spec     = ParamSpec(y_labels[2],  y_lo, y_hi),
        y3_spec     = ParamSpec(y_labels[3],  y_lo, y_hi),
        y4_spec     = ParamSpec(y_labels[4],  y_lo, y_hi),
        y5_spec     = ParamSpec(y_labels[5],  y_lo, y_hi),
        y6_spec     = ParamSpec(y_labels[6],  y_lo, y_hi),
        y7_spec     = ParamSpec(y_labels[7],  y_lo, y_hi),
        y8_spec     = ParamSpec(y_labels[8],  y_lo, y_hi),
        y9_spec     = ParamSpec(y_labels[9],  y_lo, y_hi),
        y10_spec    = ParamSpec(y_labels[10], y_lo, y_hi),
    )

def _gp_pairing_experimental(
    beta_label=r"$\beta_q$", amp_label=r"$A_q$", ls_label=r"$l_q$",
    y_labels=[r"$y_{q,%d}$" % i for i in range(5)],
    
    # Hyperparameter Bounds
    beta_lo=-4.0, beta_hi=12.0,
    amp_lo=0.01, amp_hi=5.0,
    ls_lo=0.05, ls_hi=1.0,  # Domain is 0 to 1, so ls=1 spans the whole domain smoothly
    y_lo=-7.0, y_hi=7.0     # Same generous vertical range for log-density
):
    return GaussianProcessMassRatio1D(
        beta_spec = ParamSpec(beta_label, beta_lo, beta_hi),
        amp_spec  = ParamSpec(amp_label,  amp_lo,  amp_hi),
        ls_spec   = ParamSpec(ls_label,   ls_lo,   ls_hi),
        # Unpacking the 5 node specs
        y0_spec   = ParamSpec(y_labels[0], y_lo, y_hi),
        y1_spec   = ParamSpec(y_labels[1], y_lo, y_hi),
        y2_spec   = ParamSpec(y_labels[2], y_lo, y_hi),
        y3_spec   = ParamSpec(y_labels[3], y_lo, y_hi),
        y4_spec   = ParamSpec(y_labels[4], y_lo, y_hi),
    )

def _gp2d_pairing(
    beta_label=r"$\beta$", amp_label=r"$A_q$", 
    ls_m_label=r"$l_{m,q}$", ls_q_label=r"$l_{q,q}$",
    y_labels=[r"$y_{q,%d}$" % i for i in range(16)], # Updated to 16 labels
    beta_lo=-4.0, beta_hi=12.0,
    amp_lo=0.01, amp_hi=3.0,
    ls_m_lo=0.5, ls_m_hi=3.0, # m1 correlation length (log space)
    ls_q_lo=0.2, ls_q_hi=1.0, # q correlation length (linear space)
    y_lo=-5.0, y_hi=5.0
):
    return GaussianProcessPairing2D(
        beta_spec = ParamSpec(beta_label, beta_lo, beta_hi),
        amp_spec  = ParamSpec(amp_label,  amp_lo,  amp_hi),
        ls_m_spec = ParamSpec(ls_m_label, ls_m_lo, ls_m_hi),
        ls_q_spec = ParamSpec(ls_q_label, ls_q_lo, ls_q_hi),
        y0_spec   = ParamSpec(y_labels[0], y_lo, y_hi),
        y1_spec   = ParamSpec(y_labels[1], y_lo, y_hi),
        y2_spec   = ParamSpec(y_labels[2], y_lo, y_hi),
        y3_spec   = ParamSpec(y_labels[3], y_lo, y_hi),
        y4_spec   = ParamSpec(y_labels[4], y_lo, y_hi),
        y5_spec   = ParamSpec(y_labels[5], y_lo, y_hi),
        y6_spec   = ParamSpec(y_labels[6], y_lo, y_hi),
        y7_spec   = ParamSpec(y_labels[7], y_lo, y_hi),
        y8_spec   = ParamSpec(y_labels[8], y_lo, y_hi),
        y9_spec   = ParamSpec(y_labels[9], y_lo, y_hi),
        y10_spec  = ParamSpec(y_labels[10], y_lo, y_hi),
        y11_spec  = ParamSpec(y_labels[11], y_lo, y_hi),
        y12_spec  = ParamSpec(y_labels[12], y_lo, y_hi),
        y13_spec  = ParamSpec(y_labels[13], y_lo, y_hi),
        y14_spec  = ParamSpec(y_labels[14], y_lo, y_hi),
        y15_spec  = ParamSpec(y_labels[15], y_lo, y_hi),
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


def _mixture_gp(shared_beta=True, shared_spin=True):
    # Since k=1, we enforce shared_beta and shared_spin
    masses = [_gp()]
    pairings = [_plpairing(beta_label=r"$\beta$")]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    
    return MixtureModel(masses, pairings, spins)

def _mixture_gp_experimental(shared_beta=True, shared_spin=True):
    # Since k=1, we enforce shared_beta and shared_spin
    masses = [_gp_experimental()]
    pairings = [_plpairing(beta_label=r"$\beta$")]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    
    return MixtureModel(masses, pairings, spins)


def _mixture_gp_pairing_experimental(shared_beta=True, shared_spin=True):
    # Since k=1, we enforce shared_beta and shared_spin
    # 1D GP for primary mass, 1D GP for mass ratio
    masses = [_gp_experimental()]
    pairings = [_gp_pairing_experimental()]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    
    return MixtureModel(masses, pairings, spins)


def _mixture_gp_full2d(shared_beta=True, shared_spin=True):
    # k=1 for a purely non-parametric mass & pairing model
    masses = [_gp_experimental()]
    pairings = [_gp2d_pairing()]
    spins = [_spin(mu_label=r"$\mu_\chi$", sig_label=r"$\sigma_\chi$")]
    
    return MixtureModel(masses, pairings, spins)


# ----------------------------------------------------------------------
# 8. Registry and Parsers
# ----------------------------------------------------------------------

def _make(mix_fn, shared_beta=False, shared_spin=False):
    return PopulationModel(mixture=mix_fn(shared_beta=shared_beta, shared_spin=shared_spin))


_RAW_MODELS = {
    "powerlaw+peak":                   (_mixture_plpeak,          "PL+G"),
    "brokenpowerlaw+2peaks":           (_mixture_bpl2peaks,       "BPL+2G"),
    "brokenpowerlaw+3peaks":           (_mixture_bpl3peaks,       "BPL+3G"),
    "brokenpowerlaw+2peaks+powerlaw":  (_mixture_bpl2peaks1pl,    "BPL+2G+PL"),
    "twopowerlaws+peak":               (_mixture_2pl1peak,        "2PL+G"),
    "twopowerlaws+2peaks":             (_mixture_2pl2peaks,       "2PL+2G"),
    "twopowerlaws+3peaks":             (_mixture_2pl3peaks,       "2PL+3G"),
    "gp":                              (_mixture_gp,              "GP"),
    "gp_experimental":                 (_mixture_gp_experimental, "GP EXP"),
    "gp_pairing_experimental":         (_mixture_gp_pairing_experimental, "GPxGP EXP"),
    "gp_2D_experimental":              (_mixture_gp_full2d,       "GP 2D")
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