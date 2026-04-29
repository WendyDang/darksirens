import jax
import jax.numpy as jnp
from dataclasses import dataclass
from tinygp import GaussianProcess, kernels

from .base import MassComponent, PairingModel, ParamSpec
from .utils import sfilter_low, sfilter_high, MASS_GRID

@dataclass
class GaussianProcessMass1D(MassComponent):
    m_min_spec: ParamSpec; m_max_spec: ParamSpec; dm_min_spec: ParamSpec; dm_max_spec: ParamSpec
    alpha_spec: ParamSpec; amp_spec: ParamSpec; ls_spec: ParamSpec
    y0_spec: ParamSpec; y1_spec: ParamSpec; y2_spec: ParamSpec; y3_spec: ParamSpec
    y4_spec: ParamSpec; y5_spec: ParamSpec; y6_spec: ParamSpec; y7_spec: ParamSpec
    y8_spec: ParamSpec; y9_spec: ParamSpec; y10_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.m_min_spec, self.m_max_spec, self.dm_min_spec, self.dm_max_spec, self.alpha_spec, self.amp_spec, self.ls_spec, self.y0_spec, self.y1_spec, self.y2_spec, self.y3_spec, self.y4_spec, self.y5_spec, self.y6_spec, self.y7_spec, self.y8_spec, self.y9_spec, self.y10_spec]

    def _eval_unnorm(self, m, t):
        mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3]
        alpha, amp, ls = t[4], t[5], t[6]
        y_nodes = jnp.array(t[7:18]) 
        log_MASS_GRID = jnp.log(MASS_GRID)
        log_nodes = jnp.linspace(jnp.log(2.0), jnp.log(100.0), 11)

        def mean_fn(x):
            return -alpha * x

        kernel = (amp**2) * kernels.Matern52(scale=ls)
        gp_prior = GaussianProcess(kernel=kernel, X=log_nodes, mean=mean_fn, diag=1e-5)
        _, gp_cond = gp_prior.condition(y_nodes, log_MASS_GRID)
        
        logpm_grid = gp_cond.loc
        S_grid = sfilter_low(MASS_GRID, mmin, dmmin) * sfilter_high(MASS_GRID, mmax, dmmax)
        pm_grid = S_grid * jnp.exp(logpm_grid)
        return jnp.interp(m, MASS_GRID, pm_grid, left=0.0, right=0.0)

@dataclass
class GaussianProcessMassRatio1D(PairingModel):
    beta_spec: ParamSpec; amp_spec: ParamSpec; ls_spec: ParamSpec
    y0_spec: ParamSpec; y1_spec: ParamSpec; y2_spec: ParamSpec; y3_spec: ParamSpec; y4_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.beta_spec, self.amp_spec, self.ls_spec, self.y0_spec, self.y1_spec, self.y2_spec, self.y3_spec, self.y4_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        beta, amp, ls = t[0], t[1], t[2]
        y_nodes = jnp.array(t[3:8])
        q_nodes = jnp.linspace(0.1, 1.0, 5)
        q_eval_grid = jnp.linspace(0.01, 1.0, 500)

        def mean_fn(x):
            return beta * jnp.log(x)

        kernel = (amp**2) * kernels.Matern52(scale=ls)
        gp_prior = GaussianProcess(kernel=kernel, X=q_nodes, mean=mean_fn, diag=1e-4)
        _, gp_cond = gp_prior.condition(y_nodes, q_eval_grid)
        
        log_pq_grid = jnp.nan_to_num(gp_cond.loc, nan=-20.0)
        log_pq_grid = jnp.clip(log_pq_grid, -10.0, 10.0)
        pq_grid = jnp.exp(log_pq_grid)

        pq = jnp.interp(q, q_eval_grid, pq_grid, left=0.0, right=0.0)
        m1_b, q_b = jnp.broadcast_arrays(m1, q)
        m2 = q_b * m1_b
        
        smooth_cut = sfilter_low(m2, m_min, dm_min)
        smooth_cut = jnp.nan_to_num(smooth_cut, nan=0.0)
        p = smooth_cut * pq
        return jnp.where(m2 < m_min, 1e-20, p)

@dataclass
class GaussianProcessPairing2D(PairingModel):
    beta_spec: ParamSpec; amp_spec: ParamSpec; ls_m_spec: ParamSpec; ls_q_spec: ParamSpec
    y0_spec: ParamSpec; y1_spec: ParamSpec; y2_spec: ParamSpec; y3_spec: ParamSpec
    y4_spec: ParamSpec; y5_spec: ParamSpec; y6_spec: ParamSpec; y7_spec: ParamSpec
    y8_spec: ParamSpec; y9_spec: ParamSpec; y10_spec: ParamSpec; y11_spec: ParamSpec
    y12_spec: ParamSpec; y13_spec: ParamSpec; y14_spec: ParamSpec; y15_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.beta_spec, self.amp_spec, self.ls_m_spec, self.ls_q_spec, self.y0_spec, self.y1_spec, self.y2_spec, self.y3_spec, self.y4_spec, self.y5_spec, self.y6_spec, self.y7_spec, self.y8_spec, self.y9_spec, self.y10_spec, self.y11_spec, self.y12_spec, self.y13_spec, self.y14_spec, self.y15_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        beta, amp, ls_m, ls_q = t[0], t[1], t[2], t[3]
        y_nodes = jnp.array(t[4:20])

        log_m1_nodes = jnp.log(jnp.array([10.0, 20.0, 40.0, 80.0]))
        q_nodes = jnp.array([0.20, 0.40, 0.60, 0.80])
        M_node, Q_node = jnp.meshgrid(log_m1_nodes, q_nodes, indexing="ij")
        X_nodes = jnp.stack([M_node.flatten(), Q_node.flatten()], axis=-1)

        shape = jnp.broadcast_shapes(jnp.shape(m1), jnp.shape(q))
        m1_b, q_b = jnp.broadcast_arrays(m1, q)
        X_test = jnp.stack([jnp.log(m1_b), q_b], axis=-1)
        X_test_flat = X_test.reshape(-1, 2)

        scales = jnp.array([ls_m, ls_q])
        X_nodes_scaled = X_nodes / scales
        X_test_scaled = X_test_flat / scales

        def mean_fn(x_scaled):
            q_true = x_scaled[..., 1] * ls_q
            return beta * jnp.log(q_true + 0.01)

        kernel = (amp**2) * kernels.Matern52()
        gp_prior = GaussianProcess(kernel=kernel, X=X_nodes_scaled, mean=mean_fn, diag=1e-2)
        _, gp_cond = gp_prior.condition(y_nodes, X_test_scaled)
        
        safe_loc = jnp.nan_to_num(gp_cond.loc, nan=-20.0)
        log_pq = jnp.clip(safe_loc.reshape(shape), -10.0, 10.0)
        p = jnp.exp(log_pq)

        m2 = q_b * m1_b
        smooth_cut = sfilter_low(m2, m_min, dm_min)
        smooth_cut = jnp.nan_to_num(smooth_cut, nan=0.0) 
        p = smooth_cut * p
        return jnp.where(m2 < m_min, 1e-20, p)