from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

from .utils import MASS_GRID, Q_GRID, CHI_GRID, M_LO

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
        
        p_grid = self._eval_unnorm(m1_expanded, Q_GRID, m_min, dm_min, theta)
        n = jnp.trapezoid(p_grid, Q_GRID, axis=-1)
        n = n.reshape(jnp.shape(m1))
        return p / jnp.where(n > 0, n, 1.0)

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
        specs = [ParamSpec(rf"$f_{i+1}$", 0.0, 1.0) for i in range(self.n_weight_params)]
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

@dataclass
class PopulationModel:
    mixture: MixtureModel

    @property
    def param_specs(self):
        return [*self.mixture.param_specs, ParamSpec(r"$\gamma$", -10.0, 10.0)]

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_p_pop(self, m1, q, z, chieff, theta):
        tm = theta[:-1]
        gamma = theta[-1]
        p = self.mixture(m1, q, chieff, tm)
        return jnp.where(p > 0, jnp.log(p), -1e10) + (gamma - 1.0) * jnp.log1p(z)