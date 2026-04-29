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

    def _norm(self, theta) -> jnp.ndarray:
        """
        Normalisation integral over MASS_GRID.

        Depends only on theta, not on the sample m.  Call once per theta
        and reuse across all samples rather than recomputing inside a
        per-sample loop.
        """
        return jnp.trapezoid(self._eval_unnorm(MASS_GRID, theta), MASS_GRID)

    def __call__(self, m, theta, norm=None):
        """
        Evaluate the normalised mass PDF at m.

        Parameters
        ----------
        norm : float or None
            Pre-computed value from ``_norm(theta)``.  Supply this when
            evaluating many samples at the same theta to avoid redundant
            grid integration.
        """
        p = self._eval_unnorm(m, theta)
        n = norm if norm is not None else self._norm(theta)
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
        # PairingModel norm integrates over q for each m1, so it IS
        # sample-dependent and cannot be lifted out of the per-sample loop.
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

    def _norm(self, theta) -> jnp.ndarray:
        """
        Normalisation integral over CHI_GRID.

        Depends only on theta, not on the sample chieff.  Call once per
        theta and reuse across all samples.
        """
        return jnp.trapezoid(self._eval_unnorm(CHI_GRID, theta), CHI_GRID)

    def __call__(self, chieff, theta, norm=None):
        """
        Evaluate the normalised spin PDF at chieff.

        Parameters
        ----------
        norm : float or None
            Pre-computed value from ``_norm(theta)``.  Supply this when
            evaluating many samples at the same theta.
        """
        p = self._eval_unnorm(chieff, theta)
        n = norm if norm is not None else self._norm(theta)
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

        # ------------------------------------------------------------------
        # Compute mass and spin normalisation integrals once per theta.
        #
        # Both depend only on the population parameters theta, not on the
        # individual samples (m1, chieff).  The previous code called
        # c_m(m1, tm) and c_s(chieff, ts) which each re-evaluated a
        # 500-point or 200-point grid integral internally on every call.
        # With ~250k samples per likelihood evaluation (256 PE × ~1000 sel)
        # this recomputed the same integral ~250k times identically.
        #
        # PairingModel normalisation integrates over q for each m1, making
        # it genuinely sample-dependent; it stays inside the sample loop.
        # ------------------------------------------------------------------
        mass_norms = [
            c._norm(tm_list[i])
            for i, c in enumerate(self.mass_components)
        ]
        spin_norms = [
            c._norm(ts_list[0] if self.shared_spin else ts_list[i])
            for i, c in enumerate(self.spin_components)
        ]

        out = 0.0
        for i in range(self.k):
            # Mass component and parameters
            c_m = self.mass_components[i]
            tm = tm_list[i]  # <-- This was missing!
            
            # Pairing component and parameters
            p_idx = 0 if self.shared_pairing else i
            c_p = self.pairing_components[p_idx]
            tp = tp_list[p_idx]
            
            # Spin component and parameters
            s_idx = 0 if self.shared_spin else i
            c_s = self.spin_components[s_idx]
            ts = ts_list[s_idx]

            # Extract bounds if specified
            mmin   = M_LO
            dmmin  = 0.01

            if hasattr(c_m, "m_min_spec"):
                mmin_idx = c_m.param_specs.index(c_m.m_min_spec)
                mmin = tm[mmin_idx]
            if hasattr(c_m, "dm_min_spec"):
                dmmin_idx = c_m.param_specs.index(c_m.dm_min_spec)
                dmmin = tm[dmmin_idx]

            # Compute the combined probability for this component
            out = out + w[i] * (
                c_m(m1, tm, norm=mass_norms[i])
                * c_p(m1, q, mmin, dmmin, tp)
                * c_s(chieff, ts, norm=spin_norms[s_idx])
            )

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