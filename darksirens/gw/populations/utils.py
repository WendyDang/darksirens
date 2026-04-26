import os
import jax.numpy as jnp
from jax import jit

# Set environmental XLA flags 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# ======================================================================
# Fixed grids
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
# Smooth edge filters
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