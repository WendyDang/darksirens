from jax import jit
from jax import numpy as jnp

@jit
def logdiffexp(x, y):
    """Stable log(exp(x) - exp(y)) for y <= x. If y > x, result is undefined; return -inf."""
    return jnp.where(y <= x, x + jnp.log1p(-jnp.exp(y - x)), -jnp.inf)