import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

@jit
def logdiffexp(x, y):
    return x + jnp.log1p(jnp.exp(y-x))