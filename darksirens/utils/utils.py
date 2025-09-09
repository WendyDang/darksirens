import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.99'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

@jit
def logdiffexp(x, y):
    return x + jnp.log1p(jnp.exp(y-x))