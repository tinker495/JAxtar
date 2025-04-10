from typing import Any

import jax

PyTree = Any


@jax.jit
def soft_update(target_params: PyTree, params: PyTree, tau: float) -> PyTree:
    return jax.tree.map(lambda t, n: t * tau + n * (1 - tau), target_params, params)
