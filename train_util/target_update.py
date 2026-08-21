from typing import Any

import jax
import optax

PyTree = Any


@jax.jit
def soft_update(target_params: PyTree, params: PyTree, tau: float) -> PyTree:
    return optax.incremental_update(target_params, params, tau)
