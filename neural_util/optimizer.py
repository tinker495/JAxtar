from typing import Any, Callable, Optional

import jax
import optax
import optax.tree_utils as otu
from jax import numpy as jnp

PyTree = Any


def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: Optional[jnp.dtype] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable = lambda step: step**0.25,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the ADOPT algorithm.

    ADOPT (Modified Adam Can Converge with Any β2 with the Optimal Rate) is a variant
    of Adam that can converge with any β2 value while maintaining the optimal rate.

    This implementation includes a clipping operation to improve stability, especially
    in the early stages of training. The clipping helps avoid near-zero divisions when
    some elements of the parameter gradient are near zero at initialization.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum.
      use_clipping: Whether to use gradient clipping to improve stability.
        When enabled, the clipping value is set to step**0.25, which aligns
        with the theory to ensure convergence.

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    mu_dtype = optax._src.utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        b2_ = jnp.where(state.count > 0, b2, 0)
        b1_ = jnp.where(state.count > 0, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2_, 2)
        if use_clipping:
            clip_value = clip_value_fn(state.count)
            mu_updates = jax.tree.map(
                lambda ud, nu: jnp.clip(
                    ud / jnp.maximum(jnp.sqrt(nu), eps), -clip_value, clip_value
                ),
                updates,
                state.nu,
            )
        else:
            mu_updates = jax.tree.map(
                lambda ud, nu: ud / jnp.maximum(jnp.sqrt(nu), eps), updates, state.nu
            )
        mu = otu.tree_update_moment(mu_updates, state.mu, b1_, 1)
        count_inc = optax._src.numerics.safe_increment(state.count)
        if nesterov:
            mu_ = otu.tree_update_moment(mu_updates, state.mu, b1_, 1)
        else:
            mu_ = mu
        updates = mu_
        mu = otu.tree_cast(mu, mu_dtype)
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def setup_optimizer(
    params: PyTree, num_devices: int, steps: int, one_iter_size: int, lr_init: float = 1e-3
) -> optax.OptState:
    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices
    warmup_steps = 10 * one_iter_size

    optimizer = optax.contrib.schedule_free_adamw(lr, warmup_steps, weight_decay=0.001)
    return optimizer, optimizer.init(params)
