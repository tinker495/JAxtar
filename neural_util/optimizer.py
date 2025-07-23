from typing import Any, Callable, Optional

import jax
import optax
import optax.tree_utils as otu
from jax import numpy as jnp

PyTree = Any


def scale_by_rmsprop(
    b2: float = 0.999, eps: float = 1e-8, initial_scale: float = 0.0
) -> optax.GradientTransformation:
    return optax.scale_by_rms(b2=b2, eps=eps, initial_scale=initial_scale)


def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.99,
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


OPTIMIZERS = {
    "adam": optax.scale_by_adam,
    "adopt": scale_by_adopt,
    "rmsprop": scale_by_rmsprop,
    "lamb_adam": None,  # This is a placeholder for the lamb optimizer
    "lamb_adopt": None,  # This is a placeholder for the lamb optimizer
}


def setup_optimizer(
    params: PyTree,
    num_devices: int,
    steps: int,
    one_iter_size: int,
    optimizer_name: str,
    lr_init: float = 1e-3,
    weight_decay_size: float = 0.001,
) -> optax.OptState:
    # Create the main decay schedule, making it conditional
    is_lamb = optimizer_name.startswith("lamb")
    optimizer_name = optimizer_name.replace("lamb_", "")
    is_no_wd = weight_decay_size == 0.0

    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices

    warmup_steps = 10 * one_iter_size

    # Create a warmup schedule that linearly increases from 0 to init_value
    warmup_schedule = optax.linear_schedule(
        init_value=0.0, end_value=lr, transition_steps=warmup_steps
    )

    decay_schedule = optax.schedules.exponential_decay(
        lr,
        5000,
        0.995,
    )
    # Combine the schedules
    lr_schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule], boundaries=[warmup_steps]
    )

    def mask_batch_stat_or_bias(params):
        def mask_fn(path, value):
            # Check if 'batch_stats' is part of any dictionary key in the path
            is_batch_stat = any(
                isinstance(entry, jax.tree_util.DictKey) and "batch_stats" in entry.key
                for entry in path
            )
            is_bias = (
                path and isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == "bias"
            )
            return not (is_batch_stat or is_bias)

        return jax.tree_util.tree_map_with_path(mask_fn, params)

    def optimizer_fn(learning_rate):
        if optimizer_name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scaler = OPTIMIZERS[optimizer_name]()

        if is_lamb:
            chain = optax.chain(
                scaler,
                optax.add_decayed_weights(weight_decay_size, mask=mask_batch_stat_or_bias)
                if not is_no_wd
                else optax.identity(),
                optax.scale_by_trust_ratio(),
                optax.scale_by_learning_rate(learning_rate),
            )
            return chain
        else:
            chain = optax.chain(
                scaler,
                optax.add_decayed_weights(weight_decay_size, mask=mask_batch_stat_or_bias)
                if not is_no_wd
                else optax.identity(),
                optax.scale_by_learning_rate(learning_rate),
            )
            return chain

    optimizer = optax.inject_hyperparams(optimizer_fn)(lr_schedule)
    return optimizer, optimizer.init(params)
