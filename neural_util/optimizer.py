from typing import Any, Callable, Optional

import jax
import numpy as np
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


def shrink_and_perturb(
    shrink_factor: float = 1e-6, noise_scale_factor: float = 1.0
) -> optax.GradientTransformation:
    """Soft shrink and perturb parameters as a gradient transformation.

    Applies a soft shrink to the parameters (multiplying by a factor < 1)
    and adds Gaussian noise proportional to the standard deviation of each parameter.
    The standard deviation is calculated during initialization and used for subsequent updates.
    This can be used as part of an optimization chain to help escape local minima.

    Args:
        shrink_factor: Factor to multiply parameters by (default: 0.95)
        noise_scale_factor: Factor to multiply parameter std by for noise scale (default: 0.01)

    Returns:
        An optax.GradientTransformation that applies shrinking and perturbation
    """

    def init_fn(params):
        # Calculate standard deviation for each parameter at initialization
        param_stds = jax.tree.map(lambda p: jnp.std(p), params)
        key = jax.random.PRNGKey(np.random.randint(0, 2**30))
        return (param_stds, key)

    def update_fn(updates, state, params):
        param_stds, key = state  # Unpack state

        # Split the key: one for this step's use, one for the next state
        key, key_step = jax.random.split(key)

        def _shrink_and_perturb(p, std, k):
            # Use the pre-calculated standard deviation from init
            noise = noise_scale_factor * std * jax.random.normal(k, p.shape, p.dtype)
            return (1 - shrink_factor) * p + shrink_factor * noise

        # Use key_step for generating noise keys for this update
        keys = jax.random.split(key_step, len(jax.tree_util.tree_leaves(params)))
        flat_params, treedef = jax.tree.flatten(params)
        flat_stds, _ = jax.tree.flatten(param_stds)  # Correctly using param_stds

        perturbed_flat_params = [
            _shrink_and_perturb(p, std, k) for p, std, k in zip(flat_params, flat_stds, keys)
        ]

        new_params = jax.tree.unflatten(treedef, perturbed_flat_params)
        update_diff = jax.tree.map(lambda new_p, old_p: new_p - old_p, new_params, params)
        updates = jax.tree.map(lambda u, d: u + d, updates, update_diff)

        return updates, (param_stds, key)  # Store the new key back

    return optax.GradientTransformation(init_fn, update_fn)


def setup_optimizer(
    params: PyTree, num_devices: int, steps: int, one_iter_size: int, lr_init: float = 1e-3
) -> optax.OptState:
    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices
    warmup_steps = 10 * one_iter_size

    optimizer = optax.contrib.schedule_free_adamw(lr, warmup_steps)

    optimizer = optax.chain(
        optimizer,
        shrink_and_perturb(1e-5, 1.0),
    )

    return optimizer, optimizer.init(params)
