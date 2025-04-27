from typing import Any, Callable, NamedTuple, Optional

import chex
import jax
import optax
import optax.tree_utils as otu
from jax import numpy as jnp
from optax._src import base, combine, numerics, transform
from optax.schedules import _schedule
from optax.transforms import _adding

from .schedule_free import DTypeLike, ScheduleFreeState, schedule_free

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


class ScaleByAdoptRmsState(NamedTuple):
    """State for the Adopt RMS scaling transformation."""

    count: chex.Numeric
    nu: base.Updates


def scale_by_adopt_rms(
    b2: float = 0.9999,
    eps: float = 1e-6,
    bias_correction: bool = True,
) -> base.GradientTransformation:
    """Rescale updates according to the Adopt algorithm's second moment (RMS).

    This transformation implements the scaling part of Adopt/Adam based on the
    exponential moving average of the squared gradients, similar to RMSProp.
    It does not handle the first moment (momentum).

    Args:
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      bias_correction: Whether to include bias correction.

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        nu = otu.tree_zeros_like(params)  # Second moment
        return ScaleByAdoptRmsState(count=jnp.zeros([], jnp.int32), nu=nu)

    def update_fn(updates, state, params=None):
        del params
        b2_t = b2 if not bias_correction else jnp.where(state.count > 0, b2, 0.0)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2_t, 2)

        updates_scaled = jax.tree.map(lambda g, n: g / (jnp.sqrt(n) + eps), updates, nu)

        if bias_correction:
            count_inc = numerics.safe_increment(state.count)
            bias_correction_term = 1.0 - jnp.power(b2, count_inc)
            # Ensure bias correction term doesn't cause issues (e.g., sqrt of negative)
            updates_scaled = jax.tree.map(
                lambda u: u * jnp.sqrt(jnp.maximum(0.0, bias_correction_term)), updates_scaled
            )
        else:
            # Still increment count even if not bias correcting, state needs updating
            count_inc = numerics.safe_increment(state.count)

        return updates_scaled, ScaleByAdoptRmsState(count=count_inc, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def schedule_free_adopt(
    learning_rate: float = 1e-3,
    warmup_steps: Optional[int] = None,
    b1: float = 0.9,  # Schedule-Free b1
    b2: float = 0.999,  # Adopt b2
    eps: float = 1e-6,  # Adopt eps
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    state_dtype: Optional[DTypeLike] = None,
    bias_correction: bool = True,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free wrapper for Adopt.

    Combines the Adopt second-moment scaling with the Schedule-Free wrapper.

    Args:
      learning_rate: Peak learning rate for the schedule.
      warmup_steps: Number of linear warmup steps.
      b1: Schedule-Free momentum parameter (beta_1 for y update).
      b2: Adopt decay rate for the second moment.
      eps: Adopt epsilon for numerical stability.
      weight_decay: Strength of weight decay regularization.
      weight_lr_power: Power for downweighting averaging in Schedule-Free.
      state_dtype: Optional dtype for the Schedule-Free z state.
      bias_correction: Whether to use bias correction in Adopt RMS scaling.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs`.
    """
    lr_schedule = learning_rate
    if warmup_steps is not None:
        lr_schedule = _schedule.warmup_constant_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )

    # Base optimizer chain: Adopt RMS scaling -> weight decay -> LR scaling
    optimizer_chain = [scale_by_adopt_rms(b2=b2, eps=eps, bias_correction=bias_correction)]
    if weight_decay > 0.0:
        optimizer_chain.append(_adding.add_decayed_weights(weight_decay))

    optimizer_chain.append(transform.scale_by_learning_rate(lr_schedule))

    base_optimizer = combine.chain(*optimizer_chain)

    return schedule_free(
        base_optimizer,
        learning_rate=lr_schedule,  # Pass schedule here for max_lr tracking
        b1=b1,
        weight_lr_power=weight_lr_power,
        state_dtype=state_dtype,
    )


def setup_optimizer(
    params: PyTree, num_devices: int, steps: int, one_iter_size: int, lr_init: float = 1e-3
) -> tuple[base.GradientTransformationExtraArgs, ScheduleFreeState]:
    # Add warmup to the learning rate schedule
    lr = lr_init * num_devices
    warmup_steps = 10 * one_iter_size

    optimizer = optax.contrib.schedule_free_adamw(lr, warmup_steps, weight_decay=0.001)
    # optimizer = schedule_free_adopt(lr, warmup_steps, weight_decay=0.001)

    return optimizer, optimizer.init(params)
