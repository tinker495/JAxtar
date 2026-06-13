from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp


def _pad_leading_axis(values: chex.Array, pad_width: int, pad_value) -> chex.Array:
    if pad_width <= 0:
        return values

    pad_spec = [(0, pad_width)] + [(0, 0)] * (values.ndim - 1)
    pad_constant = jnp.array(pad_value, dtype=values.dtype)
    return jnp.pad(values, pad_spec, constant_values=pad_constant)


def variable_batch_switcher_builder(
    eval_fn: Callable[[Any, Any], chex.Array],
    *,
    max_batch_size: int,
    min_batch_size: int,
    pad_value,
):
    """
    Build a callable that dynamically selects a pre-JITed branch based on how many
    entries in ``filled`` are active.

    Args:
        eval_fn: Callable taking ``(solve_config, current_states)`` and returning an
            array whose leading dimension corresponds to the batch size.
        max_batch_size: Maximum batch size that the callable should support.
        min_batch_size: Minimum batch size (inclusive) to pre-compile.
        pad_value: Value used when padding the evaluated result back to
            ``max_batch_size`` along the leading dimension.
    """

    max_batch_size = int(max_batch_size)
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    min_batch_size = int(min_batch_size)
    min_batch_size = max(0, min(min_batch_size, max_batch_size))

    branches: list[Callable[[Any, Any], chex.Array]] = []
    batch_sizes: list[int] = []

    current_batch = max_batch_size
    while True:
        # Handle 0 batch size specifically to skip computation
        if current_batch == 0:
            batch_sizes.append(0)

            # Zero batch: slice 0 items so eval_fn yields a correctly-typed empty
            # result, then pad up to max_batch_size. jax.lax.switch requires every
            # branch to share output shape/dtype, which the eval_fn + pad path keeps.
            def zero_branch(distance_fn_parameters, current):
                sliced_current = current[:0]
                values = eval_fn(distance_fn_parameters, sliced_current)
                return _pad_leading_axis(values, max_batch_size, pad_value)

            branches.append(zero_branch)
            break

        batch_sizes.append(current_batch)

        pad_width = max_batch_size - current_batch

        def make_branch(batch_size: int, pad_width: int):
            def branch(distance_fn_parameters, current):
                sliced_current = current[:batch_size]
                values = eval_fn(distance_fn_parameters, sliced_current)
                return _pad_leading_axis(values, pad_width, pad_value)

            return branch

        branches.append(make_branch(current_batch, pad_width))

        if current_batch <= min_batch_size:
            # When min_batch_size is 0 keep going so we still emit the 0-batch branch.
            if not (min_batch_size == 0 and current_batch > 0):
                break

        next_batch = max(current_batch >> 1, min_batch_size)
        if next_batch == current_batch:
            # Halving stalled at min_batch_size; drop to 0 only if 0 is allowed.
            if min_batch_size == 0 and current_batch > 0:
                next_batch = 0
            else:
                break

        current_batch = next_batch

    batch_sizes_array = jnp.array(batch_sizes, dtype=jnp.int32)
    num_branches = len(branches)

    def variable_batch_switcher(
        distance_fn_parameters,
        current,
        filled: chex.Array,
    ):
        filled_count = jnp.sum(filled.astype(jnp.int32), axis=0)
        filled_count = jnp.clip(filled_count, 0, max_batch_size)
        eligible = filled_count <= batch_sizes_array
        num_eligible = jnp.sum(eligible.astype(jnp.int32))
        branch_index = jnp.clip(num_eligible - 1, 0, num_branches - 1)
        return jax.lax.switch(
            branch_index,
            tuple(branches),
            distance_fn_parameters,
            current,
        )

    return variable_batch_switcher


__all__ = ["variable_batch_switcher_builder"]
