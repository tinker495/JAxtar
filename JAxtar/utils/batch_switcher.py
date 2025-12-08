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

            # Branch for 0 batch size: return array of pad_value without calling eval_fn
            def zero_branch(distance_fn_parameters, current):
                # Just return an array of pad_value with shape [max_batch_size, ...]
                # We need to know the output shape/dtype.
                # Since we can't infer it easily without calling eval_fn,
                # we might need a dummy call or rely on the fact that all branches must return same shape/dtype.
                # A safer way is to call eval_fn with 1 element (if possible) or use eval_fn's signature.
                # However, here we simply rely on padding logic if we assume current has correct leading dim?
                # No, current has shape [max_batch_size, ...].
                # We want output [max_batch_size, ...].
                # Let's assume eval_fn returns something compatible with _pad_leading_axis.
                # BUT, pad_leading_axis pads 'values'. If we don't call eval_fn, what is 'values'?
                # It should be an empty array with correct trailing dimensions and dtype.

                # To get correct shape/dtype, we can call eval_fn on a dummy input of size 1 and discard it?
                # That defeats the purpose of 0 cost.
                # Instead, let's assume pad_value is scalar and broadcast it?
                # JAX switch requires all branches to return same shape/dtype.
                # So we actually MUST return something that matches eval_fn's output structure.

                # If current_batch > 0 branches exist, JAX will infer the type from them.
                # But we need to construct the array.
                # Strategy: Let's just call eval_fn with slice 0?
                # slice 0 of current is empty. eval_fn might fail on empty input.

                # Correct approach:
                # If min_batch_size is allowed to be 0, we should process 0-sized batch by
                # slicing 0 items, calling eval_fn(0 items), and padding back to max.
                # If eval_fn supports 0-sized input, this works automatically!
                # If eval_fn does NOT support 0-sized input, we can't support 0 batch size easily without more info.

                # Let's try standard path: slice 0 -> eval_fn -> pad.
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
            # If we reached min_batch_size and it is > 0, check if we should add 0 case
            # If min_batch_size is 0, loop continues until current_batch hits 0
            if min_batch_size == 0 and current_batch > 0:
                # Force next iteration to hit 0 eventually
                pass
            else:
                break

        next_batch = max(current_batch >> 1, min_batch_size)
        if next_batch == current_batch:
            # If we are stuck (e.g. at min_batch_size), but we want to go to 0
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
