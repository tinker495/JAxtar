from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import chex
import jax
import jax.numpy as jnp

from JAxtar import annotate
from JAxtar.utils.array_ops import stable_partition_three

_PARTITION_MODES = {"auto", "flat", "row_scan"}


def _pad_leading_axis(values: chex.Array, pad_width: int, pad_value: Any) -> chex.Array:
    if pad_width <= 0:
        return values
    pad_spec = [(0, pad_width)] + [(0, 0)] * (values.ndim - 1)
    return jnp.pad(values, pad_spec, constant_values=pad_value)


def _infer_pad_dtype(pad_value: Any, expected_output_dtype: jnp.dtype | None) -> jnp.dtype:
    if expected_output_dtype is not None:
        return expected_output_dtype
    return jnp.asarray(pad_value).dtype


def _build_batch_sizes(
    *,
    limit_batch_size: int,
    min_batch_size: int,
    batch_sizes: Sequence[int] | None,
    batch_size_policy: str,
) -> list[int]:
    if batch_sizes is not None:
        unique_sizes = sorted({int(bs) for bs in batch_sizes if 0 <= int(bs) <= limit_batch_size})
        if not unique_sizes:
            raise ValueError(
                "batch_sizes must contain at least one integer in [0, annotate.BATCH_SPLIT_UNIT]"
            )
        if unique_sizes[-1] < limit_batch_size:
            unique_sizes.append(limit_batch_size)
        return unique_sizes

    policy = batch_size_policy or "dense_pref"
    if policy == "dense_pref":
        generated_sizes = [limit_batch_size, min_batch_size if min_batch_size > 0 else 0]
    elif policy == "balanced":
        generated_sizes = [limit_batch_size]
        if limit_batch_size >= 4:
            generated_sizes.extend(
                [
                    (limit_batch_size * 3) // 4,
                    limit_batch_size // 2,
                    limit_batch_size // 4,
                ]
            )
        generated_sizes.append(min_batch_size if min_batch_size > 0 else 0)
    elif policy == "sparse_pref":
        generated_sizes = [limit_batch_size]
        curr = limit_batch_size // 2
        while curr >= min_batch_size and curr > 0:
            generated_sizes.append(curr)
            curr = curr // 2
        if min_batch_size > 0 and min_batch_size not in generated_sizes:
            generated_sizes.append(min_batch_size)
        if min_batch_size == 0:
            generated_sizes.append(0)
    else:
        raise ValueError(f"Unknown batch_size_policy: {batch_size_policy}")

    return sorted({int(bs) for bs in generated_sizes if 0 <= int(bs) <= limit_batch_size})


def build_batch_sizes_for_cap(
    cap: int,
    *,
    min_batch_unit: int | None = None,
) -> list[int]:
    """Build power-of-two branch sizes from cap down to min_batch_unit."""
    cap = int(cap)
    if cap <= 0:
        raise ValueError("cap must be positive")

    if min_batch_unit is None:
        min_batch_unit = annotate.MIN_BATCH_UNIT
    min_batch_unit = max(1, int(min_batch_unit))

    sizes = [cap]
    curr = cap // 2
    while curr >= min_batch_unit and curr > 0:
        sizes.append(curr)
        curr //= 2
    if min_batch_unit <= cap:
        sizes.append(min_batch_unit)

    return sorted(set(sizes))


def _flatten_batched_current(current: Any, filled_shape: tuple[int, ...]) -> Any:
    n_leading = len(filled_shape)
    n_total = prod(filled_shape)

    def _reshape_leaf(leaf: chex.Array) -> chex.Array:
        if leaf.ndim < n_leading:
            raise ValueError(
                f"All Pytree leaves must have at least {n_leading} leading batch dims. "
                f"Got leaf with ndim={leaf.ndim}, shape={leaf.shape}."
            )
        if tuple(leaf.shape[:n_leading]) != tuple(filled_shape):
            raise ValueError(
                "Leading batch dims of all leaves must match filled.shape. "
                f"filled.shape={filled_shape}, leaf.shape={leaf.shape}."
            )
        trailing = leaf.shape[n_leading:]
        return leaf.reshape((n_total, *trailing))

    return jax.tree_util.tree_map(_reshape_leaf, current)


def _reshape_current_to_rows(
    current: Any, filled_shape: tuple[int, ...], num_rows: int, row_width: int
) -> Any:
    n_leading = len(filled_shape)

    def _reshape_leaf(leaf: chex.Array) -> chex.Array:
        if leaf.ndim < n_leading:
            raise ValueError(
                f"All Pytree leaves must have at least {n_leading} leading batch dims. "
                f"Got leaf with ndim={leaf.ndim}, shape={leaf.shape}."
            )
        if tuple(leaf.shape[:n_leading]) != tuple(filled_shape):
            raise ValueError(
                "Leading batch dims of all leaves must match filled.shape. "
                f"filled.shape={filled_shape}, leaf.shape={leaf.shape}."
            )
        trailing = leaf.shape[n_leading:]
        return leaf.reshape((num_rows, row_width, *trailing))

    return jax.tree_util.tree_map(_reshape_leaf, current)


def variable_batch_switcher_builder(
    eval_fn: Callable[[Any, Any], chex.Array],
    *,
    pad_value: Any,
    min_batch_size: int | None = None,
    batch_sizes: Sequence[int] | None = None,
    batch_size_policy: str = "dense_pref",
    partition_mode: str = "auto",
    expected_output_shape: tuple[int, ...] | None = None,
    expected_output_dtype: jnp.dtype | None = None,
):
    """
    Build a callable that dynamically selects a pre-JITed branch by valid-count.

    Supports automatic chunking above ``annotate.BATCH_SPLIT_UNIT`` and optional
    row-wise scan+skip when ``filled`` has 2 or more dimensions.

    Args:
        eval_fn: Callable taking ``(distance_fn_parameters, current_states)`` and returning
            an array whose leading dimension corresponds to the batch size.
        pad_value: Value used when padding evaluated results.
        min_batch_size: Minimum branch size (inclusive). Defaults to ``annotate.MIN_BATCH_UNIT``.
        batch_sizes: Optional explicit branch sizes. If provided, overrides ``batch_size_policy``.
        batch_size_policy: Preset policy used when ``batch_sizes`` is None.
            Supported: ``dense_pref`` (default), ``balanced``, ``sparse_pref``.
        partition_mode: ``auto`` (default), ``flat``, or ``row_scan``.
            In ``auto``, row-scan is used when ``filled.ndim >= 2``.
        expected_output_shape: Trailing output shape (excluding batch dim), used for zero-branch
            and row-skip pad synthesis when provided.
        expected_output_dtype: Output dtype for synthesized pad outputs.
    """

    if partition_mode not in _PARTITION_MODES:
        raise ValueError(
            f"Unknown partition_mode: {partition_mode}. Expected one of {sorted(_PARTITION_MODES)}"
        )

    limit_batch_size = int(annotate.BATCH_SPLIT_UNIT)
    if limit_batch_size <= 0:
        raise ValueError("annotate.BATCH_SPLIT_UNIT must be positive")

    if min_batch_size is None:
        min_batch_size = annotate.MIN_BATCH_UNIT
    min_batch_size = max(0, min(int(min_batch_size), limit_batch_size))

    unique_sizes = _build_batch_sizes(
        limit_batch_size=limit_batch_size,
        min_batch_size=min_batch_size,
        batch_sizes=batch_sizes,
        batch_size_policy=batch_size_policy,
    )
    if not unique_sizes:
        raise ValueError("No valid branch size is available for variable_batch_switcher_builder")

    branch_fns = []
    for batch_size in unique_sizes:
        if batch_size == 0:

            def zero_branch(distance_fn_parameters, current):
                if expected_output_shape is not None:
                    del distance_fn_parameters, current
                    out_dtype = _infer_pad_dtype(pad_value, expected_output_dtype)
                    return jnp.full(
                        (limit_batch_size, *expected_output_shape), pad_value, dtype=out_dtype
                    )
                sliced_current = jax.tree_util.tree_map(lambda x: x[:0], current)
                values = eval_fn(distance_fn_parameters, sliced_current)
                return _pad_leading_axis(values, limit_batch_size, pad_value)

            branch_fns.append(zero_branch)
            continue

        pad_width = limit_batch_size - batch_size

        def make_branch(bs: int, branch_pad_width: int):
            def standard_branch(distance_fn_parameters, current):
                sliced_current = jax.tree_util.tree_map(lambda x: x[:bs], current)
                values = eval_fn(distance_fn_parameters, sliced_current)
                return _pad_leading_axis(values, branch_pad_width, pad_value)

            return standard_branch

        branch_fns.append(make_branch(batch_size, pad_width))

    branch_fns = tuple(branch_fns)
    batch_sizes_array = jnp.asarray(unique_sizes, dtype=jnp.int32)
    max_branch_idx = len(unique_sizes) - 1

    def _run_chunk(distance_fn_parameters, current_chunk, filled_chunk):
        filled_count = jnp.sum(filled_chunk, dtype=jnp.int32)
        filled_count = jnp.clip(filled_count, 0, limit_batch_size)

        def _run_zero(_):
            if expected_output_shape is not None:
                out_dtype = _infer_pad_dtype(pad_value, expected_output_dtype)
                return jnp.full(
                    (limit_batch_size, *expected_output_shape), pad_value, dtype=out_dtype
                )
            empty_current = jax.tree_util.tree_map(lambda x: x[:0], current_chunk)
            empty_values = eval_fn(distance_fn_parameters, empty_current)
            return _pad_leading_axis(empty_values, limit_batch_size, pad_value)

        def _run_nonzero(_):
            idx = jnp.searchsorted(batch_sizes_array, filled_count, side="left")
            idx = jnp.clip(idx, 0, max_branch_idx)
            is_limit_branch = idx == max_branch_idx

            def _run_dense(__):
                return jax.lax.switch(idx, branch_fns, distance_fn_parameters, current_chunk)

            def _run_sparse(__):
                # If valid entries are already packed at prefix, skip reorder/scatter overhead.
                seen_false = jnp.cumsum((~filled_chunk).astype(jnp.int32), axis=0) > 0
                is_prefix_packed = ~jnp.any(jnp.logical_and(filled_chunk, seen_false))

                def _run_packed(___):
                    return jax.lax.switch(idx, branch_fns, distance_fn_parameters, current_chunk)

                def _run_unpack(___):
                    sort_indices = stable_partition_three(
                        filled_chunk.reshape(-1),
                        jnp.zeros_like(filled_chunk.reshape(-1), dtype=jnp.bool_),
                    )
                    sorted_current = jax.tree_util.tree_map(
                        lambda x: x[sort_indices], current_chunk
                    )
                    sorted_res = jax.lax.switch(
                        idx, branch_fns, distance_fn_parameters, sorted_current
                    )
                    return jnp.zeros_like(sorted_res).at[sort_indices].set(sorted_res)

                return jax.lax.cond(is_prefix_packed, _run_packed, _run_unpack, operand=None)

            return jax.lax.cond(is_limit_branch, _run_dense, _run_sparse, operand=None)

        return jax.lax.cond(filled_count == 0, _run_zero, _run_nonzero, operand=None)

    def _run_flat(distance_fn_parameters, current_flat, filled_flat):
        n_total = filled_flat.shape[0]
        remainder = n_total % limit_batch_size
        pad_len = (limit_batch_size - remainder) % limit_batch_size
        target_len = n_total + pad_len
        num_chunks = target_len // limit_batch_size

        filled_padded = jnp.pad(filled_flat, (0, pad_len), constant_values=False)
        filled_chunks = filled_padded.reshape((num_chunks, limit_batch_size))

        def _pad_leaf(leaf: chex.Array) -> chex.Array:
            return jnp.pad(leaf, ((0, pad_len),) + ((0, 0),) * (leaf.ndim - 1))

        current_padded = jax.tree_util.tree_map(_pad_leaf, current_flat)

        def _reshape_leaf_to_chunks(leaf: chex.Array) -> chex.Array:
            trailing = leaf.shape[1:]
            return leaf.reshape((num_chunks, limit_batch_size, *trailing))

        current_chunks = jax.tree_util.tree_map(_reshape_leaf_to_chunks, current_padded)

        def scan_body(carry, scan_input):
            del carry
            current_chunk, filled_chunk = scan_input
            return None, _run_chunk(distance_fn_parameters, current_chunk, filled_chunk)

        _, result_chunks = jax.lax.scan(scan_body, None, (current_chunks, filled_chunks))
        result_flat = result_chunks.reshape((target_len, *result_chunks.shape[2:]))
        return result_flat[:n_total]

    def _run_with_leading(distance_fn_parameters, current, filled):
        filled_shape = tuple(filled.shape)
        current_flat = _flatten_batched_current(current, filled_shape)
        filled_flat = filled.reshape(-1)
        result_flat = _run_flat(distance_fn_parameters, current_flat, filled_flat)
        return result_flat.reshape((*filled_shape, *result_flat.shape[1:]))

    def _build_row_pad(distance_fn_parameters, current_rows, filled_rows, row_width: int):
        if expected_output_shape is not None:
            out_dtype = _infer_pad_dtype(pad_value, expected_output_dtype)
            return jnp.full((row_width, *expected_output_shape), pad_value, dtype=out_dtype)

        row_current = jax.tree_util.tree_map(lambda x: x[0], current_rows)
        row_filled = filled_rows[0]
        row_spec = jax.eval_shape(
            lambda c, f: _run_with_leading(distance_fn_parameters, c, f),
            row_current,
            row_filled,
        )
        return jnp.full(row_spec.shape, pad_value, dtype=row_spec.dtype)

    def _run_row_scan(distance_fn_parameters, current, filled):
        filled_shape = tuple(filled.shape)
        if len(filled_shape) < 2:
            return _run_with_leading(distance_fn_parameters, current, filled)

        row_shape = filled_shape[:-1]
        row_width = filled_shape[-1]
        num_rows = prod(row_shape)
        if num_rows == 0:
            return _run_with_leading(distance_fn_parameters, current, filled)

        current_rows = _reshape_current_to_rows(current, filled_shape, num_rows, row_width)
        filled_rows = filled.reshape((num_rows, row_width))
        pad_row = _build_row_pad(distance_fn_parameters, current_rows, filled_rows, row_width)

        def scan_body(carry, scan_input):
            del carry
            row_current, row_filled = scan_input
            row_has_valid = jnp.any(row_filled)
            row_output = jax.lax.cond(
                row_has_valid,
                lambda _: _run_with_leading(distance_fn_parameters, row_current, row_filled),
                lambda _: pad_row,
                operand=None,
            )
            return None, row_output

        _, row_outputs = jax.lax.scan(scan_body, None, (current_rows, filled_rows))
        return row_outputs.reshape((*filled_shape, *row_outputs.shape[2:]))

    def variable_batch_switcher(
        distance_fn_parameters,
        current: chex.Array,
        filled: chex.Array,
    ):
        use_row_scan = partition_mode == "row_scan" or (
            partition_mode == "auto" and filled.ndim >= 2
        )
        if use_row_scan:
            return _run_row_scan(distance_fn_parameters, current, filled)
        return _run_with_leading(distance_fn_parameters, current, filled)

    return variable_batch_switcher


__all__ = ["build_batch_sizes_for_cap", "variable_batch_switcher_builder"]
