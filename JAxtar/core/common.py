"""
JAxtar Core Common Helpers
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import HashIdx

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.core.result import Current, SearchResult


def resolve_neighbour_layout(puzzle: Puzzle, *, is_backward: bool = False) -> str:
    """Resolve declared neighbour layout contract for puzzle transitions."""
    if is_backward:
        layout = getattr(puzzle, "inverse_neighbour_layout", None)
        if layout is None:
            layout = getattr(puzzle, "neighbour_layout", "action_major")
    else:
        layout = getattr(puzzle, "neighbour_layout", "action_major")

    if layout not in ("action_major", "batch_major"):
        raise ValueError(
            "Invalid neighbour layout contract. "
            "Expected 'action_major' or 'batch_major', "
            f"got {layout!r}."
        )
    return layout


def normalize_neighbour_cost_layout(
    neighbours: Puzzle.State,
    step_costs: chex.Array,
    action_size: int,
    batch_size: int,
    *,
    layout: str,
) -> tuple[Puzzle.State, chex.Array]:
    """Normalize neighbour tensors into canonical action-major layout."""
    if step_costs.ndim != 2:
        return neighbours, step_costs

    if layout == "batch_major":
        expected_shape = (batch_size, action_size)
        if step_costs.shape != expected_shape:
            raise ValueError(
                f"batch_major layout expects costs shape {expected_shape}, got {step_costs.shape}"
            )
        return jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), neighbours), jnp.swapaxes(
            step_costs, 0, 1
        )

    if layout == "action_major":
        expected_shape = (action_size, batch_size)
        if step_costs.shape != expected_shape:
            raise ValueError(
                f"action_major layout expects costs shape {expected_shape}, got {step_costs.shape}"
            )
        return neighbours, step_costs

    raise ValueError(f"Unsupported neighbour layout: {layout!r}")


def build_action_major_parent_layout(action_size: int, batch_size: int):
    """
    Create indices for action-major parent layout.
    """
    flat_size = action_size * batch_size
    flat_positions = jnp.arange(flat_size, dtype=jnp.int32)
    flat_parent_indices = flat_positions % batch_size
    flat_parent_actions = (flat_positions // batch_size).astype(ACTION_DTYPE)
    unflatten_shape = (action_size, batch_size)
    return flat_parent_indices, flat_parent_actions, unflatten_shape


def build_action_major_parent_context(
    hash_idx: HashIdx,
    cost: chex.Array,
    filled: chex.Array,
    action_size: int,
    batch_size: int,
):
    """
    Build context for expanding nodes in action-major order.
    """
    (
        flat_parent_indices,
        flat_parent_actions,
        unflatten_shape,
    ) = build_action_major_parent_layout(action_size, batch_size)

    flat_parent_hashidx = hash_idx[flat_parent_indices]
    costs = jnp.broadcast_to(cost[jnp.newaxis, :], unflatten_shape).reshape(-1)
    filled_tiles = jnp.broadcast_to(filled[jnp.newaxis, :], unflatten_shape).reshape(-1)

    return (
        flat_parent_hashidx,
        flat_parent_actions,
        costs,
        filled_tiles,
        unflatten_shape,
    )


def sort_and_pack_action_candidates(
    flattened_keys: chex.Array,
    flattened_vals: Any,
    optimal_mask: chex.Array,
    action_size: int,
    sr_batch_size: int,
):
    """
    Sort and pack candidates for PQ insertion.
    """
    neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

    num_candidates = flattened_keys.shape[0]
    sorted_keys, sorted_idx = jax.lax.sort_key_val(neighbour_keys, jnp.arange(num_candidates))
    sorted_vals = flattened_vals[sorted_idx]
    sorted_mask = jnp.isfinite(sorted_keys)
    return sorted_keys, sorted_vals, sorted_mask


def insert_priority_queue_batches(
    search_result: SearchResult,
    neighbour_keys: chex.Array,
    vals: Any,
    optimal_mask: chex.Array,
):
    """
    Insert batches into PQ.
    """
    # Flatten just in case
    neighbour_keys = neighbour_keys.flatten()
    vals = vals.flatten() if hasattr(vals, "flatten") else vals  # vals might be struct
    optimal_mask = optimal_mask.flatten()

    def _insert_batch(carry, inputs):
        pq = carry
        key_chunk, val_chunk, mask_chunk = inputs
        pq = jax.lax.cond(
            jnp.any(mask_chunk),
            lambda p, k, v: p.insert(k, v),
            lambda p, *_: p,
            pq,
            key_chunk,
            val_chunk,
        )
        return pq, None

    # We need to reshape into chunks of `batch_size`.
    batch_size = search_result.batch_size
    total_size = neighbour_keys.shape[0]
    # Pad to multiple of batch_size
    pad_len = (-total_size) % batch_size
    if pad_len > 0:
        neighbour_keys = jnp.pad(neighbour_keys, (0, pad_len), constant_values=jnp.inf)
        vals = xnp.pad(vals, (0, pad_len))  # Struct padding
        optimal_mask = jnp.pad(optimal_mask, (0, pad_len), constant_values=False)

    num_chunks = neighbour_keys.shape[0] // batch_size
    keys_reshaped = neighbour_keys.reshape(num_chunks, batch_size)
    vals_reshaped = vals.reshape(num_chunks, batch_size)
    mask_reshaped = optimal_mask.reshape(num_chunks, batch_size)

    # Note: BGPQ insert handles generic vals if registered.
    # `vals` is `Parant_with_Costs` or `Current` or `Parent`.
    # `xnp.pad` and reshape handles struct.

    new_pq, _ = jax.lax.scan(
        _insert_batch,
        search_result.priority_queue,
        (keys_reshaped, vals_reshaped, mask_reshaped),
    )
    search_result.priority_queue = new_pq
    return search_result


def packed_masked_state_eval(
    flat_states: Puzzle.State,
    flat_mask: chex.Array,
    action_size: int,
    batch_size: int,
    heuristic_fn: Callable[[Puzzle.State, chex.Array], chex.Array],
    dtype=KEY_DTYPE,
) -> chex.Array:
    """
    Evaluates heuristic efficiently by packing valid states.
    """
    num_items = flat_mask.shape[0]
    sort_idx = jnp.argsort(flat_mask, descending=True)
    sorted_states = flat_states[sort_idx]
    sorted_mask = flat_mask[sort_idx]

    def _eval_chunk(carry, inputs):
        start_idx = carry
        s_chunk, m_chunk = inputs
        h_vals = heuristic_fn(s_chunk, m_chunk)
        return start_idx + batch_size, h_vals

    pad_len = (-num_items) % batch_size
    if pad_len > 0:
        sorted_states = xnp.pad(sorted_states, (0, pad_len))
        sorted_mask = jnp.pad(sorted_mask, (0, pad_len), constant_values=False)

    first_leaf = jax.tree_util.tree_leaves(sorted_states)[0]
    num_chunks = first_leaf.shape[0] // batch_size
    state_chunks = sorted_states.reshape(num_chunks, batch_size)
    mask_chunks = sorted_mask.reshape(num_chunks, batch_size)

    _, res_chunks = jax.lax.scan(_eval_chunk, 0, (state_chunks, mask_chunks))
    flat_res = res_chunks.flatten()

    result = jnp.zeros(num_items + pad_len, dtype=dtype)
    result = result.at[sort_idx].set(flat_res[:num_items])
    result = result[:num_items]

    return result


def loop_continue_if_not_solved(
    search_result: SearchResult,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
    states: Puzzle.State,
    filled: chex.Array,
) -> chex.Array:
    """
    Condition to continue loop.
    """
    not_solved = jnp.logical_not(search_result.solved)
    frontier_has_work = jnp.logical_or(
        jnp.any(filled),
        search_result.priority_queue.size > 0,
    )

    # Also check if ANY of the current popped states are solutions
    # (Typically this is done in body/finalize, but we can do lazy check here or rely on body)
    # `astar.py` does: `loop_continue_if_not_solved(search_result, puzzle, solve_config, states, filled)`
    # It checks `batched_is_solved` on `states`.
    is_solved_now = jnp.any(jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled))

    return jnp.logical_and(
        not_solved,
        jnp.logical_and(jnp.logical_not(is_solved_now), frontier_has_work),
    )


def finalize_search_result(
    search_result: SearchResult,
    current: Current,
    solved_mask: chex.Array,
) -> SearchResult:
    """
    Finalize the search result when a solution is found.
    """
    any_solved = jnp.any(solved_mask)

    def _update_solved(sr):
        first_idx = jnp.argmax(solved_mask)
        best_current = jax.tree_util.tree_map(lambda x: x[None, ...], current[first_idx])
        sr.solved = jnp.array(True)
        sr.solved_idx = best_current
        return sr

    search_result = jax.lax.cond(
        any_solved,
        _update_solved,
        lambda sr: sr,
        search_result,
    )
    return search_result


def partition_and_pack_frontier_candidates(
    flatten_new_states_mask: chex.Array,
    final_process_mask: chex.Array,
    flatten_neighbours: Puzzle.State,
    flatten_nextcosts: chex.Array,
    hash_idx: HashIdx,
    action_size: int,
    sr_batch_size: int,
):
    """
    Partition candidates into New (needs heuristic) and Old (reuse heuristic),
    and pack them for processing.
    """
    currents = Current(hashidx=hash_idx, cost=flatten_nextcosts)
    pack_mask = final_process_mask

    _, sorted_idx = jax.lax.sort_key_val(
        jnp.where(pack_mask, flatten_nextcosts, jnp.inf),
        jnp.arange(flatten_nextcosts.shape[0]),
    )

    sorted_currents = currents[sorted_idx]
    sorted_neighbours = flatten_neighbours[sorted_idx]
    sorted_new_mask = flatten_new_states_mask[sorted_idx]
    sorted_process_mask = pack_mask[sorted_idx]

    num_items = flatten_nextcosts.shape[0]
    pad_len = (-num_items) % sr_batch_size
    if pad_len > 0:
        sorted_currents = xnp.pad(
            sorted_currents, (0, pad_len)
        )  # Struct pad ? Current usually works
        sorted_neighbours = xnp.pad(sorted_neighbours, (0, pad_len))
        sorted_new_mask = jnp.pad(sorted_new_mask, (0, pad_len), constant_values=False)
        sorted_process_mask = jnp.pad(sorted_process_mask, (0, pad_len), constant_values=False)

    num_chunks = sorted_currents.cost.shape[0] // sr_batch_size

    vals_reshaped = sorted_currents.reshape(num_chunks, sr_batch_size)
    neighbours_reshaped = sorted_neighbours.reshape(num_chunks, sr_batch_size)
    new_mask_reshaped = sorted_new_mask.reshape(num_chunks, sr_batch_size)
    process_mask_reshaped = sorted_process_mask.reshape(num_chunks, sr_batch_size)

    return vals_reshaped, neighbours_reshaped, new_mask_reshaped, process_mask_reshaped


def compute_pop_process_mask(
    keys: chex.Array,
    pop_ratio: float,
    min_pop: int,
):
    """
    Compute mask for limiting population size.
    """
    # Assuming keys are sorted (best first)
    batch_size = keys.shape[0]
    valid_count = jnp.sum(jnp.isfinite(keys))

    # target count
    target = jnp.minimum(
        valid_count, jnp.maximum(min_pop, (valid_count * pop_ratio).astype(jnp.int32))
    )
    target = jnp.minimum(target, batch_size)  # Cap at batch

    idx = jnp.arange(batch_size)
    mask = idx < target
    # Also ensure key is finite
    mask = jnp.logical_and(mask, jnp.isfinite(keys))
    return mask
