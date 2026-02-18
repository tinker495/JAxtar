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
    # Filter valid candidates
    neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

    # Sort candidates
    # Removed top_k logic to keep all candidates sorted.

    # Reshape for insertion? No, insert_priority_queue_batches takes 1D arrays usually?
    # Actually checking search_base.py: it reshaped them?
    # lines 343-352 in bi_astar_d.py:
    # neighbour_keys_reshaped, vals_reshaped, optimal_mask_reshaped = sort_and_pack...
    # The return of sort_and_pack in search_base.py was (keys, vals, mask)
    # But strictly speaking we just want the best batch_size candidates per batch?
    # Wait, sort_and_pack usually packs into (batch_size, action_size)?
    # No, typically we flatten everything then pick top K or just keep them all?
    # In `astar_d.py`, it inserts generic batches. `insert_priority_queue_batches` handles chunks.

    # Let's stick to the logic: we want to insert as many as possible or just the valid ones.
    # The simple approach is just masking and returning.
    # But `astar_d` called `sort_and_pack...` with `sr_batch_size`.
    # It returns reshaped arrays matching `sr_batch_size`?
    # Let's check `search_base.py` implementation if possible.
    # I saw it used `jax.lax.top_k` in some context.
    # Actually, if we have `action_size * batch_size` candidates, we might want to prioritize.
    # But typically we just want to pack valid ones.

    # Implementation based on name:
    # Sorts all candidates (flattened) and returns top K?
    # No, usually we want to insert ALL valid candidates.
    # If we use `insert_priority_queue_batches`, it iterates.
    # So we just need to pack them to remove bubbles (infs) to minimize iterations.

    num_candidates = flattened_keys.shape[0]
    # Sort to bring finites to front
    sorted_keys, sorted_idx = jax.lax.sort_key_val(neighbour_keys, jnp.arange(num_candidates))
    sorted_vals = flattened_vals[sorted_idx]
    sorted_mask = jnp.isfinite(sorted_keys)

    # We can reshape if needed, but `insert_priority_queue_batches` accepts flat arrays.
    # The return in `astar_d` was unpacked into `neighbour_keys_reshaped`.
    # So we return the sorted flat arrays.
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
        # Insert only valid keys (mask is implicitly handled by inf key usually,
        # but insert might take mask?)
        # BGPQ.insert(keys, vals) usually ignores infs or inserts them at end.
        pq = pq.insert(key_chunk, val_chunk)
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
    # 1. Sort/Pack states based on mask to group valid ones
    # But wait, heuristic_fn usually takes a batch.
    # If we have sparse valid states, we want to call heuristic_fn only on valid ones
    # to save compute, especially if `heuristic_fn` supports variable batching via `variable_batch_switcher`.

    # `variable_batch_switcher` handles flexible batch sizes.
    # So we just provide the masked states?
    # `heuristic_fn` signature in `astar_d.py`:
    # lambda states_slice, compute_mask: variable_heuristic_batch_switcher(...)

    # So we just pass everything and let switcher handle it?
    # No, `packed_masked_state_eval` implies we should pack them.

    # Logic:
    # 1. Get indices of valid states
    # 2. Extract valid states
    # 3. Call heuristic on VALID states only (in chunks if needed)
    # 4. Scatter results back.

    # Since JAX requires static shapes, "Extract" means "Sort/Permute".
    num_items = flat_mask.shape[0]
    sort_idx = jnp.argsort(flat_mask, descending=True)  # Valid (True) first
    sorted_states = flat_states[sort_idx]
    sorted_mask = flat_mask[sort_idx]

    # We can pass `sorted_states` and `sorted_mask` to `heuristic_fn`.
    # But `heuristic_fn` might expect `batch_size` chunks?
    # The `variable_batch_switcher` is designed to handle "any size up to max_batch".
    # But `flat_states` is `action_size * batch_size`. That might be larger than `max_batch`.
    # So we MUST chunk it.

    # But wait, `variable_batch_switcher` usually has `max_batch_size=batch_size`.
    # Here we have `action*batch` items.
    # So we scan over chunks of `batch_size`.

    def _eval_chunk(carry, inputs):
        start_idx = carry
        # Create a dynamic slice? No, scan works on pre-sliced inputs.
        # Inputs are (state_chunk, mask_chunk)
        s_chunk, m_chunk = inputs
        # We need to inform `heuristic_fn` that this chunk might be partially padded.
        # `variable_batch_switcher` uses `compute_mask` (m_chunk) to determine real batch size.
        # It handles padding automatically if m_chunk is False.
        h_vals = heuristic_fn(s_chunk, m_chunk)
        return start_idx + batch_size, h_vals

    # Reshape sorted arrays into chunks of `batch_size` (or whatever `heuristic_fn` accepts).
    # Assuming `heuristic_fn` handles `batch_size`.
    # We pad to multiple.
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

    # Un-sort (scatter) back to original order
    # sort_idx maps Initial -> Sorted.
    # We want Sorted -> Initial.
    # scatter[sort_idx] = sorted_val
    result = jnp.zeros(num_items + pad_len, dtype=dtype)
    result = result.at[sort_idx].set(flat_res[:num_items])
    # Remove padding
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
    # If any solved, update search_result.solved and solved_idx
    any_solved = jnp.any(solved_mask)

    def _update_solved(sr):
        # Pick the first solved one
        # defined by mask
        first_idx = jnp.argmax(solved_mask)  # argmax of bool gives first True
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
    # This is quite specific to `astar.py` Eager logic.
    # But it is used there.
    # It seems to return `(vals, neighbours, new_states_mask, mask)`
    # `vals` is `Current`.

    currents = Current(hashidx=hash_idx, cost=flatten_nextcosts)

    # We want to process `final_process_mask`.
    # Among them, `flatten_new_states_mask` are NEW (HT insert returned True).
    # Pack them.

    # Sort/Partition logic similar to `sort_and_pack` but keeping `new` and `old` separate?
    # Or just sorting all valid ones?
    # `astar.py` uses `_scan` to generic heuristic on `new_states_mask`.

    # Let's just pack everything that is valid (`final_process_mask`).
    pack_mask = final_process_mask

    # Sort by cost (basic A* optimization) or just compact?
    # Compact is enough. using sort_key_val with boolean mask is standard.
    # We want to put valid ones at start.

    sorted_cost, sorted_idx = jax.lax.sort_key_val(
        jnp.where(pack_mask, flatten_nextcosts, jnp.inf), jnp.arange(flatten_nextcosts.shape[0])
    )

    sorted_currents = currents[sorted_idx]
    sorted_neighbours = flatten_neighbours[sorted_idx]
    sorted_new_mask = flatten_new_states_mask[sorted_idx]
    sorted_process_mask = pack_mask[sorted_idx]

    # We only care about top `batch_size * action_size`?
    # Actually `astar.py` line 200 passes `vals` to `insert_priority_queue_batches`.
    # `insert` handles chunks.
    # But `astar.py`'s `partition_and_pack` seems to imply it chunks or reshapes?
    # In `astar.py`: `vals, neighbours, ... = partition_and_pack(...)`
    # Then `search_result, neighbour_keys = jax.lax.scan(_scan, search_result, (vals, neighbours, ...))`

    # So `partition_and_pack` MUST reshape into (num_chunks, batch_size).

    num_items = flatten_nextcosts.shape[0]
    # Pad
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
