import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.beamsearch.search_base import (
    ACTION_PAD,
    TRACE_INDEX_DTYPE,
    TRACE_INVALID,
    BeamSearchResult,
    non_backtracking_mask,
    select_beam,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def beam_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    non_backtracking_steps: int = 3,
):
    """Construct a batched heuristic beam-search solver."""

    statecls = puzzle.State
    beam_width = batch_size
    denom = max(1, puzzle.action_size // 2)
    min_keep = max(1, beam_width // denom)
    pop_ratio = float(pop_ratio)
    max_depth = max(1, (max_nodes + beam_width - 1) // beam_width)
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        lambda solve_config, current: heuristic.batched_distance(solve_config, current),
        max_batch_size=beam_width,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    def beam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> BeamSearchResult:
        result = BeamSearchResult.build(
            statecls,
            beam_width,
            max_depth,
        )

        result.beam = result.beam.at[0].set(start)
        result.cost = result.cost.at[0].set(0)

        init_filled = jnp.zeros(beam_width, dtype=jnp.bool_).at[0].set(True)
        start_dist = variable_heuristic_batch_switcher(
            solve_config, result.beam, init_filled
        ).astype(KEY_DTYPE)[0]
        start_score = (cost_weight * result.cost[0] + start_dist).astype(KEY_DTYPE)
        result.dist = result.dist.at[0].set(start_dist)
        result.scores = result.scores.at[0].set(start_score)
        result.parent_index = result.parent_index.at[0].set(-1)
        result.active_trace = result.active_trace.at[0].set(0)
        result.trace_cost = result.trace_cost.at[0].set(0)
        result.trace_dist = result.trace_dist.at[0].set(start_dist)
        result.trace_depth = result.trace_depth.at[0].set(0)
        result.trace_action = result.trace_action.at[0].set(ACTION_PAD)
        result.trace_state = result.trace_state.at[0].set(start)

        def _cond(search_result: BeamSearchResult):
            filled_mask = search_result.filled_mask()
            has_states = filled_mask.any()
            depth_ok = search_result.depth < max_depth

            beam_states = search_result.beam
            solved = puzzle.batched_is_solved(solve_config, beam_states)
            solved = jnp.logical_and(solved, filled_mask)
            return jnp.logical_and(jnp.logical_and(depth_ok, has_states), ~solved.any())

        def _body(search_result: BeamSearchResult):
            filled_mask = search_result.filled_mask()
            beam_states = search_result.beam

            neighbours, transition_cost = puzzle.batched_get_neighbours(
                solve_config, beam_states, filled_mask
            )

            num_actions = transition_cost.shape[0]
            base_costs = search_result.cost
            child_costs = (base_costs[jnp.newaxis, :] + transition_cost).astype(KEY_DTYPE)
            child_valid = jnp.logical_and(filled_mask[jnp.newaxis, :], jnp.isfinite(child_costs))
            child_costs = jnp.where(child_valid, child_costs, jnp.inf)

            flat_states = neighbours.flatten()
            flat_cost = child_costs.reshape(-1)
            flat_valid = child_valid.reshape(-1)

            unique_flat_mask = xnp.unique_mask(
                flat_states,
                key=flat_cost,
                filled=flat_valid,
            )
            unique_mask = unique_flat_mask.reshape(child_valid.shape)
            child_valid = jnp.logical_and(child_valid, unique_mask)

            if non_backtracking_steps:
                parent_trace_matrix = jnp.broadcast_to(
                    search_result.active_trace[jnp.newaxis, :], child_valid.shape
                )
                flat_parent_trace = parent_trace_matrix.reshape(-1)
                allowed_mask = non_backtracking_mask(
                    flat_states,
                    flat_parent_trace,
                    search_result.trace_state,
                    search_result.trace_parent,
                    non_backtracking_steps,
                ).reshape(child_valid.shape)
                child_valid = jnp.logical_and(child_valid, allowed_mask)

            child_costs = jnp.where(child_valid, child_costs, jnp.inf)

            parent_matrix = jnp.broadcast_to(
                jnp.arange(beam_width, dtype=jnp.int32), (num_actions, beam_width)
            )

            perm_keys = jnp.where(child_valid, 0, 1).astype(jnp.int32)
            perm = jnp.argsort(perm_keys, axis=1)

            def _reorder_leaf(leaf):
                expanded = perm
                for _ in range(leaf.ndim - perm.ndim):
                    expanded = expanded[..., None]
                expanded_perm = jnp.broadcast_to(expanded, leaf.shape)
                return jnp.take_along_axis(leaf, expanded_perm, axis=1)

            neighbours = jax.tree_util.tree_map(_reorder_leaf, neighbours)
            child_costs = jnp.take_along_axis(child_costs, perm, axis=1)
            child_valid = jnp.take_along_axis(child_valid, perm, axis=1)
            parent_matrix = jnp.take_along_axis(parent_matrix, perm, axis=1)

            def _pad_tree_leading_axis(tree, pad_len: int):
                if pad_len <= 0:
                    return tree

                def _pad(arr):
                    pad_spec = [(0, pad_len)] + [(0, 0)] * (arr.ndim - 1)
                    return jnp.pad(arr, pad_spec, constant_values=0)

                return jax.tree_util.tree_map(_pad, tree)

            child_count = num_actions * beam_width
            flat_states_tree = jax.tree_util.tree_map(
                lambda x: x.reshape((child_count,) + x.shape[2:]),
                neighbours,
            )
            flat_valid = child_valid.reshape(child_count)
            global_perm_keys = jnp.where(flat_valid, 0, 1).astype(jnp.int32)
            global_perm = jnp.argsort(global_perm_keys, axis=0)
            ordered_states_tree = jax.tree_util.tree_map(
                lambda x: jnp.take(x, global_perm, axis=0),
                flat_states_tree,
            )
            ordered_valid = jnp.take(flat_valid, global_perm, axis=0)

            pad_len = (-child_count) % beam_width
            padded_states_tree = _pad_tree_leading_axis(ordered_states_tree, pad_len)
            padded_valid = jnp.pad(ordered_valid, (0, pad_len), constant_values=False)

            num_chunks = (child_count + pad_len) // beam_width
            chunk_states_tree = jax.tree_util.tree_map(
                lambda x: x.reshape((num_chunks, beam_width) + x.shape[1:]),
                padded_states_tree,
            )
            chunk_valid = padded_valid.reshape((num_chunks, beam_width))

            chunk_dists = jnp.full((num_chunks, beam_width), jnp.inf, dtype=KEY_DTYPE)

            def _compute_chunk(i, acc):
                row_mask = chunk_valid[i]

                def _calc(_):
                    chunk_states = jax.tree_util.tree_map(lambda x: x[i], chunk_states_tree)
                    dist_row = variable_heuristic_batch_switcher(
                        solve_config,
                        chunk_states,
                        row_mask,
                    ).astype(KEY_DTYPE)
                    return jnp.where(row_mask, dist_row, jnp.inf)

                dist_row = jax.lax.cond(
                    jnp.any(row_mask),
                    _calc,
                    lambda _: acc[i],
                    None,
                )
                return acc.at[i].set(dist_row)

            dists_compacted = jax.lax.fori_loop(0, num_chunks, _compute_chunk, chunk_dists)
            ordered_dists = dists_compacted.reshape(-1)[:child_count]
            flat_dists = jnp.full(child_count, jnp.inf, dtype=KEY_DTYPE)
            flat_dists = flat_dists.at[global_perm].set(ordered_dists)
            dists = flat_dists.reshape(child_valid.shape)

            scores = (cost_weight * child_costs + dists).astype(KEY_DTYPE)
            scores = jnp.where(child_valid, scores, jnp.inf)

            flat_states = neighbours.flatten()
            flat_cost = child_costs.reshape(-1)
            flat_dist = dists.reshape(-1)
            flat_scores = scores.reshape(-1)
            flat_valid = child_valid.reshape(-1)
            flat_parents = parent_matrix.reshape(-1)

            selected_scores, selected_idx, keep_mask = select_beam(
                flat_scores,
                beam_width,
                pop_ratio=pop_ratio,
                min_keep=min_keep,
            )

            selected_states = flat_states[selected_idx]
            selected_costs = flat_cost[selected_idx]
            selected_dists = flat_dist[selected_idx]
            selected_actions = (selected_idx // beam_width).astype(ACTION_DTYPE)
            selected_parents = flat_parents[selected_idx]
            selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])
            unique_valid = xnp.unique_mask(
                selected_states,
                key=selected_scores,
                filled=selected_valid,
            )
            selected_valid = jnp.logical_and(selected_valid, unique_valid)

            parent_trace_ids = search_result.active_trace[selected_parents]

            selected_costs = jnp.where(selected_valid, selected_costs, jnp.inf)
            selected_dists = jnp.where(selected_valid, selected_dists, jnp.inf)
            selected_scores = jnp.where(selected_valid, selected_scores, jnp.inf)
            selected_actions = selected_actions.astype(ACTION_DTYPE)

            invalid_parent = jnp.full_like(parent_trace_ids, TRACE_INVALID)
            parent_trace_ids = jnp.where(selected_valid, parent_trace_ids, invalid_parent)

            next_depth_idx = jnp.minimum(search_result.depth + 1, max_depth)
            trace_offset = next_depth_idx.astype(TRACE_INDEX_DTYPE) * jnp.asarray(
                beam_width, dtype=TRACE_INDEX_DTYPE
            )
            slot_indices = jnp.arange(beam_width, dtype=TRACE_INDEX_DTYPE)
            next_trace_ids = trace_offset + slot_indices

            trace_actions = jnp.where(
                selected_valid,
                selected_actions,
                jnp.full_like(selected_actions, ACTION_PAD),
            )
            depth_fill = jnp.full((beam_width,), next_depth_idx, dtype=jnp.int32)
            depth_default = -jnp.ones((beam_width,), dtype=jnp.int32)
            trace_depths = jnp.where(selected_valid, depth_fill, depth_default)

            search_result.trace_parent = search_result.trace_parent.at[next_trace_ids].set(
                parent_trace_ids
            )
            search_result.trace_action = search_result.trace_action.at[next_trace_ids].set(
                trace_actions
            )
            search_result.trace_cost = search_result.trace_cost.at[next_trace_ids].set(
                selected_costs
            )
            search_result.trace_dist = search_result.trace_dist.at[next_trace_ids].set(
                selected_dists
            )
            search_result.trace_depth = search_result.trace_depth.at[next_trace_ids].set(
                trace_depths
            )
            search_result.trace_state = search_result.trace_state.at[next_trace_ids].set(
                selected_states
            )

            invalid_trace = jnp.full_like(next_trace_ids, TRACE_INVALID)
            next_active_trace = jnp.where(selected_valid, next_trace_ids, invalid_trace)

            search_result.beam = selected_states
            search_result.cost = selected_costs
            search_result.dist = selected_dists
            search_result.scores = selected_scores
            search_result.parent_index = jnp.where(
                selected_valid, selected_parents, -jnp.ones_like(selected_parents)
            )
            search_result.active_trace = next_active_trace
            selected_count = selected_valid.astype(jnp.int32).sum()
            search_result.generated_size = search_result.generated_size + selected_count
            search_result.depth = search_result.depth + 1
            return search_result

        search_result = jax.lax.while_loop(_cond, _body, result)

        filled_mask = search_result.filled_mask()
        solved_mask = puzzle.batched_is_solved(solve_config, search_result.beam)
        solved_mask = jnp.logical_and(solved_mask, filled_mask)

        solved_any = solved_mask.any()
        solved_idx = jnp.argmax(solved_mask)
        solved_idx = jnp.where(
            solved_any, solved_idx.astype(jnp.int32), jnp.array(-1, dtype=jnp.int32)
        )

        search_result.solved = solved_any
        search_result.solved_idx = solved_idx
        return search_result

    beam_fn = jax.jit(beam)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Compile ahead of time to surface potential tracer issues early. This uses
    # empty defaults, mirroring the A*/Q* builders.
    beam_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return beam_fn
