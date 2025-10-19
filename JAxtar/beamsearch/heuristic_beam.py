import time

import jax
import jax.numpy as jnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.beamsearch.search_base import ACTION_PAD, BeamSearchResult, select_beam


def beam_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
):
    """Construct a batched heuristic beam-search solver."""

    statecls = puzzle.State
    beam_width = batch_size
    denom = max(1, puzzle.action_size // 2)
    min_keep = max(1, beam_width // denom)
    pop_ratio = float(pop_ratio)
    max_depth = max(1, (max_nodes + beam_width - 1) // beam_width)

    def beam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> BeamSearchResult:
        result = BeamSearchResult.build(
            statecls,
            beam_width,
            max_depth,
        )
        default_path_states = statecls.default((beam_width, max_depth + 1))
        default_path_costs = jnp.full((beam_width, max_depth + 1), jnp.inf, dtype=KEY_DTYPE)
        default_path_dists = jnp.full((beam_width, max_depth + 1), jnp.inf, dtype=KEY_DTYPE)

        result.beam = result.beam.at[0].set(start)
        result.cost = result.cost.at[0].set(0)
        result.path_states = result.path_states.at[0, 0].set(start)
        result.path_costs = result.path_costs.at[0, 0].set(0)

        start_dist = heuristic.batched_distance(solve_config, result.beam[:1]).astype(KEY_DTYPE)[0]
        start_score = (cost_weight * result.cost[0] + start_dist).astype(KEY_DTYPE)
        result.dist = result.dist.at[0].set(start_dist)
        result.scores = result.scores.at[0].set(start_score)
        result.path_dists = result.path_dists.at[0, 0].set(start_dist)
        result.parent_index = result.parent_index.at[0].set(-1)

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
            parent_actions = jnp.tile(
                jnp.arange(num_actions, dtype=ACTION_DTYPE)[:, jnp.newaxis],
                (1, beam_width),
            )
            parent_indices = jnp.tile(
                jnp.arange(beam_width, dtype=jnp.int32)[jnp.newaxis, :],
                (num_actions, 1),
            )

            base_costs = search_result.cost
            child_costs = (base_costs[jnp.newaxis, :] + transition_cost).astype(KEY_DTYPE)
            child_valid = jnp.logical_and(filled_mask[jnp.newaxis, :], jnp.isfinite(child_costs))
            child_costs = jnp.where(child_valid, child_costs, jnp.inf)

            dists = jax.vmap(heuristic.batched_distance, in_axes=(None, 0), out_axes=0,)(
                solve_config, neighbours
            ).astype(KEY_DTYPE)
            dists = jnp.where(child_valid, dists, jnp.inf)

            scores = (cost_weight * child_costs + dists).astype(KEY_DTYPE)
            scores = jnp.where(child_valid, scores, jnp.inf)

            flat_states = neighbours.flatten()
            flat_cost = child_costs.reshape(-1)
            flat_dist = dists.reshape(-1)
            flat_scores = scores.reshape(-1)
            flat_actions = parent_actions.reshape(-1)
            flat_parent = parent_indices.reshape(-1)
            flat_valid = child_valid.reshape(-1)

            selected_scores, selected_idx, keep_mask = select_beam(
                flat_scores,
                beam_width,
                pop_ratio=pop_ratio,
                min_keep=min_keep,
            )

            selected_states = flat_states[selected_idx]
            selected_costs = flat_cost[selected_idx]
            selected_dists = flat_dist[selected_idx]
            selected_actions = flat_actions[selected_idx]
            selected_parents = flat_parent[selected_idx]
            selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])

            selected_costs = jnp.where(selected_valid, selected_costs, jnp.inf)
            selected_dists = jnp.where(selected_valid, selected_dists, jnp.inf)
            selected_scores = jnp.where(selected_valid, selected_scores, jnp.inf)
            selected_actions = selected_actions.astype(ACTION_DTYPE)

            safe_parents = jnp.where(selected_valid, selected_parents, 0)
            parent_paths = search_result.path_actions[safe_parents]
            empty_path = jnp.full((beam_width, max_depth), ACTION_PAD, dtype=ACTION_DTYPE)
            action_indices = jnp.arange(beam_width, dtype=jnp.int32)
            parent_paths = parent_paths.at[action_indices, search_result.depth].set(
                selected_actions
            )
            next_paths = jnp.where(selected_valid[:, jnp.newaxis], parent_paths, empty_path)

            next_depth_idx = jnp.minimum(search_result.depth + 1, max_depth)
            parent_state_paths = search_result.path_states[safe_parents]
            parent_cost_paths = search_result.path_costs[safe_parents]
            parent_dist_paths = search_result.path_dists[safe_parents]

            parent_state_paths = parent_state_paths.at[action_indices, next_depth_idx].set(
                selected_states
            )
            parent_cost_paths = parent_cost_paths.at[action_indices, next_depth_idx].set(
                selected_costs
            )
            parent_dist_paths = parent_dist_paths.at[action_indices, next_depth_idx].set(
                selected_dists
            )

            mask = selected_valid

            def _mask_tree(tree, default):
                def _mask(field, default_field):
                    broadcast_shape = (mask.shape[0],) + (1,) * (field.ndim - 1)
                    mask_exp = mask.reshape(broadcast_shape)
                    return jnp.where(mask_exp, field, default_field)

                return jax.tree_util.tree_map(_mask, tree, default)

            next_state_paths = _mask_tree(parent_state_paths, default_path_states)
            mask_2d = mask[:, jnp.newaxis]
            next_cost_paths = jnp.where(mask_2d, parent_cost_paths, default_path_costs)
            next_dist_paths = jnp.where(mask_2d, parent_dist_paths, default_path_dists)

            search_result.beam = selected_states
            search_result.cost = selected_costs
            search_result.dist = selected_dists
            search_result.scores = selected_scores
            search_result.parent_index = jnp.where(
                selected_valid, selected_parents, -jnp.ones_like(selected_parents)
            )
            search_result.path_actions = next_paths
            search_result.path_states = next_state_paths
            search_result.path_costs = next_cost_paths
            search_result.path_dists = next_dist_paths
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
