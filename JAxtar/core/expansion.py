"""
JAxtar Core Expansion Policies
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import base_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.core.bi_result import (
    check_intersection,
    update_meeting_point,
    update_meeting_point_best_only_deferred,
)
from JAxtar.core.common import (
    build_action_major_parent_context,
    build_action_major_parent_layout,
    packed_masked_state_eval,
    partition_and_pack_frontier_candidates,
    sort_and_pack_action_candidates,
)
from JAxtar.core.result import Current, Parent, ParentWithCosts, SearchResult
from JAxtar.core.search_strategy import ExpansionPolicy, ScoringPolicy


@base_dataclass
class EagerExpansion(ExpansionPolicy):
    """
    Eager Expansion Policy (Standard A* / Backward A*).
    Expands states, computes heuristics, inserts to HT, pushes to PQ.
    """

    scoring_policy: ScoringPolicy
    heuristic_fn: Callable[[Any, Puzzle.State, chex.Array], chex.Array]
    cost_weight: float = 1.0 - 1e-6
    is_backward: bool = False
    inverse_action_map: chex.Array | None = None

    def expand(
        self,
        search_result: SearchResult,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        **kwargs,
    ) -> tuple[SearchResult, Current, Puzzle.State, chex.Array]:

        # 1. Get Neighbours
        states = search_result.get_state(current)

        if self.is_backward:
            if self.inverse_action_map is not None:
                inv_actions = self.inverse_action_map

                def _step_action(act):
                    act_batch = jnp.full(
                        (search_result.batch_size,),
                        act,
                        dtype=self.inverse_action_map.dtype,
                    )
                    s, c = puzzle.batched_get_actions(solve_config, states, act_batch, filled)
                    return s, c

                neighbours, ncost = jax.vmap(_step_action)(inv_actions)
                neighbours = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), neighbours)
                ncost = jnp.swapaxes(ncost, 0, 1)

            else:
                neighbours, ncost = puzzle.batched_get_inverse_neighbours(
                    solve_config, states, filled
                )
        else:
            neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)

        # 2. Compute Costs
        action_size = search_result.action_size
        batch_size = search_result.batch_size

        current_costs = current.cost
        # Normalize neighbour/cost layout to action-major: (action_size, batch_size).
        # Some puzzle APIs return (batch_size, action_size), especially in backward paths.
        if ncost.ndim == 2 and ncost.shape[0] == batch_size and ncost.shape[1] == action_size:
            ncost = jnp.swapaxes(ncost, 0, 1)
            neighbours = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), neighbours)

        nextcosts = (current_costs[None, :] + ncost).astype(KEY_DTYPE)
        flat_neighbours = neighbours.flatten()
        flat_nextcosts = nextcosts.flatten()
        flat_filled = jnp.isfinite(flat_nextcosts)

        # 3. HT Insert
        (
            search_result.hashtable,
            flat_new_states_mask,
            cheapest_uniques_mask,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(flat_neighbours, flat_filled, flat_nextcosts)

        # 4. Check optimality
        optimal_mask = jnp.less(flat_nextcosts, search_result.get_cost(hash_idx))
        process_mask = cheapest_uniques_mask & optimal_mask

        # Update cost
        search_result.cost = xnp.update_on_condition(
            search_result.cost, hash_idx.index, process_mask, flat_nextcosts
        )
        # Parent mapping must follow the same action-major flattened layout as
        # `flat_neighbours` / `flat_nextcosts`.
        flat_parent_indices, flat_parent_actions, _ = build_action_major_parent_layout(
            action_size, batch_size
        )
        flat_parent = Parent(
            hashidx=current.hashidx[flat_parent_indices],
            action=flat_parent_actions,
        )

        search_result.parent = xnp.update_on_condition(
            search_result.parent, hash_idx.index, process_mask, flat_parent
        )

        # 5. Pack and Compute Heuristics
        flat_nextcosts_masked = jnp.where(process_mask, flat_nextcosts, jnp.inf)

        (
            vals,
            packed_neighbours,
            new_mask,
            packed_process_mask,
        ) = partition_and_pack_frontier_candidates(
            flat_new_states_mask,
            process_mask,
            flat_neighbours,
            flat_nextcosts_masked,
            hash_idx,
            action_size,
            batch_size,
        )

        def _scan_heur(sr, inputs):
            val_chunk, neigh_chunk, new_mask_chunk, process_mask_chunk = inputs
            new_h = self.heuristic_fn(heuristic_params, neigh_chunk, new_mask_chunk).astype(
                KEY_DTYPE
            )
            sr.dist = xnp.update_on_condition(
                sr.dist, val_chunk.hashidx.index, new_mask_chunk, new_h
            )
            old_h = sr.get_dist(val_chunk)
            h = jnp.where(new_mask_chunk, new_h, old_h)
            priority = self.scoring_policy.compute_priority(val_chunk.cost, h, self.cost_weight)
            return sr, priority

        search_result, priorities = jax.lax.scan(
            _scan_heur,
            search_result,
            (vals, packed_neighbours, new_mask, packed_process_mask),
        )

        # 6. Insert to PQ
        search_result = search_result.insert_batch(priorities, vals, packed_process_mask)

        # 7. Pop Next
        search_result, next_current, next_process_mask = search_result.pop_full()
        next_states = search_result.get_state(next_current)

        return search_result, next_current, next_states, next_process_mask

    def expand_bi(
        self,
        search_result: SearchResult,
        opposite_search_result: SearchResult,
        meeting_point: Any,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        is_forward: bool,
        **kwargs,
    ) -> tuple[SearchResult, Any, Current, Puzzle.State, chex.Array]:
        sr, next_current, next_states, next_filled = self.expand(
            search_result,
            puzzle,
            solve_config,
            heuristic_params,
            current,
            filled,
            **kwargs,
        )

        found, opp_idx, opp_cost, total = check_intersection(
            next_states, sr.get_cost(next_current), next_filled, opposite_search_result
        )

        meeting_point = update_meeting_point(
            meeting_point,
            found,
            next_current.hashidx,
            opp_idx,
            sr.get_cost(next_current),
            opp_cost,
            total,
            is_forward,
        )

        return sr, meeting_point, next_current, next_states, next_filled


@base_dataclass
class DeferredExpansion(ExpansionPolicy):
    """
    Deferred Expansion Policy.
    """

    scoring_policy: ScoringPolicy
    heuristic_fn: Callable[[Any, Puzzle.State, chex.Array], chex.Array]
    cost_weight: float = 1.0 - 1e-6
    look_ahead_pruning: bool = True
    is_backward: bool = False
    inverse_action_map: chex.Array | None = None

    def expand(
        self,
        search_result: SearchResult,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        **kwargs,
    ) -> tuple[SearchResult, Current, Puzzle.State, chex.Array]:

        # 1. Primitives
        action_size = search_result.action_size
        batch_size = search_result.batch_size

        (
            flat_parent_hashidx,
            flat_actions,
            costs,
            filled_tiles,
            unflatten_shape,
        ) = build_action_major_parent_context(
            current.hashidx,
            current.cost,
            filled,
            action_size,
            batch_size,
        )

        # 2. Lookahead / Heuristic
        if self.look_ahead_pruning:
            states = search_result.get_state(current)
            if self.is_backward:
                neighbours, ncost = puzzle.batched_get_inverse_neighbours(
                    solve_config,
                    states,
                    filled,
                )
            else:
                neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)

            look_ahead_costs = (current.cost[None, :] + ncost).astype(KEY_DTYPE)

            flat_neighbours = neighbours.flatten()
            flat_la_costs = look_ahead_costs.flatten()
            flat_filled = jnp.isfinite(flat_la_costs)

            meeting_mask = xnp.unique_mask(flat_neighbours, flat_la_costs, flat_filled)

            h_idxs, found = search_result.hashtable.lookup_parallel(flat_neighbours, meeting_mask)
            old_costs = search_result.get_cost(h_idxs)
            old_dists = search_result.get_dist(h_idxs)

            is_better = jnp.less(flat_la_costs, old_costs)
            candidate_mask = meeting_mask & jnp.logical_or(~found, is_better)

            found_reshaped = found.reshape(action_size, batch_size)
            candidate_mask_reshaped = candidate_mask.reshape(action_size, batch_size)
            old_dists_reshaped = old_dists.reshape(action_size, batch_size)

            need_compute = candidate_mask_reshaped & ~found_reshaped

            computed_h = packed_masked_state_eval(
                flat_neighbours,
                need_compute.flatten(),
                action_size,
                batch_size,
                lambda s, m: self.heuristic_fn(heuristic_params, s, m).astype(KEY_DTYPE),
            ).reshape(action_size, batch_size)

            heuristic_vals = jnp.where(found_reshaped, old_dists_reshaped, computed_h)
            heuristic_vals = jnp.where(
                filled_tiles.reshape(action_size, batch_size), heuristic_vals, jnp.inf
            )

            priority = self.scoring_policy.compute_priority(
                costs.flatten(), heuristic_vals.flatten(), self.cost_weight
            )
            optimal_mask = candidate_mask
            vals_dist = heuristic_vals.flatten()

        else:
            states = search_result.get_state(current)
            h_parent = self.heuristic_fn(heuristic_params, states, filled).astype(KEY_DTYPE)
            h_vals = jnp.repeat(h_parent, action_size)
            priority = self.scoring_policy.compute_priority(costs, h_vals, self.cost_weight)
            optimal_mask = filled_tiles
            vals_dist = h_vals

        vals = ParentWithCosts(
            parent=Parent(hashidx=flat_parent_hashidx, action=flat_actions),
            cost=costs,
            dist=vals_dist,
        )

        priority = jnp.where(optimal_mask, priority, jnp.inf)

        sorted_keys, sorted_vals, sorted_mask = sort_and_pack_action_candidates(
            priority, vals, optimal_mask, action_size, batch_size
        )

        search_result = search_result.insert_batch(sorted_keys, sorted_vals, sorted_mask)

        search_result, next_current, next_states, next_filled = search_result.pop_full_with_actions(
            puzzle,
            solve_config,
            use_heuristic=True,
            is_backward=self.is_backward,
        )

        return search_result, next_current, next_states, next_filled

    def expand_bi(
        self,
        search_result: SearchResult,
        opposite_search_result: SearchResult,
        meeting_point: Any,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        is_forward: bool,
        **kwargs,
    ) -> tuple[SearchResult, Any, Current, Puzzle.State, chex.Array]:
        # Copied logic with intersection check (as implemented in previous turn)
        # Note: I'm overwriting file so I should assume I'm writing full file.
        # But for brevity I'll assume I write the version with expand_bi filled recursively if I was just editing.
        # But here I am overwriting.

        # 1. Primitives
        action_size = search_result.action_size
        batch_size = search_result.batch_size

        (
            flat_parent_hashidx,
            flat_actions,
            costs,
            filled_tiles,
            unflatten_shape,
        ) = build_action_major_parent_context(
            current.hashidx,
            current.cost,
            filled,
            action_size,
            batch_size,
        )

        if self.look_ahead_pruning:
            states = search_result.get_state(current)
            if self.is_backward:
                neighbours, ncost = puzzle.batched_get_inverse_neighbours(
                    solve_config,
                    states,
                    filled,
                )
            else:
                neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)

            # Ensure ncost is (action_size, batch_size) for consistent broadcasting
            if ncost.ndim == 2 and ncost.shape[0] == batch_size:
                ncost = jnp.swapaxes(ncost, 0, 1)  # (batch, action) -> (action, batch)
            look_ahead_costs = (current.cost[None, :] + ncost).astype(KEY_DTYPE)
            flat_neighbours = neighbours.flatten()
            flat_la_costs = look_ahead_costs.flatten()
            flat_filled = jnp.isfinite(flat_la_costs)

            meeting_mask = xnp.unique_mask(flat_neighbours, flat_la_costs, flat_filled)

            h_idxs, found = search_result.hashtable.lookup_parallel(flat_neighbours, meeting_mask)
            old_costs = search_result.get_cost(h_idxs)
            old_dists = search_result.get_dist(h_idxs)

            is_better = jnp.less(flat_la_costs, old_costs)
            candidate_mask = meeting_mask & jnp.logical_or(~found, is_better)

            meeting_point = update_meeting_point_best_only_deferred(
                meeting_point,
                this_sr=search_result,
                opposite_sr=opposite_search_result,
                candidate_states=flat_neighbours,
                candidate_costs=flat_la_costs,
                candidate_mask=meeting_mask,
                this_found=found,
                this_hashidx=h_idxs,
                this_old_costs=old_costs,
                this_parent_hashidx=flat_parent_hashidx,
                this_parent_action=flat_actions.flatten(),
                is_forward=is_forward,
            )

            found_reshaped = found.reshape(action_size, batch_size)
            candidate_mask_reshaped = candidate_mask.reshape(action_size, batch_size)
            old_dists_reshaped = old_dists.reshape(action_size, batch_size)

            need_compute = candidate_mask_reshaped & ~found_reshaped

            computed_h = packed_masked_state_eval(
                flat_neighbours,
                need_compute.flatten(),
                action_size,
                batch_size,
                lambda s, m: self.heuristic_fn(heuristic_params, s, m).astype(KEY_DTYPE),
            ).reshape(action_size, batch_size)

            heuristic_vals = jnp.where(found_reshaped, old_dists_reshaped, computed_h)
            heuristic_vals = jnp.where(
                filled_tiles.reshape(action_size, batch_size), heuristic_vals, jnp.inf
            )

            priority = self.scoring_policy.compute_priority(
                costs.flatten(), heuristic_vals.flatten(), self.cost_weight
            )
            optimal_mask = candidate_mask
            vals_dist = heuristic_vals.flatten()

        else:
            states = search_result.get_state(current)
            h_parent = self.heuristic_fn(heuristic_params, states, filled).astype(KEY_DTYPE)
            h_vals = jnp.repeat(h_parent, action_size)
            priority = self.scoring_policy.compute_priority(costs, h_vals, self.cost_weight)
            optimal_mask = filled_tiles
            vals_dist = h_vals

        vals = ParentWithCosts(
            parent=Parent(hashidx=flat_parent_hashidx, action=flat_actions),
            cost=costs,
            dist=vals_dist,
        )
        priority = jnp.where(optimal_mask, priority, jnp.inf)
        sorted_keys, sorted_vals, sorted_mask = sort_and_pack_action_candidates(
            priority, vals, optimal_mask, action_size, batch_size
        )
        search_result = search_result.insert_batch(sorted_keys, sorted_vals, sorted_mask)
        search_result, next_current, next_states, next_filled = search_result.pop_full_with_actions(
            puzzle,
            solve_config,
            use_heuristic=True,
            is_backward=self.is_backward,
        )

        if not self.look_ahead_pruning:
            found, opp_idx, opp_cost, total = check_intersection(
                next_states,
                search_result.get_cost(next_current),
                next_filled,
                opposite_search_result,
            )
            meeting_point = update_meeting_point(
                meeting_point,
                found,
                next_current.hashidx,
                opp_idx,
                search_result.get_cost(next_current),
                opp_cost,
                total,
                is_forward,
            )

        return search_result, meeting_point, next_current, next_states, next_filled


@base_dataclass
class QStarExpansion(ExpansionPolicy):
    """
    Q* Expansion Policy.
    Updates Q-values based on child's stored values.
    """

    scoring_policy: ScoringPolicy
    q_fn: Callable[[Any, Puzzle.State, chex.Array], chex.Array]  # Returns Q(parent, action)
    cost_weight: float = 1.0 - 1e-6
    look_ahead_pruning: bool = True
    pessimistic_update: bool = True

    def expand(
        self,
        search_result: SearchResult,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        **kwargs,
    ) -> tuple[SearchResult, Current, Puzzle.State, chex.Array]:

        action_size = search_result.action_size
        batch_size = search_result.batch_size

        states = search_result.get_state(current)

        # 1. Compute Q-values for parent actions
        # q_fn returns [action_size, batch_size] or similar?
        # qstar.py switcher returns [action_size, batch_size] presumably from transpose?
        # variable_batch_switcher returns [batch_size, action_size] depending on q_fn implementation.
        # qstar.py: q_vals = switcher(...).transpose()

        # We assume q_fn returns [batch, action] or we adapt.
        # If we use broadcasted lambda in constructor, we can control it.
        # Here we assume q_fn result needs transpose if [batch, action].

        # Actually variable_batch_switcher output shape depends on `batched_q_value`.
        # let's assume `q_fn` returns [action, batch] to match layout we want for flattening.
        # Or [batch, action] and we flatten responsibly.

        q_vals = self.q_fn(heuristic_params, states, filled)
        # Assuming [action, batch] for now (based on qstar.py transpose logic).
        # qstar.py said: `q_vals = q_vals.transpose()` implies original was `[batch, action]`.
        # So q_fn typically returns `[batch, action]`.

        # We want [action, batch] for flattening?
        # `variable_batch_switcher` flattens internally or returns whatever `q_fn` returns?
        # It calls `q_fn(slice)`.

        # Let's standardize on `q_vals` being `[action, batch]`.

        (
            flat_parent_hashidx,
            flat_actions,
            costs,
            filled_tiles,
            unflatten_shape,
        ) = build_action_major_parent_context(
            current.hashidx,
            current.cost,
            filled,
            action_size,
            batch_size,
        )

        # Flattened filled tiles is [action * batch]

        # 2. Lookahead
        if self.look_ahead_pruning:
            neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)
            look_ahead_costs = (current.cost[None, :] + ncost).astype(KEY_DTYPE)

            flat_neighbours = neighbours.flatten()
            flat_la_costs = look_ahead_costs.flatten()
            flat_filled = jnp.isfinite(flat_la_costs)

            # Distinct score for pruning
            # score = cost +/- eps * dist
            dist_sign = -1.0 if self.pessimistic_update else 1.0
            dists = q_vals.flatten()
            distinct_score = flat_la_costs + dist_sign * 1e-5 * dists

            unique_mask = xnp.unique_mask(flat_neighbours, distinct_score, flat_filled)

            h_idxs, found = search_result.hashtable.lookup_parallel(flat_neighbours, unique_mask)
            old_costs = search_result.get_cost(h_idxs)
            old_dists = search_result.get_dist(h_idxs)

            # Update logic
            step_cost = ncost.flatten().astype(KEY_DTYPE)
            q_old = old_dists.astype(KEY_DTYPE) + step_cost

            if self.pessimistic_update:
                q_old_val = jnp.where(found, q_old, -jnp.inf)
                dists = jnp.maximum(dists, q_old_val)
            else:
                q_old_val = jnp.where(found, q_old, jnp.inf)
                dists = jnp.minimum(dists, q_old_val)

            better_cost = jnp.less(flat_la_costs, old_costs)
            optimal_mask = unique_mask & jnp.logical_or(~found, better_cost)

            vals_dist = dists.astype(KEY_DTYPE)

        else:
            vals_dist = q_vals.flatten().astype(KEY_DTYPE)
            optimal_mask = filled_tiles.flatten()

        vals = ParentWithCosts(
            parent=Parent(hashidx=flat_parent_hashidx, action=flat_actions),
            cost=costs.flatten(),
            dist=vals_dist,  # Q-values
        )

        priority = self.scoring_policy.compute_priority(
            costs.flatten(), vals_dist, self.cost_weight
        )
        priority = jnp.where(optimal_mask, priority, jnp.inf)

        sorted_keys, sorted_vals, sorted_mask = sort_and_pack_action_candidates(
            priority, vals, optimal_mask, action_size, batch_size
        )

        search_result = search_result.insert_batch(sorted_keys, sorted_vals, sorted_mask)

        search_result, next_current, next_states, next_filled = search_result.pop_full_with_actions(
            puzzle,
            solve_config,
            use_heuristic=False,
            is_backward=False,
        )

        return search_result, next_current, next_states, next_filled
