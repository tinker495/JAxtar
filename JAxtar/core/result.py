"""
JAxtar Core Result Structures and Operations
"""

import math
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashIdx,
    HashTable,
    base_dataclass,
    xtructure_dataclass,
)

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)

POP_BATCH_FILLED_RATIO = 0.99


@xtructure_dataclass
class Parent:
    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


@xtructure_dataclass
class Current:
    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@xtructure_dataclass
class ParentWithCosts:
    parent: FieldDescriptor(dtype=Parent)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    dist: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@base_dataclass(static_fields=("action_size",))
class SearchResult:
    """
    Unified Data Structure for Search Results.
    """

    hashtable: HashTable
    priority_queue: BGPQ
    action_size: int
    pop_ratio: float
    min_pop: int
    cost: chex.Array
    dist: chex.Array
    parent: FieldDescriptor(dtype=Parent)
    solved: chex.Array
    solved_idx: FieldDescriptor(dtype=Current)
    pop_generation: chex.Array
    pop_count: chex.Array

    @property
    def capacity(self) -> int:
        return self.hashtable.capacity

    @property
    def batch_size(self) -> int:
        return self.priority_queue.batch_size

    @property
    def generated_size(self) -> int:
        return self.hashtable.size

    def get_state(self, idx: Any) -> Puzzle.State:
        """Retrieve state from hash table using HashIdx or Current."""
        if isinstance(idx, Current):
            return self.hashtable[idx.hashidx]
        elif isinstance(idx, Parent):
            # Parent stores (hashidx, action). We need the STATE at hashidx.
            return self.hashtable[idx.hashidx]
        return self.hashtable[idx]

    def get_cost(self, idx: Any) -> chex.Array:
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return self.cost[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return self.cost[idx.index]
        return self.cost[idx]

    def get_dist(self, idx: Any) -> chex.Array:
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return self.dist[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return self.dist[idx.index]
        return self.dist[idx]

    def get_parent(self, idx: Any) -> Parent:
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return self.parent[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return self.parent[idx.index]
        raise ValueError(f"Invalid index type: {type(idx)}")

    def get_solved_path(self) -> list[Current | Parent]:
        """Reconstruct solved path as parent-chain nodes compatible with CLI visualizers."""
        if not bool(jax.device_get(self.solved)):
            return []

        solved_current = self.solved_idx
        first_leaf = jax.tree_util.tree_leaves(solved_current)[0]
        if first_leaf.ndim > 0:
            solved_current = solved_current[0]

        solved_index = int(np.asarray(jax.device_get(solved_current.hashidx.index)).reshape(-1)[0])
        max_steps = max(1, int(jax.device_get(self.generated_size)) + 1)

        parent_indices = np.asarray(jax.device_get(self.parent.hashidx.index)).reshape(-1)
        costs = np.asarray(jax.device_get(self.cost)).reshape(-1)
        node_count = costs.shape[0]

        path: list[Current | Parent] = [solved_current]
        if solved_index < 0 or solved_index >= node_count:
            return path

        visited = {solved_index}
        current_index = solved_index

        for _ in range(max_steps):
            if current_index < 0 or current_index >= node_count:
                break

            current_cost = float(costs[current_index])
            if (not np.isfinite(current_cost)) or current_cost <= 0.0:
                break

            parent_index = int(parent_indices[current_index])
            if parent_index < 0 or parent_index >= node_count or parent_index in visited:
                break

            path.append(self.parent[current_index])
            visited.add(parent_index)
            current_index = parent_index

        path.reverse()
        return path

    def insert_batch(
        self,
        keys: chex.Array,
        vals: Any,
        masks: chex.Array | None = None,
    ) -> "SearchResult":
        """Insert batches into the priority queue."""
        # vals can be Current or ParentWithCosts.
        # masks is optional; usually implied by inf keys, but if provided, we use it.
        # Logic adapted from common.py/search_base.py

        flat_keys = keys.flatten()
        flat_vals = vals.flatten() if hasattr(vals, "flatten") else vals
        if masks is not None:
            flat_masks = masks.flatten()
        else:
            flat_masks = jnp.isfinite(flat_keys)  # Default mask

        # Pad to batch_size multiple
        total_size = flat_keys.shape[0]
        batch_size = self.batch_size
        pad_len = (-total_size) % batch_size

        if pad_len > 0:
            flat_keys = jnp.pad(flat_keys, (0, pad_len), constant_values=jnp.inf)
            flat_vals = xnp.pad(flat_vals, (0, pad_len))
            flat_masks = jnp.pad(flat_masks, (0, pad_len), constant_values=False)

        num_chunks = flat_keys.shape[0] // batch_size
        keys_reshaped = flat_keys.reshape(num_chunks, batch_size)
        vals_reshaped = flat_vals.reshape(num_chunks, batch_size)
        masks_reshaped = flat_masks.reshape(num_chunks, batch_size)

        def _scan(sr, inputs):
            k, v, m = inputs
            # Insert only valid items? BGPQ usually handles insertion of masked items if we implement it so?
            # Or just rely on inf key.
            # search_base.py used conditional insert if ANY mask is true.

            def _do_insert(pq):
                return pq.insert(k, v)

            new_pq = jax.lax.cond(jnp.any(m), _do_insert, lambda pq: pq, sr.priority_queue)
            sr.priority_queue = new_pq
            return sr, None

        new_sr, _ = jax.lax.scan(_scan, self, (keys_reshaped, vals_reshaped, masks_reshaped))
        return new_sr

    def pop_full(self) -> Tuple["SearchResult", Current, chex.Array]:
        """
        Pop a full batch of unique states with lowest costs.
        Used by Eager Expansion (A*).
        """
        # Logic ported from search_base.py pop_full

        # 1. Pop initial batch
        pq, min_key, min_val = self.priority_queue.delete_mins()
        self.priority_queue = pq
        min_key = _mask_unoptimal_current(self, min_key, min_val)

        batch_size = self.batch_size
        min_filled_count = max(1, int(math.ceil(POP_BATCH_FILLED_RATIO * batch_size)))

        # We need _expand_and_filter logic here.
        # For Eager, min_val is Current.
        # We extract states from HT.

        def _extract_state(sr, val):
            return sr.get_state(val), sr.get_cost(val), sr.get_dist(val), val

        states, costs, dists, _ = _extract_state(self, min_val)

        # Unique mask
        filled = jnp.isfinite(min_key)
        unique_mask = xnp.unique_mask(states, costs, filled)
        min_key = jnp.where(unique_mask, min_key, jnp.inf)

        # Buffer
        buffer_key = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current.default((batch_size,))  # Assuming pq_val_type is Current

        # Loop
        def _cond(state):
            sr, _, _, _, key, _, _, _ = state
            pq_not_empty = sr.priority_queue.size > 0
            filled_count = jnp.sum(jnp.isfinite(key))
            return jnp.logical_and(pq_not_empty, filled_count < min_filled_count)

        def _body(state):
            sr, cur_states, cur_costs, cur_dists, key, val, _, _ = state
            pq, new_key, new_val = sr.priority_queue.delete_mins()
            sr.priority_queue = pq
            new_key = _mask_unoptimal_current(sr, new_key, new_val)

            new_states, new_costs, new_dists, _ = _extract_state(sr, new_val)

            # Merge
            stack_states = xnp.concatenate((cur_states, new_states))
            stack_costs = jnp.concatenate((cur_costs, new_costs))
            stack_dists = jnp.concatenate((cur_dists, new_dists))
            stack_key = jnp.concatenate((key, new_key))
            stack_val = xnp.concatenate((val, new_val))

            # Unique
            stack_filled = jnp.isfinite(stack_key)
            stack_mask = xnp.unique_mask(stack_states, stack_costs, stack_filled)
            stack_key = jnp.where(stack_mask, stack_key, jnp.inf)

            # Sort
            stack_size = batch_size * 2
            sorted_key, sorted_idx = jax.lax.sort_key_val(stack_key, jnp.arange(stack_size))
            sorted_val = stack_val[sorted_idx]
            sorted_states = stack_states[sorted_idx]
            sorted_costs = stack_costs[sorted_idx]
            sorted_dists = stack_dists[sorted_idx]

            return (
                sr,
                sorted_states[:batch_size],
                sorted_costs[:batch_size],
                sorted_dists[:batch_size],
                sorted_key[:batch_size],
                sorted_val[:batch_size],
                sorted_key[batch_size:],
                sorted_val[batch_size:],
            )

        (
            new_sr,
            final_states,
            final_costs,
            final_dists,
            final_key,
            final_val,
            overflow_keys,
            overflow_vals,
        ) = jax.lax.while_loop(
            _cond,
            _body,
            (self, states, costs, dists, min_key, min_val, buffer_key, buffer_val),
        )
        self = new_sr

        # Re-insert overflow
        self = self.insert_batch(overflow_keys, overflow_vals)

        # Pop ratio
        process_mask = _compute_pop_process_mask(final_key, self.pop_ratio, self.min_pop)

        # Return masked out nodes to PQ
        return_keys = jnp.where(process_mask, jnp.inf, final_key)
        self = self.insert_batch(return_keys, final_val)

        # Update generation statistics
        hash_idx = final_val.hashidx
        self.pop_generation = xnp.update_on_condition(
            self.pop_generation, hash_idx.index, process_mask, self.pop_count
        )
        self.pop_count += 1

        # Return
        # We need to filter final_val to only contain valid ones?
        # Typically we return the whole batch and a mask.
        return self, final_val, process_mask

    def pop_full_with_actions(
        self,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        use_heuristic: bool = False,
        is_backward: bool = False,
    ) -> Tuple["SearchResult", Current, Puzzle.State, chex.Array]:
        """
        Pop a full batch, expanding (parent, action) to states.
        Used by Deferred Expansion (A*d, Q*).
        """

        # Logic for Deferred pop
        # Helper to expand state from parent+action
        def _expand_and_filter_actions(sr, key, val):
            # val is ParentWithCosts
            parent = val.parent
            parent_states = sr.hashtable[parent.hashidx]
            filled_actions = jnp.isfinite(key)

            if is_backward:
                if hasattr(puzzle, "inverse_action_map"):
                    inv_actions = puzzle.inverse_action_map[parent.action.astype(jnp.int32)]
                    next_states, step_costs = puzzle.batched_get_actions(
                        solve_config,
                        parent_states,
                        inv_actions,
                        filled_actions,
                    )
                else:
                    all_inv_states, all_inv_costs = puzzle.batched_get_inverse_neighbours(
                        solve_config,
                        parent_states,
                        filled_actions,
                    )
                    from JAxtar.core.common import (
                        normalize_neighbour_cost_layout,
                        resolve_neighbour_layout,
                    )

                    all_inv_states, all_inv_costs = normalize_neighbour_cost_layout(
                        all_inv_states,
                        all_inv_costs,
                        self.action_size,
                        parent.action.shape[0],
                        layout=resolve_neighbour_layout(puzzle, is_backward=True),
                    )
                    parent_actions = parent.action.astype(jnp.int32)
                    batch_idx = jnp.arange(parent_actions.shape[0], dtype=jnp.int32)
                    next_states = all_inv_states[parent_actions, batch_idx]
                    step_costs = all_inv_costs[parent_actions, batch_idx]
            else:
                next_states, step_costs = puzzle.batched_get_actions(
                    solve_config,
                    parent_states,
                    parent.action,
                    filled_actions,
                )

            parent_costs = sr.get_cost(parent)
            current_costs = parent_costs + step_costs
            current_dists = jnp.where(use_heuristic, val.dist, val.dist - step_costs)

            return next_states, current_costs, current_dists, key

        # ... (Similar while loop structure but using _expand_and_filter_actions)
        # And importantly, AFTER getting final unique batch, we INSERT them into HT.

        # 1. Pop initial
        pq, min_key, min_val = self.priority_queue.delete_mins()
        self.priority_queue = pq

        batch_size = self.batch_size
        min_filled_count = max(1, int(math.ceil(POP_BATCH_FILLED_RATIO * batch_size)))

        # Expand acts
        states, costs, dists, min_key = _expand_and_filter_actions(self, min_key, min_val)

        # Unique check (using HT lookup parallel?)
        # Deferred expansion means we check HT now.
        filled = jnp.isfinite(min_key)

        # We also need to check internal uniqueness in the batch
        unique_mask = xnp.unique_mask(states, costs, filled)

        # We ALSO need to check against global HT (to avoid duplicates)
        # check if states are already in HT with better/equal cost.
        hash_idxs, found = self.hashtable.lookup_parallel(states, unique_mask)
        old_costs = self.get_cost(hash_idxs)

        # Valid if: (Unique in batch) AND (Not found OR Cost < OldCost)
        better = jnp.less(costs, old_costs)
        valid_mask = unique_mask & jnp.logical_or(~found, better)

        min_key = jnp.where(valid_mask, min_key, jnp.inf)

        # Loop ...
        buffer_key = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = ParentWithCosts.default((batch_size,))

        def _cond(state):
            # same cond
            sr, _, _, _, key, _, _, _ = state
            return jnp.logical_and(
                sr.priority_queue.size > 0,
                jnp.sum(jnp.isfinite(key)) < min_filled_count,
            )

        def _body(state):
            # same loop body structure but using _expand_and_filter_actions and valid_mask logic
            sr, cur_states, cur_costs, cur_dists, key, val, _, _ = state
            pq, new_key, new_val = sr.priority_queue.delete_mins()
            sr.priority_queue = pq

            new_states, new_costs, new_dists, new_key = _expand_and_filter_actions(
                sr, new_key, new_val
            )

            # Stack
            stack_states = xnp.concatenate((cur_states, new_states))
            stack_costs = jnp.concatenate((cur_costs, new_costs))
            stack_dists = jnp.concatenate((cur_dists, new_dists))
            stack_key = jnp.concatenate((key, new_key))
            stack_val = xnp.concatenate((val, new_val))

            # Unique & Global check
            stack_filled = jnp.isfinite(stack_key)
            stack_uniq = xnp.unique_mask(stack_states, stack_costs, stack_filled)

            # Global check
            s_hash_idxs, s_found = sr.hashtable.lookup_parallel(stack_states, stack_uniq)
            s_old_costs = sr.get_cost(s_hash_idxs)
            s_valid = stack_uniq & jnp.logical_or(~s_found, jnp.less(stack_costs, s_old_costs))

            stack_key = jnp.where(s_valid, stack_key, jnp.inf)

            # Sort
            stack_size = batch_size * 2
            sorted_key, sorted_idx = jax.lax.sort_key_val(stack_key, jnp.arange(stack_size))
            sorted_val = stack_val[sorted_idx]
            sorted_states = stack_states[sorted_idx]
            sorted_costs = stack_costs[sorted_idx]
            sorted_dists = stack_dists[sorted_idx]

            return (
                sr,
                sorted_states[:batch_size],
                sorted_costs[:batch_size],
                sorted_dists[:batch_size],
                sorted_key[:batch_size],
                sorted_val[:batch_size],
                sorted_key[batch_size:],
                sorted_val[batch_size:],
            )

        (
            new_sr,
            final_states,
            final_costs,
            final_dists,
            final_key,
            final_val,
            overflow_keys,
            overflow_vals,
        ) = jax.lax.while_loop(
            _cond,
            _body,
            (self, states, costs, dists, min_key, min_val, buffer_key, buffer_val),
        )
        self = new_sr
        self = self.insert_batch(overflow_keys, overflow_vals)

        process_mask = _compute_pop_process_mask(final_key, self.pop_ratio, self.min_pop)
        return_keys = jnp.where(process_mask, jnp.inf, final_key)
        final_costs = jnp.where(process_mask, final_costs, jnp.inf)
        final_dists = jnp.where(process_mask, final_dists, jnp.inf)
        self = self.insert_batch(return_keys, final_val)

        # INSERT Valid processed states into HT
        # final_states are the child states.
        # final_costs are the g-values.
        # final_val.parent is the parent info.

        # We masked them with process_mask.

        (self.hashtable, _, _, hash_idx) = self.hashtable.parallel_insert(
            final_states, process_mask, final_costs
        )

        # Update cost/dist/parent
        # Note: parallel_insert might have found existing states.
        # But we already filtered by cost so we should overwrite if cheaper.
        # HashTable usually handles "insert if new or key match".
        # We need to explicitly update cost/parent.

        self.cost = xnp.update_on_condition(self.cost, hash_idx.index, process_mask, final_costs)
        # Use stored dist (heuristic of child, if available) or inf?
        # In deferred, we computed dist (heuristic) during expansion (in astar_d loop).
        # So final_dists IS the heuristic of the child.
        self.dist = xnp.update_on_condition(self.dist, hash_idx.index, process_mask, final_dists)
        self.parent = self.parent.at[hash_idx.index].set_as_condition(
            process_mask, final_val.parent
        )

        self.pop_generation = xnp.update_on_condition(
            self.pop_generation, hash_idx.index, process_mask, self.pop_count
        )
        self.pop_count += 1

        final_current = Current(hashidx=hash_idx, cost=final_costs)

        return self, final_current, final_states, process_mask

    @staticmethod
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        action_size: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        seed=42,
        pq_val_type: type = Current,
        hash_size_multiplier: int = HASH_SIZE_MULTIPLIER,
    ):
        hashtable = HashTable.build(statecls, seed, max_nodes, CUCKOO_TABLE_N, hash_size_multiplier)
        priority_queue = BGPQ.build(max_nodes, batch_size, pq_val_type, KEY_DTYPE)

        cost = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        dist = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        parent = Parent.default(hashtable.table.shape.batch)
        solved = jnp.array(False)
        solved_idx = Current.default((1,))
        pop_generation = jnp.full(hashtable.table.shape.batch, -1, dtype=jnp.int32)
        pop_count = jnp.array(0, dtype=jnp.int32)

        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            action_size=action_size,
            pop_ratio=jnp.maximum(1.0 + pop_ratio, 1.01),
            min_pop=min_pop,
            cost=cost,
            dist=dist,
            parent=parent,
            solved=solved,
            solved_idx=solved_idx,
            pop_generation=pop_generation,
            pop_count=pop_count,
        )


def _compute_pop_process_mask(
    key: chex.Array,
    pop_ratio: float,
    min_pop: int,
) -> chex.Array:
    """Compute which popped entries are processed now vs. re-queued."""
    filled = jnp.isfinite(key)
    # If key is empty (all inf), mask is all False.
    # Logic:
    # 1. Sort helps, but keys here from while_loop are likely sorted.
    # 2. Threshold based on best key.

    # We need at least 1 valid key to compute threshold.
    # If all inf, threshold is inf.

    min_k = jnp.min(key)  # Assuming sorted? keys[0]
    # But keys might be all inf.
    threshold = min_k * pop_ratio + 1e-6

    process_mask = jnp.less_equal(key, threshold)
    base_process_mask = jnp.logical_and(filled, process_mask)

    # Also ensure min_pop
    min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= min_pop, filled)

    return jnp.logical_or(base_process_mask, min_pop_mask)


def _mask_unoptimal_current(
    search_result: SearchResult,
    popped_keys: chex.Array,
    popped_current: Current,
) -> chex.Array:
    """Mask stale Current PQ entries whose popped cost is worse than HT best cost."""
    best_costs = search_result.get_cost(popped_current)
    is_optimal = jnp.less_equal(popped_current.cost, best_costs)
    return jnp.where(is_optimal, popped_keys, jnp.inf)
