"""
Search Base for Iterative Deepening Algorithms (IDA* / ID-Q*)
"""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import (
    FieldDescriptor,
    HashTable,
    Xtructurable,
    base_dataclass,
    xtructure_dataclass,
)
from xtructure.stack import Stack

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)

ACTION_PAD = jnp.array(jnp.iinfo(ACTION_DTYPE).max, dtype=ACTION_DTYPE)


@base_dataclass
class IDFrontier:
    """
    Data structure for the pre-computed frontier.
    Used to restart IDA* from a deeper set of nodes.
    """

    states: Xtructurable  # [frontier_size, ...]
    costs: chex.Array  # [frontier_size]
    depths: chex.Array  # [frontier_size]
    valid_mask: chex.Array  # [frontier_size]
    f_scores: chex.Array  # [frontier_size]
    trail: Xtructurable  # [frontier_size, non_backtracking_steps]
    action_history: chex.Array  # [frontier_size, max_path_len]

    # Solution info if found during frontier generation
    solved: chex.Array  # bool scalar
    solution_state: Xtructurable  # [1, state_shape]
    solution_cost: chex.Array  # scalar
    solution_actions_arr: chex.Array  # [max_path_len]


def compact_by_valid(
    values: Any,
    valid_mask: chex.Array,
) -> tuple[Any, chex.Array, chex.Array, chex.Array]:
    """
    Pack valid entries to the front for use with variable_batch_switcher.

    Returns:
        Tuple of (packed_values, packed_valid_mask, valid_count, packed_indices).
    """
    flat_size = valid_mask.shape[0]
    valid_idx = jnp.nonzero(valid_mask, size=flat_size, fill_value=0)[0].astype(jnp.int32)
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    packed_values = xnp.take(values, valid_idx, axis=0)
    packed_valid = jnp.arange(flat_size, dtype=jnp.int32) < valid_count
    return packed_values, packed_valid, valid_count, valid_idx


@base_dataclass(static_fields=("params"))
class IDLoopState:
    """
    Loop state for the inner DFS loop of IDA*.
    """

    search_result: "IDSearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    frontier: IDFrontier


@base_dataclass(static_fields=("capacity", "action_size", "ItemCls"))
class IDSearchResult:
    """
    Data structure for Iterative Deepening Search.
    Maintains an explicit stack for Batched DFS.
    """

    capacity: int
    action_size: int
    ItemCls: type

    # Global search state
    bound: chex.Array  # Current cost bound (scalar)
    next_bound: chex.Array  # Next cost bound (scalar)
    solved: chex.Array  # Boolean scalar
    solved_idx: chex.Array  # Index in the stack of the solution (if found)

    # Stack storage (Fixed size = capacity)
    # Allows LIFO access for DFS
    # Stack storage (Fixed size = capacity)
    # Allows LIFO access for DFS
    stack: Stack
    hashtable: HashTable
    ht_cost: chex.Array
    ht_dist: chex.Array

    solution_state: Xtructurable  # Single state
    start_state: Xtructurable  # Single state (start)
    solution_cost: chex.Array  # Scalar - cost of solution
    generated_count: chex.Array  # Scalar - count of generated nodes
    solution_actions_arr: chex.Array  # [max_path_len]

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
    def build(
        statecls: Puzzle.State,
        capacity: int,
        action_size: int,
        non_backtracking_steps: int = 0,
        max_path_len: int = 256,
        seed: int = 42,
    ) -> "IDSearchResult":
        """
        Initialize the IDSearchResult with empty stack.
        """
        # Define dynamic Item class
        if non_backtracking_steps < 0:
            raise ValueError("non_backtracking_steps must be non-negative")

        trail_shape = (int(non_backtracking_steps),)

        @xtructure_dataclass
        class IDStackItem:
            state: FieldDescriptor.scalar(dtype=statecls)
            cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
            depth: FieldDescriptor.scalar(dtype=jnp.int32)
            action: FieldDescriptor.scalar(dtype=jnp.int32)
            trail: FieldDescriptor.tensor(dtype=statecls, shape=trail_shape)
            action_history: FieldDescriptor.tensor(dtype=jnp.int32, shape=(max_path_len,))

        bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        next_bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)

        stack = Stack.build(capacity, IDStackItem)
        hashtable = HashTable.build(statecls, seed, capacity, CUCKOO_TABLE_N, HASH_SIZE_MULTIPLIER)
        ht_cost = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        ht_dist = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)

        solution_state = statecls.default((1,))  # Placeholder batch-1
        start_state = statecls.default((1,))  # Placeholder batch-1
        solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solution_actions = jnp.full((max_path_len,), ACTION_PAD, dtype=jnp.int32)
        generated_count = jnp.array(0, dtype=jnp.int32)

        return IDSearchResult(
            capacity=capacity,
            action_size=action_size,
            ItemCls=IDStackItem,
            bound=bound,
            next_bound=next_bound,
            solved=solved,
            solved_idx=solved_idx,
            stack=stack,
            hashtable=hashtable,
            ht_cost=ht_cost,
            ht_dist=ht_dist,
            solution_state=solution_state,
            start_state=start_state,
            solution_cost=solution_cost,
            solution_actions_arr=solution_actions,
            generated_count=generated_count,
        )

    @property
    def stack_ptr(self):
        return self.stack.size

    def get_solved_path(self, puzzle: Puzzle = None, solve_config: Puzzle.SolveConfig = None):
        """
        Return the solved path.
        Requires 'puzzle' and 'solve_config' to reconstruct the path by replaying actions.
        """
        if not self.solved:
            return []

        if puzzle is None or solve_config is None:
            # Fallback for when puzzle is not provided (legacy/compat)
            # Cannot reconstruct without transition model
            # But the user requested a fix, so detailed reconstruction is the goal.
            return [self.solution_state[0]]

        # Reconstruct path
        path = [self.start_state[0]]
        actions = self.solution_actions()

        # We need to run on host or jit? get_solved_path is typically host-side.
        # But we need puzzle logic.
        # We can use a small jitted function to step.

        curr = xnp.expand_dims(self.start_state[0], 0)  # (1, ...)

        # Prepare step function
        @jax.jit
        def step_fn(acc_state, action):
            # batched_get_actions expects (solve_config, states, actions, valid)
            # actions should be shape (batch_size,) i.e., (1,)
            act_batch = jnp.array([action], dtype=jnp.int32)
            valid_batch = jnp.array([True], dtype=jnp.bool_)
            next_states, _ = puzzle.batched_get_actions(
                solve_config, acc_state, act_batch, valid_batch
            )
            return next_states

        # Iterate actions on host (safe since path length is small)
        # Note: actions are JAX arrays. Move to cpu.
        actions_cpu = jax.device_get(actions)
        for i, act in enumerate(actions_cpu):
            curr = step_fn(curr, act)
            state_single = curr[0]
            path.append(state_single)

        return path

    def solution_actions(self):
        """
        Return the list of actions to reach the solution.
        """
        # Filter out padding (ACTION_PAD)
        valid_mask = self.solution_actions_arr != ACTION_PAD
        return self.solution_actions_arr[valid_mask]

    def get_generated_size(self):
        return self.generated_count

    @property
    def generated_size(self):
        return self.generated_count

    def get_cost(self, idx: int):
        return self.solution_cost

    def get_state(self, idx: int):
        return self.solution_state[0]

    def get_dist(self, idx: int):
        return 0.0

    def get_top_batch(
        self, batch_size: int
    ) -> tuple[
        "IDSearchResult",
        chex.Array,
        chex.Array,
        chex.Array,
        Xtructurable,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        """
        Pop the top `batch_size` items from the stack.
        Returns (updated_self, states, costs, depths, valid_mask, indices)
        """
        # Ensure consistent index type for JAX
        current_size = self.stack.size.astype(jnp.int32)

        new_stack, popped_items = self.stack.pop(batch_size)

        valid_count = jnp.minimum(current_size, batch_size)
        indices_range = jnp.arange(batch_size, dtype=jnp.int32)
        valid_mask = indices_range < valid_count

        # Recast back to stack native size type if needed
        actual_new_size = jnp.maximum(0, current_size - batch_size).astype(self.stack.size.dtype)
        new_stack = new_stack.replace(size=actual_new_size)

        states = popped_items.state
        costs = popped_items.cost
        depths = popped_items.depth
        trails = popped_items.trail

        indices = jnp.zeros(batch_size, dtype=jnp.int32)

        new_self = self.replace(stack=new_stack)

        action_histories = popped_items.action_history

        return new_self, states, costs, depths, trails, action_histories, valid_mask, indices

    def reset_tables(self, statecls: Puzzle.State, seed: int = 42) -> "IDSearchResult":
        """
        Resets the hashtable and associated cost/dist arrays.
        Useful for Iterative Deepening to clear visited set between bounds.
        """
        hashtable = HashTable.build(
            statecls, seed, self.capacity, CUCKOO_TABLE_N, HASH_SIZE_MULTIPLIER
        )
        ht_cost = jnp.full_like(self.ht_cost, jnp.inf)
        ht_dist = jnp.full_like(self.ht_dist, jnp.inf)
        return self.replace(hashtable=hashtable, ht_cost=ht_cost, ht_dist=ht_dist)

    def push_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        actions: chex.Array,
        trails: Xtructurable,
        action_histories: chex.Array,
        valid_mask: chex.Array,
    ) -> "IDSearchResult":
        """
        Push a batch of items onto the stack.
        Only pushes items where valid_mask is True.
        """
        n_push = jnp.sum(valid_mask.astype(jnp.int32))

        sort_keys = jnp.where(valid_mask, 0, 1)
        perm = jnp.argsort(sort_keys)

        states_sorted = xnp.take(states, perm, axis=0)
        costs_sorted = costs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]

        current_ptr = self.stack.size.astype(jnp.int32)
        capacity = self.stack.max_size

        safe_n_push = jnp.minimum(n_push, capacity - current_ptr)

        items_sorted = self.ItemCls(
            state=states_sorted,
            cost=costs_sorted,
            depth=depths_sorted,
            action=actions_sorted,
            trail=xnp.take(trails, perm, axis=0),
            action_history=action_histories[perm],
        )

        def _update_leaf(stack_arr, update_arr):
            # Explicitly use int32 indices
            start_indices = (current_ptr,) + (0,) * (stack_arr.ndim - 1)
            return jax.lax.dynamic_update_slice(stack_arr, update_arr, start_indices)

        new_val_store = jax.tree_util.tree_map(_update_leaf, self.stack.val_store, items_sorted)

        new_ptr = (current_ptr + safe_n_push).astype(self.stack.size.dtype)
        new_generated_count = self.generated_count + safe_n_push

        new_stack = self.stack.replace(val_store=new_val_store, size=new_ptr)

        return self.replace(
            stack=new_stack,
            generated_count=new_generated_count,
        )

    def push_packed_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        actions: chex.Array,
        trails: Xtructurable,
        action_histories: chex.Array,
        n_push: chex.Array,
    ) -> "IDSearchResult":
        """
        Push a pre-packed batch of items onto the stack.
        Assumes the items are already sorted/packed such that the valid items
        to be pushed are at the beginning of the arrays.
        """
        current_ptr = self.stack.size.astype(jnp.int32)
        capacity = self.stack.max_size

        safe_n_push = jnp.minimum(n_push.astype(jnp.int32), capacity - current_ptr)

        items = self.ItemCls(
            state=states,
            cost=costs,
            depth=depths,
            action=actions,
            trail=trails,
            action_history=action_histories,
        )

        def _update_leaf(stack_arr, update_arr):
            start_indices = (current_ptr,) + (0,) * (stack_arr.ndim - 1)
            return jax.lax.dynamic_update_slice(stack_arr, update_arr, start_indices)

        new_val_store = jax.tree_util.tree_map(_update_leaf, self.stack.val_store, items)

        new_ptr = (current_ptr + safe_n_push).astype(self.stack.size.dtype)
        new_generated_count = self.generated_count + safe_n_push

        new_stack = self.stack.replace(val_store=new_val_store, size=new_ptr)

        return self.replace(
            stack=new_stack,
            generated_count=new_generated_count,
        )
