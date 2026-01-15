"""
Search Base for Iterative Deepening Algorithms (IDA* / ID-Q*)
"""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass
from xtructure.stack import Stack

from JAxtar.annotate import KEY_DTYPE


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

    # Solution info if found during frontier generation
    solved: chex.Array  # bool scalar
    solution_state: Xtructurable  # [1, state_shape]
    solution_cost: chex.Array  # scalar


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
    stack: Stack

    solution_state: Xtructurable  # Single state
    solution_cost: chex.Array  # Scalar - cost of solution
    generated_count: chex.Array  # Scalar - count of generated nodes

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def build(
        statecls: Puzzle.State,
        capacity: int,
        action_size: int,
    ) -> "IDSearchResult":
        """
        Initialize the IDSearchResult with empty stack.
        """
        # Define dynamic Item class
        @xtructure_dataclass
        class IDStackItem:
            state: FieldDescriptor.scalar(dtype=statecls)
            cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
            depth: FieldDescriptor.scalar(dtype=jnp.int32)
            action: FieldDescriptor.scalar(dtype=jnp.int32)

        bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        next_bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)

        stack = Stack.build(capacity, IDStackItem)

        solution_state = statecls.default((1,))  # Placeholder batch-1
        solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
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
            solution_state=solution_state,
            solution_cost=solution_cost,
            generated_count=generated_count,
        )

    @property
    def stack_ptr(self):
        return self.stack.size

    def get_solved_path(self):
        """
        Return the solved path.
        Current implementation only returns [solution_state].
        """
        return [self.solution_state[0]]

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
    ) -> tuple["IDSearchResult", chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
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

        indices = jnp.zeros(batch_size, dtype=jnp.int32)

        new_self = self.replace(stack=new_stack)

        return new_self, states, costs, depths, valid_mask, indices

    def push_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        actions: chex.Array,
        valid_mask: chex.Array,
    ) -> "IDSearchResult":
        """
        Push a batch of items onto the stack.
        Only pushes items where valid_mask is True.
        """
        n_push = jnp.sum(valid_mask.astype(jnp.int32))

        sort_keys = jnp.where(valid_mask, 0, 1)
        perm = jnp.argsort(sort_keys)

        states_sorted = states[perm]
        costs_sorted = costs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]

        current_ptr = self.stack.size.astype(jnp.int32)
        capacity = self.stack.max_size

        safe_n_push = jnp.minimum(n_push, capacity - current_ptr)

        items_sorted = self.ItemCls(
            state=states_sorted, cost=costs_sorted, depth=depths_sorted, action=actions_sorted
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

        items = self.ItemCls(state=states, cost=costs, depth=depths, action=actions)

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
