"""
Search Base for Iterative Deepening Algorithms (IDA* / ID-Q*)
"""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import Xtructurable, base_dataclass

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


@base_dataclass(static_fields=("capacity", "action_size"))
class IDSearchResult:
    """
    Data structure for Iterative Deepening Search.
    Maintains an explicit stack for Batched DFS.
    """

    capacity: int
    action_size: int

    # Global search state
    bound: chex.Array  # Current cost bound (scalar)
    next_bound: chex.Array  # Next cost bound (scalar)
    solved: chex.Array  # Boolean scalar
    solved_idx: chex.Array  # Index in the stack of the solution (if found)

    # Stack storage (Fixed size = capacity)
    # Allows LIFO access for DFS
    stack_ptr: chex.Array  # Current number of items in stack (scalar int32)

    stack_state: Xtructurable  # States [capacity, ...]
    stack_cost: chex.Array  # g-values [capacity]
    stack_depth: chex.Array  # depth/level [capacity] (int32)
    # Note: For full path reconstruction, we might need parent pointers,
    # but for high-depth IDA*, storing full tree prevents linear memory.
    # We store minimal info. Reconstruction can be done by re-running or
    # using a very limited trace if depth is small.
    # For now, we focus on finding the solution.

    # We can store the 'action' that led to this state to help 1-step reconstruction
    stack_action_from_parent: chex.Array  # [capacity]

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
        bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        next_bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)
        stack_ptr = jnp.array(0, dtype=jnp.int32)

        stack_state = statecls.default((capacity,))
        stack_cost = jnp.full((capacity,), jnp.inf, dtype=KEY_DTYPE)
        stack_depth = jnp.full((capacity,), 0, dtype=jnp.int32)
        stack_action_from_parent = jnp.full((capacity,), -1, dtype=jnp.int32)
        solution_state = statecls.default((1,))  # Placeholder batch-1
        solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        generated_count = jnp.array(0, dtype=jnp.int32)

        return IDSearchResult(
            capacity=capacity,
            action_size=action_size,
            bound=bound,
            next_bound=next_bound,
            solved=solved,
            solved_idx=solved_idx,
            stack_ptr=stack_ptr,
            stack_state=stack_state,
            stack_cost=stack_cost,
            stack_depth=stack_depth,
            stack_action_from_parent=stack_action_from_parent,
            solution_state=solution_state,
            solution_cost=solution_cost,
            generated_count=generated_count,
        )

    def get_solved_path(self):
        """
        Return the solved path.
        Current implementation only returns [solution_state].
        """
        # We need to return a list of states.
        # Since 'solution_state' is a batched state of size 1, we access index 0.
        return [self.solution_state[0]]

    def get_generated_size(self):
        return self.generated_count

    @property
    def generated_size(self):
        # Needed for CLI stats
        return self.generated_count

    def get_cost(self, idx: int):
        # Return the actual cost of the solution.
        return self.solution_cost

    def get_state(self, idx: int):
        # CLI uses this to print solution state if no path logic.
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
        # Calculate how many we can pop
        avail = self.stack_ptr
        n_pop = jnp.minimum(avail, batch_size)

        # New pointer
        new_ptr = avail - n_pop

        # Indices to gather: [new_ptr, new_ptr + batch_size)
        # We want LIFO, so effectively the "top" items are at the end of the active region.
        # Let's just take the range [new_ptr, new_ptr + batch_size).
        # Mask out indices >= avail (in case n_pop < batch_size)

        indices = jnp.arange(batch_size, dtype=jnp.int32) + new_ptr
        valid_mask = indices < avail

        # Gather
        # Use simple indexing since indices are contiguous range?
        # Actually dynamic slicing is cleaner in JAX: jax.lax.dynamic_slice
        # But `Xtructurable` might not support dynamic_slice directly easily unless implemented.
        # Fallback to scatter/gather style or simple boolean masking if batch_size is static.
        # Since batch_size is static compilation arg usually, we can use simple indexing.

        # To keep it robust against non-contiguous (if we change logic), we use array indexing.
        states = self.stack_state[indices]
        costs = self.stack_cost[indices]
        depths = self.stack_depth[indices]

        # Update self
        new_self = self.replace(stack_ptr=new_ptr)

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
        # Compress the batch to contiguous valid items
        # This is expensive (sort/permute), but necessary for stack management.

        # 1. Count valid items
        n_push = jnp.sum(valid_mask.astype(jnp.int32))

        # 2. Check overflow
        # If stack_ptr + n_push > capacity, we drop items
        # For IDA*, dropping essentially breaks the search completeness but prevents crash.
        # We can clip n_push or allow circular buffer? Stack usually implies crash on overflow.
        # Let's just clip and maybe warn (but print in JAX is hard).
        safe_n_push = jnp.minimum(n_push, self.capacity - self.stack_ptr)
        start_idx = self.stack_ptr

        # 3. Compact the input batch
        # Sort so that valid items come first
        sort_keys = jnp.where(valid_mask, 0, 1)  # 0 for valid, 1 for invalid
        perm = jnp.argsort(sort_keys)

        # Permute inputs
        states_sorted = states[perm]
        costs_sorted = costs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]

        # 4. Insert into stack using dynamic update
        # We only define indices for the valid portion
        # Since `dynamic_update_slice` requires contiguous update, we can use that!

        # Use `dynamic_update_slice(stack, states_sorted, (start_idx,))`
        # This writes ALL of states_sorted.
        # But states_sorted contains garbage at the end (invalid items).
        # This is fine IF we consider everything after `stack_ptr` as garbage anyway!
        # YES! As long as we update `stack_ptr` correctly, we don't care what's in the memory beyond it.

        # Safe guard: if start_idx + batch_size > capacity, dynamic_update_slice crops automatically?
        # JAX documentation says "sliced if it overflows".

        # So:
        # 1. Sort inputs (valid first).
        # 2. dynamic_update_slice at stack_ptr.
        # 3. Update stack_ptr.

        # NOTE: `states` is Xtructurable. `dynamic_update_slice` for Xtructurable might be supported or use tree_map.

        def _update_leaf(stack_arr, update_arr):
            start_indices = (start_idx,) + (0,) * (stack_arr.ndim - 1)
            return jax.lax.dynamic_update_slice(stack_arr, update_arr, start_indices)

        new_stack_state = jax.tree_util.tree_map(_update_leaf, self.stack_state, states_sorted)
        new_stack_cost = _update_leaf(self.stack_cost, costs_sorted)
        new_stack_depth = _update_leaf(self.stack_depth, depths_sorted)
        new_stack_action = _update_leaf(self.stack_action_from_parent, actions_sorted)

        # Update ptr and generated count
        final_ptr = start_idx + safe_n_push
        new_generated_count = self.generated_count + safe_n_push

        return self.replace(
            stack_state=new_stack_state,
            stack_cost=new_stack_cost,
            stack_depth=new_stack_depth,
            stack_action_from_parent=new_stack_action,
            stack_ptr=final_ptr,
            generated_count=new_generated_count,
        )
