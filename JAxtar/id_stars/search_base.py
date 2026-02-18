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
from xtructure import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass
from xtructure.stack import Stack

from helpers.jax_compile import jit_with_warmup
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.id_stars.id_frontier import (
    ACTION_PAD,
    IDFrontier,
    build_flat_children,
    compact_by_valid,
)
from JAxtar.utils.array_ops import stable_partition_three


def _batched_state_equal(lhs: Puzzle.State, rhs: Puzzle.State) -> jnp.ndarray:
    equality_tree = lhs == rhs
    leaves, _ = jax.tree_util.tree_flatten(equality_tree)
    if not leaves:
        raise ValueError("State comparison received an empty tree")
    result = leaves[0]
    for leaf in leaves[1:]:
        result = jnp.logical_and(result, leaf)
    return result


def apply_non_backtracking(
    candidate_states: Puzzle.State,
    parent_states: Puzzle.State,
    parent_trail: Puzzle.State,
    parent_depths: jnp.ndarray,
    valid_mask: jnp.ndarray,
    non_backtracking_steps: int,
    action_size: int,
    flat_size: int,
    trail_indices: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    if non_backtracking_steps <= 0:
        return valid_mask

    parent_states_tiled = xnp.stack([parent_states] * action_size, axis=0)
    flat_parent_states = xnp.reshape(parent_states_tiled, (flat_size,))
    blocked = _batched_state_equal(candidate_states, flat_parent_states)
    blocked = jnp.logical_and(blocked, valid_mask)

    parent_trail_tiled = xnp.stack([parent_trail] * action_size, axis=0)
    flat_trail = xnp.reshape(parent_trail_tiled, (flat_size, non_backtracking_steps))
    flat_parent_depths = jnp.broadcast_to(parent_depths, (action_size, batch_size)).reshape(
        (flat_size,)
    )

    def _loop(i, carry):
        trail_state = xnp.take(flat_trail, trail_indices[i], axis=1)
        matches = _batched_state_equal(candidate_states, trail_state)
        valid_trail = trail_indices[i] < flat_parent_depths
        matches = jnp.logical_and(matches, valid_trail)
        return jnp.logical_or(carry, matches)

    blocked = jax.lax.fori_loop(0, non_backtracking_steps, _loop, blocked)
    return jnp.logical_and(valid_mask, jnp.logical_not(blocked))


@base_dataclass
class IDLoopState:
    """
    Loop state for the inner DFS loop of IDA*.
    """

    search_result: "IDSearchBase"
    solve_config: Puzzle.SolveConfig
    params: Any
    frontier: IDFrontier


@base_dataclass(static_fields=("capacity", "action_size", "ItemCls"))
class IDSearchBase:
    """
    Data structure for Iterative Deepening Search.
    Maintains an explicit stack for Batched DFS.

    Renamed from IDSearchResult for clarity:
    - "Base" indicates this is the core search state container
    - Methods handle stack operations and trace management
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
    start_state: Xtructurable  # Single state (start)
    solution_cost: chex.Array  # Scalar - cost of solution
    generated_count: chex.Array  # Scalar - count of generated nodes
    solution_actions_arr: chex.Array  # [max_path_len]
    trace_parent: chex.Array  # [capacity]
    trace_action: chex.Array  # [capacity]
    trace_root: chex.Array  # [capacity]
    trace_size: chex.Array  # scalar
    frontier_action_history: chex.Array  # [batch_size, max_path_len]

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
    def build(
        statecls: Puzzle.State,
        capacity: int,
        action_size: int,
        non_backtracking_steps: int = 0,
        max_path_len: int = 256,
        seed: int = 42,
    ) -> "IDSearchBase":
        """
        Initialize the IDSearchBase with empty stack.
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
            parent_index: FieldDescriptor.scalar(dtype=jnp.int32)
            root_index: FieldDescriptor.scalar(dtype=jnp.int32)
            trace_index: FieldDescriptor.scalar(dtype=jnp.int32)
            trail: FieldDescriptor.tensor(dtype=statecls, shape=trail_shape)
            action_history: FieldDescriptor.tensor(dtype=ACTION_DTYPE, shape=(max_path_len,))

        bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        next_bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)

        stack = Stack.build(capacity, IDStackItem)

        solution_state = statecls.default((1,))  # Placeholder batch-1
        start_state = statecls.default((1,))  # Placeholder batch-1
        solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solution_actions = jnp.full((max_path_len,), ACTION_PAD, dtype=ACTION_DTYPE)
        generated_count = jnp.array(0, dtype=jnp.int32)
        trace_parent = jnp.full((capacity,), -1, dtype=jnp.int32)
        trace_action = jnp.full((capacity,), ACTION_PAD, dtype=ACTION_DTYPE)
        trace_root = jnp.full((capacity,), -1, dtype=jnp.int32)
        trace_size = jnp.array(0, dtype=jnp.int32)
        frontier_action_history = jnp.full((1, max_path_len), ACTION_PAD, dtype=ACTION_DTYPE)

        return IDSearchBase(
            capacity=capacity,
            action_size=action_size,
            ItemCls=IDStackItem,
            bound=bound,
            next_bound=next_bound,
            solved=solved,
            solved_idx=solved_idx,
            stack=stack,
            solution_state=solution_state,
            start_state=start_state,
            solution_cost=solution_cost,
            solution_actions_arr=solution_actions,
            trace_parent=trace_parent,
            trace_action=trace_action,
            trace_root=trace_root,
            trace_size=trace_size,
            frontier_action_history=frontier_action_history,
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
            return [self.solution_state[0]]

        # Reconstruct path
        path = [self.start_state[0]]
        actions = self.solution_actions()

        curr = xnp.expand_dims(self.start_state[0], 0)  # (1, ...)

        # Prepare step function
        @jax.jit
        def step_fn(acc_state, action):
            act_batch = jnp.array([action], dtype=jnp.int32)
            valid_batch = jnp.array([True], dtype=jnp.bool_)
            next_states, _ = puzzle.batched_get_actions(
                solve_config, acc_state, act_batch, valid_batch
            )
            return next_states

        # Iterate actions on host
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
        "IDSearchBase",
        Puzzle.State,
        chex.Array,
        chex.Array,
        Puzzle.State,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        """
        Pop the top `batch_size` items from the stack.
        Returns (updated_self, states, costs, depths, trails, action_histories, valid_mask, indices)
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
        action_histories = popped_items.action_history
        trace_indices = popped_items.trace_index
        root_indices = popped_items.root_index

        new_self = self.replace(stack=new_stack)

        return (
            new_self,
            states,
            costs,
            depths,
            trails,
            action_histories,
            valid_mask,
            trace_indices,
            root_indices,
        )

    def push_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        actions: chex.Array,
        parent_indices: chex.Array,
        root_indices: chex.Array,
        trails: Xtructurable,
        action_histories: chex.Array,
        valid_mask: chex.Array,
    ) -> "IDSearchBase":
        """
        Push a batch of items onto the stack.
        Only pushes items where valid_mask is True.
        """
        n_push = jnp.sum(valid_mask.astype(jnp.int32))

        def _empty(sr: "IDSearchBase") -> "IDSearchBase":
            return sr

        def _dense(sr: "IDSearchBase") -> "IDSearchBase":
            return sr.push_packed_batch(
                states=states,
                costs=costs,
                depths=depths,
                actions=actions,
                parent_indices=parent_indices,
                root_indices=root_indices,
                trails=trails,
                action_histories=action_histories,
                n_push=n_push,
            )

        def _sparse(sr: "IDSearchBase") -> "IDSearchBase":
            perm = stable_partition_three(valid_mask, jnp.zeros_like(valid_mask, dtype=jnp.bool_))

            states_sorted = xnp.take(states, perm, axis=0)
            costs_sorted = costs[perm]
            depths_sorted = depths[perm]
            actions_sorted = actions[perm]
            parent_indices_sorted = parent_indices[perm]
            root_indices_sorted = root_indices[perm]

            current_ptr = sr.stack.size.astype(jnp.int32)
            stack_capacity = jnp.array(sr.stack.max_size, dtype=jnp.int32)
            generated_capacity = jnp.array(sr.capacity, dtype=jnp.int32)
            remaining_stack = jnp.maximum(stack_capacity - current_ptr, 0)
            remaining_generated = jnp.maximum(generated_capacity - sr.generated_count, 0)
            remaining_trace = jnp.maximum(generated_capacity - sr.trace_size, 0)
            safe_n_push = jnp.minimum(
                n_push,
                jnp.minimum(remaining_stack, jnp.minimum(remaining_generated, remaining_trace)),
            )

            trace_base = sr.trace_size
            batch_len = valid_mask.shape[0]
            trace_ids_sorted = trace_base + jnp.arange(batch_len, dtype=jnp.int32)
            valid_sorted = valid_mask[perm]
            push_mask_sorted = jnp.logical_and(
                valid_sorted, jnp.arange(batch_len, dtype=jnp.int32) < safe_n_push
            )
            trace_ids_sorted = jnp.where(push_mask_sorted, trace_ids_sorted, -1)

            items_sorted = sr.ItemCls(
                state=states_sorted,
                cost=costs_sorted,
                depth=depths_sorted,
                action=actions_sorted,
                parent_index=parent_indices_sorted,
                root_index=root_indices_sorted,
                trace_index=trace_ids_sorted,
                trail=xnp.take(trails, perm, axis=0),
                action_history=action_histories[perm],
            )

            trace_parent_updates = parent_indices_sorted
            trace_action_updates = actions_sorted
            trace_root_updates = root_indices_sorted
            trace_mask = trace_ids_sorted >= 0

            stack_indices = current_ptr + jnp.arange(batch_len, dtype=jnp.int32)
            new_val_store = xnp.update_on_condition(
                sr.stack.val_store, stack_indices, push_mask_sorted, items_sorted
            )

            trace_ids_dense = jnp.where(trace_mask, trace_ids_sorted, 0)
            new_trace_parent = xnp.update_on_condition(
                sr.trace_parent, trace_ids_dense, trace_mask, trace_parent_updates
            )
            new_trace_action = xnp.update_on_condition(
                sr.trace_action, trace_ids_dense, trace_mask, trace_action_updates
            )
            new_trace_root = xnp.update_on_condition(
                sr.trace_root, trace_ids_dense, trace_mask, trace_root_updates
            )

            new_ptr = (current_ptr + safe_n_push).astype(sr.stack.size.dtype)
            new_generated_count = sr.generated_count + safe_n_push
            new_trace_size = sr.trace_size + safe_n_push
            new_stack = sr.stack.replace(val_store=new_val_store, size=new_ptr)

            return sr.replace(
                stack=new_stack,
                generated_count=new_generated_count,
                trace_parent=new_trace_parent,
                trace_action=new_trace_action,
                trace_root=new_trace_root,
                trace_size=new_trace_size,
            )

        return jax.lax.cond(
            n_push > 0,
            lambda sr: jax.lax.cond(jnp.all(valid_mask), _dense, _sparse, sr),
            _empty,
            self,
        )

    def push_packed_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        actions: chex.Array,
        parent_indices: chex.Array,
        root_indices: chex.Array,
        trails: Xtructurable,
        action_histories: chex.Array,
        n_push: chex.Array,
    ) -> "IDSearchBase":
        """
        Push a pre-packed batch of items onto the stack.
        Assumes the items are already sorted/packed such that the valid items
        to be pushed are at the beginning of the arrays.
        """
        current_ptr = self.stack.size.astype(jnp.int32)
        stack_capacity = jnp.array(self.stack.max_size, dtype=jnp.int32)
        generated_capacity = jnp.array(self.capacity, dtype=jnp.int32)
        remaining_stack = jnp.maximum(stack_capacity - current_ptr, 0)
        remaining_generated = jnp.maximum(generated_capacity - self.generated_count, 0)
        remaining_trace = jnp.maximum(generated_capacity - self.trace_size, 0)
        safe_n_push = jnp.minimum(
            n_push.astype(jnp.int32),
            jnp.minimum(remaining_stack, jnp.minimum(remaining_generated, remaining_trace)),
        )

        trace_base = self.trace_size
        batch_len = parent_indices.shape[0]
        trace_ids = trace_base + jnp.arange(batch_len, dtype=jnp.int32)
        trace_ids = jnp.where(
            jnp.arange(batch_len, dtype=jnp.int32) < safe_n_push,
            trace_ids,
            -1,
        )

        items = self.ItemCls(
            state=states,
            cost=costs,
            depth=depths,
            action=actions,
            parent_index=parent_indices,
            root_index=root_indices,
            trace_index=trace_ids,
            trail=trails,
            action_history=action_histories,
        )

        stack_indices = current_ptr + jnp.arange(batch_len, dtype=jnp.int32)
        push_mask = jnp.arange(batch_len, dtype=jnp.int32) < safe_n_push
        new_val_store = xnp.update_on_condition(
            self.stack.val_store, stack_indices, push_mask, items
        )

        trace_mask = trace_ids >= 0
        trace_ids_dense = jnp.where(trace_mask, trace_ids, 0)
        new_trace_parent = xnp.update_on_condition(
            self.trace_parent, trace_ids_dense, trace_mask, parent_indices
        )
        new_trace_action = xnp.update_on_condition(
            self.trace_action, trace_ids_dense, trace_mask, actions
        )
        new_trace_root = xnp.update_on_condition(
            self.trace_root, trace_ids_dense, trace_mask, root_indices
        )

        new_ptr = (current_ptr + safe_n_push).astype(self.stack.size.dtype)
        new_generated_count = self.generated_count + safe_n_push
        new_trace_size = self.trace_size + safe_n_push

        new_stack = self.stack.replace(val_store=new_val_store, size=new_ptr)

        return self.replace(
            stack=new_stack,
            generated_count=new_generated_count,
            trace_parent=new_trace_parent,
            trace_action=new_trace_action,
            trace_root=new_trace_root,
            trace_size=new_trace_size,
        )

    def push_frontier_to_stack(
        self,
        frontier: "IDFrontier",
        bound: chex.Array,
        frontier_actions: chex.Array,
    ) -> "IDSearchBase":
        """
        Push allowed frontier nodes to stack and update next_bound.

        Common logic extracted from both id_astar.py and id_qstar.py.
        This method filters frontier nodes by bound, updates next_bound
        with the minimum f-score of pruned nodes, and pushes valid nodes.

        Args:
            frontier: IDFrontier containing states to push
            bound: Current cost bound (scalar)
            frontier_actions: Placeholder actions for frontier nodes [batch_size]

        Returns:
            Updated IDSearchBase with nodes pushed and next_bound updated
        """
        fs = frontier.f_scores
        batch_size = frontier.valid_mask.shape[0]

        # Filter nodes within bound
        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)

        # Determine next_bound from pruned nodes
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(self.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = self.replace(next_bound=new_next_bound)

        # Prepare indices
        parent_indices = jnp.full((batch_size,), -1, dtype=jnp.int32)
        root_indices = jnp.where(frontier.valid_mask, jnp.arange(batch_size), -1)

        return sr.push_batch(
            frontier.states,
            frontier.costs,
            frontier.depths,
            frontier_actions,
            parent_indices,
            root_indices,
            frontier.trail,
            frontier.action_history,
            keep_mask,
        )

    def expand_and_push(
        self,
        node_batch: Xtructurable,
        fs: chex.Array,
        valid: chex.Array,
        update_next_bound: bool = True,
    ) -> "IDSearchBase":
        """
        Expand valid nodes and push to stack, sorted by f-value descending (LIFO).

        Common logic extracted from _expand_step() in both id_astar.py and id_qstar.py.
        This method:
        1. Filters nodes within the current bound
        2. Compacts and sorts by f-value (descending for LIFO)
        3. Optionally updates next_bound from pruned nodes
        4. Pushes to stack

        Args:
            node_batch: IDNodeBatch containing states, costs, depths, actions, etc.
            fs: f-scores for each node [flat_size]
            valid: validity mask for each node [flat_size]
            update_next_bound: If True, update next_bound from pruned nodes (ID-A* needs this,
                               ID-Q* updates next_bound earlier in inner_body)

        Returns:
            Updated IDSearchBase with nodes pushed to stack
        """
        active_bound = self.bound
        keep_mask = jnp.logical_and(valid, fs <= active_bound + 1e-6)

        # Optionally update next_bound from pruned nodes
        sr = self
        if update_next_bound:
            prune_mask = jnp.logical_and(valid, fs > active_bound + 1e-6)
            pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
            min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
            new_next_bound = jnp.minimum(self.next_bound, min_pruned_f).astype(KEY_DTYPE)
            sr = self.replace(next_bound=new_next_bound)

        def _push_nonempty(sr_curr: "IDSearchBase") -> "IDSearchBase":
            # Compact valid nodes
            packed_batch, packed_valid, _, packed_idx = compact_by_valid(node_batch, keep_mask)
            packed_fs = jax.lax.cond(
                jnp.all(keep_mask),
                lambda _: fs,
                lambda _: jnp.where(packed_valid, fs[packed_idx], jnp.inf),
                operand=None,
            )

            # Sort by f-value descending (worst -> best) for LIFO stack order
            f_key = jnp.where(packed_valid, -packed_fs, jnp.inf)
            perm = jnp.argsort(f_key)
            ordered = xnp.take(packed_batch, perm, axis=0)
            n_push = jnp.sum(packed_valid)

            return sr_curr.push_packed_batch(
                ordered.state,
                ordered.cost,
                ordered.depth,
                ordered.action,
                ordered.parent_index,
                ordered.root_index,
                ordered.trail,
                ordered.action_history,
                n_push,
            )

        return jax.lax.cond(
            jnp.any(keep_mask),
            _push_nonempty,
            lambda sr_curr: sr_curr,
            sr,
        )

    def mark_solved(
        self,
        any_solved: chex.Array,
        solved_state: Xtructurable,
        solved_cost: chex.Array,
        solved_actions: chex.Array,
        solved_trace_idx: chex.Array,
    ) -> "IDSearchBase":
        """
        Mark search as solved and store solution info.
        """
        new_solution_state = jax.lax.cond(
            any_solved, lambda _: solved_state, lambda _: self.solution_state, None
        )
        new_solution_cost = jax.lax.cond(
            any_solved,
            lambda _: solved_cost.astype(KEY_DTYPE),
            lambda _: self.solution_cost,
            None,
        )
        new_solution_actions = jax.lax.cond(
            any_solved,
            lambda _: solved_actions,
            lambda _: self.solution_actions_arr,
            None,
        )

        return self.replace(
            solved=jnp.logical_or(self.solved, any_solved),
            solved_idx=jnp.where(any_solved, solved_trace_idx, -1),
            solution_state=new_solution_state,
            solution_cost=new_solution_cost,
            solution_actions_arr=new_solution_actions,
        )

    def initialize_from_frontier(
        self,
        frontier: IDFrontier,
        cost_weight: float,
        eval_fn: callable,
        params: Any,
        frontier_actions: chex.Array,
    ) -> "IDSearchBase":
        """
        Initialize bound and stack from a pre-computed frontier.
        """
        # Calculate F values for the frontier
        frontier_h = eval_fn(params, frontier.states, frontier.valid_mask).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)
        frontier = frontier.replace(f_scores=frontier_f)

        # Initial Bound = min(f) of valid frontier nodes
        valid_fs = jnp.where(frontier.valid_mask, frontier.f_scores, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        # Update search result with bound and solution from frontier
        sr = self.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solution_actions_arr=frontier.solution_actions_arr,
            solved_idx=jnp.where(frontier.solved, -1, -1),
            frontier_action_history=frontier.action_history,
        )

        # Push frontier nodes within bound to stack
        return (
            sr.push_frontier_to_stack(frontier, start_bound, frontier_actions),
            frontier,
        )

    @staticmethod
    def detect_solution(
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        states: Puzzle.State,
        costs: chex.Array,
        action_history: chex.Array,
        valid_mask: chex.Array,
    ):
        is_solved_mask = puzzle.batched_is_solved(solve_config, states)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        first_idx = jnp.argmax(is_solved_mask)
        solved_state = xnp.take(states, first_idx[jnp.newaxis], axis=0)
        solved_cost = costs[first_idx]
        solved_actions = action_history[first_idx]

        return any_solved, solved_state, solved_cost, solved_actions, first_idx

    def prepare_for_expansion(
        self,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        batch_size: int,
    ):
        """
        Pop top batch, check for solution, and get neighbours.
        """
        (
            sr,
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_action_histories,
            valid_mask,
            parent_trace_indices,
            parent_root_indices,
        ) = self.get_top_batch(batch_size)

        any_solved, solved_state, solved_cost, solved_actions, first_idx = self.detect_solution(
            puzzle,
            solve_config,
            parents,
            parent_costs,
            parent_action_histories,
            valid_mask,
        )
        solved_trace_idx = parent_trace_indices[first_idx]

        sr_solved = sr.mark_solved(
            any_solved, solved_state, solved_cost, solved_actions, solved_trace_idx
        )

        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)

        return (
            sr,
            sr_solved,
            any_solved,
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_action_histories,
            valid_mask,
            parent_trace_indices,
            parent_root_indices,
            neighbours,
            step_costs,
        )

    def apply_standard_deduplication(
        self,
        flat_neighbours: Xtructurable,
        flat_g: chex.Array,
        flat_valid: chex.Array,
        parents: Xtructurable,
        parent_trails: Xtructurable,
        parent_depths: chex.Array,
        non_backtracking_steps: int,
        action_size: int,
        flat_size: int,
        trail_indices: jnp.ndarray,
        batch_size: int,
    ) -> tuple["IDSearchBase", chex.Array]:
        """
        Apply in-batch deduplication and non-backtracking.
        """
        unique_mask = xnp.unique_mask(
            flat_neighbours,
            key=flat_g,
            filled=flat_valid,
        )
        flat_valid = jnp.logical_and(flat_valid, unique_mask)

        flat_valid = apply_non_backtracking(
            flat_neighbours,
            parents,
            parent_trails,
            parent_depths,
            flat_valid,
            non_backtracking_steps,
            action_size,
            flat_size,
            trail_indices,
            batch_size,
        )

        return self, flat_valid


def finalize_builder(
    puzzle: Puzzle,
    init_loop: callable,
    cond: callable,
    body: callable,
    name: str = "ID-Search",
    show_compile_time: bool = False,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Finalize the search builder by JIT-compiling and optionally warming up.
    """

    def search_fn(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        loop_state = init_loop(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(cond, body, loop_state)
        return loop_state.search_result

    return jit_with_warmup(
        search_fn,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
        init_message=f"initializing jit for {name}",
        elapsed_message=f"{name} JIT compile time: {{elapsed:.2f}}s",
    )


def expand_and_push_flat_batch(
    search_result: IDSearchBase,
    node_batch_cls: type,
    states: Puzzle.State,
    gs: chex.Array,
    depths: chex.Array,
    actions: chex.Array,
    trails: Puzzle.State,
    action_histories: chex.Array,
    valid: chex.Array,
    fs: chex.Array,
    parent_indices: chex.Array,
    root_indices: chex.Array,
    *,
    update_next_bound: bool,
) -> IDSearchBase:
    """Build a flat node batch and push it into the search stack."""
    flat_batch = node_batch_cls(
        state=states,
        cost=gs,
        depth=depths,
        action=actions,
        trail=trails,
        action_history=action_histories,
        parent_index=parent_indices,
        root_index=root_indices,
    )
    return search_result.expand_and_push(flat_batch, fs, valid, update_next_bound=update_next_bound)


def prepare_flat_expansion_inputs(
    search_result: IDSearchBase,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
    *,
    batch_size: int,
    action_ids: chex.Array,
    action_size: int,
    flat_size: int,
    non_backtracking_steps: int,
    max_path_len: int,
    empty_trail_flat: Puzzle.State,
):
    """Prepare common flat child tensors for ID-A*/ID-Q* inner loops."""
    (
        sr,
        sr_solved,
        any_solved,
        parents,
        parent_costs,
        parent_depths,
        parent_trails,
        parent_action_histories,
        valid_mask,
        parent_trace_indices,
        parent_root_indices,
        neighbours,
        step_costs,
    ) = search_result.prepare_for_expansion(puzzle, solve_config, batch_size)

    (
        flat_neighbours,
        flat_g,
        flat_depth,
        flat_trail,
        flat_action_history,
        flat_actions,
        flat_valid,
    ) = build_flat_children(
        neighbours,
        step_costs,
        parent_costs,
        parent_depths,
        parents,
        parent_trails,
        parent_action_histories,
        action_ids,
        action_size,
        batch_size,
        flat_size,
        non_backtracking_steps,
        max_path_len,
        empty_trail_flat,
        valid_mask,
    )

    flat_valid = jnp.logical_and(flat_valid, flat_depth <= max_path_len)

    flat_parent_indices = jnp.tile(parent_trace_indices, action_size)
    flat_parent_indices = jnp.where(flat_valid, flat_parent_indices, -1)
    flat_root_indices = jnp.tile(parent_root_indices, action_size)
    flat_root_indices = jnp.where(flat_valid, flat_root_indices, -1)

    return (
        sr,
        sr_solved,
        any_solved,
        parents,
        parent_costs,
        parent_depths,
        parent_trails,
        parent_action_histories,
        valid_mask,
        flat_neighbours,
        flat_g,
        flat_depth,
        flat_trail,
        flat_action_history,
        flat_actions,
        flat_valid,
        flat_parent_indices,
        flat_root_indices,
    )


def apply_dedup_and_mask_actions(
    search_result: IDSearchBase,
    flat_neighbours: Xtructurable,
    flat_g: chex.Array,
    flat_valid: chex.Array,
    parents: Xtructurable,
    parent_trails: Xtructurable,
    parent_depths: chex.Array,
    non_backtracking_steps: int,
    action_size: int,
    flat_size: int,
    trail_indices: jnp.ndarray,
    batch_size: int,
    flat_action_history: chex.Array,
) -> tuple[IDSearchBase, chex.Array, chex.Array]:
    """Apply standard deduplication and mask invalid action histories."""
    search_result, flat_valid = search_result.apply_standard_deduplication(
        flat_neighbours,
        flat_g,
        flat_valid,
        parents,
        parent_trails,
        parent_depths,
        non_backtracking_steps,
        action_size,
        flat_size,
        trail_indices,
        batch_size,
    )
    flat_action_history = jnp.where(
        flat_valid[:, None],
        flat_action_history,
        jnp.full_like(flat_action_history, ACTION_PAD),
    )
    return search_result, flat_valid, flat_action_history


def build_inner_cond():
    """
    Build inner loop condition function for ID search algorithms.

    Returns a function that checks if the DFS stack has items and search is not solved.
    """

    def inner_cond(loop_state: IDLoopState) -> jnp.ndarray:
        sr = loop_state.search_result
        has_items = sr.stack_ptr > 0
        not_solved = ~sr.solved
        return jnp.logical_and(has_items, not_solved)

    return inner_cond


def build_outer_loop(
    inner_cond,
    inner_body,
    statecls,
    frontier_actions: jnp.ndarray,
):
    def outer_cond(loop_state: IDLoopState) -> jnp.ndarray:
        sr = loop_state.search_result
        return jnp.logical_and(~sr.solved, jnp.isfinite(sr.bound))

    def outer_body(loop_state: IDLoopState) -> IDLoopState:
        loop_state = jax.lax.while_loop(inner_cond, inner_body, loop_state)

        sr = loop_state.search_result
        new_bound = sr.next_bound

        reset_sr = sr.replace(
            bound=new_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            stack=sr.stack.replace(size=jnp.array(0, dtype=jnp.uint32)),
            trace_size=jnp.array(0, dtype=jnp.int32),
        )

        reset_sr = reset_sr.push_frontier_to_stack(loop_state.frontier, new_bound, frontier_actions)

        return loop_state.replace(search_result=reset_sr)

    return outer_cond, outer_body
