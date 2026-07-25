"""
Search Base for Iterative Deepening Algorithms (IDA* / ID-Q*)
"""

import time
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass
from xtructure.core.packing import pack_rows
from xtructure.stack import Stack

from helpers.jax_compile import compile_with_example, warmup_with_example
from JAxtar.annotate import (
    ACTION_DTYPE,
    KEY_DTYPE,
    TRACE_INDEX_DTYPE,
    TRACE_INVALID,
)
from JAxtar.expansion_trace import ExpansionTrace
from JAxtar.id_stars.id_frontier import ACTION_PAD, IDFrontier, compact_by_valid
from JAxtar.solution_trace import (
    SolutionTrace,
    action_pad_int,
)
from JAxtar.utils.array_ops import batched_state_equal, stable_partition_three


def _bounded_scatter_leaf(stack_arr, update_arr, start, n_write):
    """Scatter ``update_arr[:n_write]`` into ``stack_arr`` at ``[start, start + n_write)``.

    Rows at/after ``n_write`` map to an out-of-bounds index and are dropped. This
    replaces ``dynamic_update_slice``, which clamps a near-capacity start index and
    would slide the whole write block down, silently overwriting already-committed
    live stack entries below ``start``.
    """
    rows = jnp.arange(update_arr.shape[0], dtype=jnp.int32)
    target = jnp.where(rows < n_write, start + rows, stack_arr.shape[0])
    return stack_arr.at[target].set(update_arr, mode="drop")


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
    blocked = batched_state_equal(candidate_states, flat_parent_states)
    blocked = jnp.logical_and(blocked, valid_mask)

    parent_trail_tiled = xnp.stack([parent_trail] * action_size, axis=0)
    flat_trail = xnp.reshape(parent_trail_tiled, (flat_size, non_backtracking_steps))
    flat_parent_depths = jnp.broadcast_to(parent_depths, (action_size, batch_size)).reshape(
        (flat_size,)
    )

    def _loop(i, carry):
        trail_state = xnp.take(flat_trail, trail_indices[i], axis=1)
        matches = batched_state_equal(candidate_states, trail_state)
        valid_trail = trail_indices[i] < flat_parent_depths
        matches = jnp.logical_and(matches, valid_trail)
        return jnp.logical_or(carry, matches)

    blocked = jax.lax.fori_loop(0, non_backtracking_steps, _loop, blocked)
    return jnp.logical_and(valid_mask, jnp.logical_not(blocked))


def build_frontier_cond(batch_size: int, max_steps: int = 100):
    def cond_bounded(val: tuple["IDFrontier", jnp.int32]):
        frontier, i = val
        num_valid = jnp.sum(frontier.valid_mask)
        has_capacity = num_valid < batch_size
        has_nodes = num_valid > 0
        within_limit = i < max_steps
        return jnp.logical_and(
            ~frontier.solved,
            jnp.logical_and(within_limit, jnp.logical_and(has_capacity, has_nodes)),
        )

    return cond_bounded


def merge_frontier_solution(
    frontier: "IDFrontier",
    any_solved: chex.Array,
    found_sol_state: Xtructurable,
    found_sol_cost: chex.Array,
    found_sol_actions: chex.Array,
):
    # Pure value selection: where instead of lax.cond (host predicate sync).
    return (
        jnp.logical_or(frontier.solved, any_solved),
        xnp.where(any_solved, found_sol_state, frontier.solution_state),
        jnp.where(any_solved, found_sol_cost, frontier.solution_cost),
        jnp.where(any_solved, found_sol_actions, frontier.solution_actions_arr),
    )


@base_dataclass
class IDLoopState:
    """
    Loop state for the inner DFS loop of IDA*.
    """

    search_result: "IDSearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    frontier: IDFrontier


@base_dataclass(
    static_fields=("capacity", "action_size", "ItemCls", "max_path_len", "trace_prefix")
)
class IDSearchResult:
    """
    Data structure for Iterative Deepening Search.
    Maintains an explicit stack for Batched DFS. Each stack item stores only a
    ``trace_index`` into a shared parent-pointer arena (``trace_parent``/``trace_action``,
    the same shape Beam uses); the action sequence is walked out of that arena once, at
    the end of the search. Carrying the full ``uint8[max_path_len]`` path on every stack
    slot instead cost 256 of 299 packed bytes per node.
    """

    capacity: int
    action_size: int
    ItemCls: type
    max_path_len: int
    trace_prefix: int  # arena rows reserved for the immutable frontier chains

    # Global search state
    bound: chex.Array  # Current cost bound (scalar)
    next_bound: chex.Array  # Next cost bound (scalar)
    solved: chex.Array  # Boolean scalar
    solved_idx: chex.Array  # Index in the stack of the solution (if found)

    # Stack storage (Fixed size = capacity)
    # Allows LIFO access for DFS
    stack: Stack

    # Parent-pointer arena. Rows [0, trace_prefix) are the frontier chains, written once
    # and never reset; [trace_prefix, len) is refilled from scratch at each threshold.
    trace_parent: chex.Array  # [trace_capacity] uint32, TRACE_INVALID at a chain root
    trace_action: chex.Array  # [trace_capacity] ACTION_DTYPE
    trace_ptr: chex.Array  # Scalar - next free dynamic arena row
    trace_overflow: chex.Array  # Scalar - nodes dropped because the arena was full

    solution_state: Xtructurable  # Single state
    start_state: Xtructurable  # Single state (start)
    solution_cost: chex.Array  # Scalar - cost of solution
    generated_count: chex.Array  # Scalar - count of generated nodes
    solution_actions_arr: chex.Array  # [max_path_len]
    solved_trace_index: chex.Array  # Scalar - arena row of the solution node

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
    def build(
        statecls: Puzzle.State,
        capacity: int,
        action_size: int,
        non_backtracking_steps: int = 0,
        max_path_len: int = 256,
        frontier_size: int = 1,
        seed: int = 42,
    ) -> "IDSearchResult":
        """
        Initialize the IDSearchResult with an empty stack and an empty trace arena.
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
            trail: FieldDescriptor.tensor(dtype=statecls, shape=trail_shape)
            trace_index: FieldDescriptor.scalar(dtype=TRACE_INDEX_DTYPE)

        bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        next_bound = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)

        stack = Stack.build(capacity, IDStackItem)

        # Each frontier node owns a block of `max_path_len` rows so its chain can be
        # spliced in once and re-used by every threshold. At 5 B/row this prefix is
        # negligible next to the dynamic region.
        trace_prefix = int(frontier_size) * int(max_path_len)
        trace_capacity = trace_prefix + int(capacity)
        trace_parent = jnp.full((trace_capacity,), TRACE_INVALID, dtype=TRACE_INDEX_DTYPE)
        trace_action = jnp.full((trace_capacity,), ACTION_PAD, dtype=ACTION_DTYPE)

        solution_state = statecls.default((1,))  # Placeholder batch-1
        start_state = statecls.default((1,))  # Placeholder batch-1
        solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
        solution_actions = jnp.full((max_path_len,), ACTION_PAD, dtype=ACTION_DTYPE)
        generated_count = jnp.array(0, dtype=jnp.int32)

        return IDSearchResult(
            capacity=capacity,
            action_size=action_size,
            ItemCls=IDStackItem,
            max_path_len=int(max_path_len),
            trace_prefix=trace_prefix,
            bound=bound,
            next_bound=next_bound,
            solved=solved,
            solved_idx=solved_idx,
            stack=stack,
            trace_parent=trace_parent,
            trace_action=trace_action,
            trace_ptr=jnp.array(trace_prefix, dtype=jnp.int32),
            trace_overflow=jnp.array(0, dtype=jnp.int32),
            solution_state=solution_state,
            start_state=start_state,
            solution_cost=solution_cost,
            solution_actions_arr=solution_actions,
            generated_count=generated_count,
            solved_trace_index=TRACE_INVALID,
        )

    @property
    def stack_ptr(self):
        return self.stack.size

    @property
    def trace_capacity(self) -> int:
        return self.trace_parent.shape[0]

    def splice_frontier_trace(self, frontier: IDFrontier) -> "IDSearchResult":
        """Write each frontier node's action history into its reserved arena block.

        Node ``i`` occupies rows ``[i * max_path_len, i * max_path_len + depth_i)``; the
        node's own trace index is the last row of its block. Called once per search --
        the block is immutable, so every threshold re-seeds the same indices.
        """
        mpl = self.max_path_len
        frontier_size = frontier.valid_mask.shape[0]
        block = jnp.arange(frontier_size, dtype=jnp.int32)[:, None] * mpl
        step = jnp.arange(mpl, dtype=jnp.int32)[None, :]
        rows = (block + step).reshape(-1)

        # Row k of a block points at row k-1; row 0 is a chain root.
        in_chain = jnp.logical_and(step > 0, step < frontier.depths[:, None])
        parents = jnp.where(in_chain, (block + step - 1).astype(TRACE_INDEX_DTYPE), TRACE_INVALID)

        trace_parent = self.trace_parent.at[rows].set(parents.reshape(-1), mode="drop")
        trace_action = self.trace_action.at[rows].set(
            frontier.action_history.reshape(-1), mode="drop"
        )
        return self.replace(trace_parent=trace_parent, trace_action=trace_action)

    def frontier_trace_indices(self, frontier: IDFrontier) -> chex.Array:
        """Arena row of each frontier node: the last row of its block, or INVALID at depth 0."""
        frontier_size = frontier.valid_mask.shape[0]
        block = jnp.arange(frontier_size, dtype=jnp.int32) * self.max_path_len
        last = (block + frontier.depths.astype(jnp.int32) - 1).astype(TRACE_INDEX_DTYPE)
        return jnp.where(frontier.depths > 0, last, TRACE_INVALID)

    def _append_trace(
        self, parent_trace: chex.Array, actions: chex.Array, n_push: chex.Array
    ) -> tuple["IDSearchResult", chex.Array, chex.Array]:
        """Reserve arena rows for a packed child batch; return (sr, indices, n_written).

        ``n_push`` is clamped to the free arena rows, so a full arena drops the tail of the
        batch exactly like a full stack does -- and the dropped count is recorded in
        ``trace_overflow`` rather than vanishing.
        """
        rows = jnp.arange(parent_trace.shape[0], dtype=jnp.int32)
        free = jnp.maximum(0, self.trace_capacity - self.trace_ptr)
        writable = jnp.minimum(n_push.astype(jnp.int32), free)

        targets = jnp.where(rows < writable, self.trace_ptr + rows, self.trace_capacity)
        trace_parent = self.trace_parent.at[targets].set(
            parent_trace.astype(TRACE_INDEX_DTYPE), mode="drop"
        )
        trace_action = self.trace_action.at[targets].set(actions.astype(ACTION_DTYPE), mode="drop")
        sr = self.replace(
            trace_parent=trace_parent,
            trace_action=trace_action,
            trace_ptr=self.trace_ptr + writable,
            trace_overflow=self.trace_overflow + (n_push.astype(jnp.int32) - writable),
        )
        return sr, targets.astype(TRACE_INDEX_DTYPE), writable

    def _get_solved_path(self, puzzle: Puzzle = None, solve_config: Puzzle.SolveConfig = None):
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
        actions = self._solution_actions()

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

    def _solution_actions(self):
        """
        Return the list of actions to reach the solution.
        """
        valid_mask = self.solution_actions_arr != ACTION_PAD
        return self.solution_actions_arr[valid_mask]

    def to_expansion_trace(self) -> ExpansionTrace | None:
        """This family records no per-node expansion order; adapters skip expansion analysis."""
        return None

    def to_solution_trace(
        self,
        *,
        puzzle: Puzzle | None = None,
    ) -> SolutionTrace:
        """Return the host-side solution trace for CLI/evaluation adapters."""
        if not bool(jax.device_get(self.solved)):
            return SolutionTrace.unsolved()

        return SolutionTrace.from_raw(
            solved=True,
            raw_actions=jax.device_get(self.solution_actions_arr),
            action_pad=action_pad_int(ACTION_DTYPE),
        )

    def get_generated_size(self):
        return self.generated_count

    @property
    def generated_size(self):
        return self.generated_count

    def get_cost(self, idx: int):
        return self.solution_cost

    def get_top_batch(
        self, batch_size: int
    ) -> tuple[
        "IDSearchResult",
        Puzzle.State,
        chex.Array,
        chex.Array,
        Puzzle.State,
        chex.Array,
        chex.Array,
    ]:
        """
        Pop the top `batch_size` items from the stack.
        Returns (updated_self, states, costs, depths, trails, trace_indices, valid_mask)
        """
        # Ensure consistent index type for JAX
        current_size = self.stack.size.astype(jnp.int32)

        new_stack, popped_items = self.stack.pop(batch_size)

        # Stack.pop packs survivors at the TAIL of the popped array on a partial pop:
        # it gathers indices = size - arange(batch, 0, -1), so when size < batch_size the
        # low (negative) indices wrap to unwritten default slots at the array head and the
        # real items land in the last `size` positions. The valid mask must therefore mark
        # the TAIL, not the head; a head-aligned mask selects default garbage and drops the
        # real DFS nodes. Reduces to all-True when size >= batch_size.
        valid_count = jnp.minimum(current_size, batch_size)
        indices_range = jnp.arange(batch_size, dtype=jnp.int32)
        valid_mask = indices_range >= (batch_size - valid_count)

        # Recast back to stack native size type if needed
        actual_new_size = jnp.maximum(0, current_size - batch_size).astype(self.stack.size.dtype)
        new_stack = new_stack.replace(size=actual_new_size)

        return (
            self.replace(stack=new_stack),
            popped_items.state,
            popped_items.cost,
            popped_items.depth,
            popped_items.trail,
            popped_items.trace_index,
            valid_mask,
        )

    def push_batch(
        self,
        states: Xtructurable,
        costs: chex.Array,
        depths: chex.Array,
        trails: Xtructurable,
        trace_indices: chex.Array,
        valid_mask: chex.Array,
        count_generated: bool = True,
    ) -> "IDSearchResult":
        """
        Push a batch of items onto the stack.
        Only pushes items where valid_mask is True.

        ``count_generated`` gates the ``generated_count`` metric: frontier re-pushes at
        each IDA* threshold reset pass False so seed nodes are counted once, not once
        per iteration.
        """
        n_push = jnp.sum(valid_mask.astype(jnp.int32))

        perm = stable_partition_three(valid_mask, jnp.zeros_like(valid_mask, dtype=jnp.bool_))

        current_ptr = self.stack.size.astype(jnp.int32)
        capacity = self.stack.max_size

        safe_n_push = jnp.minimum(n_push, capacity - current_ptr)

        items_sorted = self.ItemCls(
            state=xnp.take(states, perm, axis=0),
            cost=costs[perm],
            depth=depths[perm],
            trail=xnp.take(trails, perm, axis=0),
            trace_index=trace_indices[perm],
        )

        new_val_store = _bounded_scatter_leaf(
            self.stack.val_store,
            pack_rows(self.stack.value_class, items_sorted),
            current_ptr,
            safe_n_push,
        )

        new_ptr = (current_ptr + safe_n_push).astype(self.stack.size.dtype)
        new_generated_count = self.generated_count + jnp.where(count_generated, safe_n_push, 0)

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
        trails: Xtructurable,
        trace_indices: chex.Array,
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
            trail=trails,
            trace_index=trace_indices,
        )

        new_val_store = _bounded_scatter_leaf(
            self.stack.val_store,
            pack_rows(self.stack.value_class, items),
            current_ptr,
            safe_n_push,
        )

        new_ptr = (current_ptr + safe_n_push).astype(self.stack.size.dtype)
        new_generated_count = self.generated_count + safe_n_push

        new_stack = self.stack.replace(val_store=new_val_store, size=new_ptr)

        return self.replace(
            stack=new_stack,
            generated_count=new_generated_count,
        )

    def push_frontier_to_stack(
        self,
        frontier: "IDFrontier",
        bound: chex.Array,
        count_generated: bool = True,
    ) -> "IDSearchResult":
        """
        Push allowed frontier nodes to stack and update next_bound.

        Common logic extracted from both id_astar.py and id_qstar.py.
        This method filters frontier nodes by bound, updates next_bound
        with the minimum f-score of pruned nodes, and pushes valid nodes.

        Args:
            frontier: IDFrontier containing states to push
            bound: Current cost bound (scalar)

        Returns:
            Updated IDSearchResult with nodes pushed and next_bound updated
        """
        fs = frontier.f_scores

        # Filter nodes within bound
        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)

        # Determine next_bound from pruned nodes
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(self.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = self.replace(next_bound=new_next_bound)

        return sr.push_batch(
            frontier.states,
            frontier.costs,
            frontier.depths,
            frontier.trail,
            self.frontier_trace_indices(frontier),
            keep_mask,
            count_generated=count_generated,
        )

    def expand_and_push(
        self,
        node_batch: Xtructurable,
        fs: chex.Array,
        valid: chex.Array,
        update_next_bound: bool = True,
    ) -> "IDSearchResult":
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
            Updated IDSearchResult with nodes pushed to stack
        """
        active_bound = self.bound
        keep_mask = jnp.logical_and(valid, fs <= active_bound + 1e-6)

        # Compact valid nodes
        packed_batch, packed_valid, _, packed_idx = compact_by_valid(node_batch, keep_mask)
        packed_fs = jnp.where(packed_valid, fs[packed_idx], jnp.inf)

        # Sort by f-value descending (worst -> best) for LIFO stack order
        f_key = jnp.where(packed_valid, -packed_fs, jnp.inf)
        perm = jnp.argsort(f_key)
        ordered = xnp.take(packed_batch, perm, axis=0)

        n_push = jnp.sum(packed_valid)

        # Optionally update next_bound from pruned nodes
        sr = self
        if update_next_bound:
            prune_mask = jnp.logical_and(valid, fs > active_bound + 1e-6)
            pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
            min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
            new_next_bound = jnp.minimum(self.next_bound, min_pruned_f).astype(KEY_DTYPE)
            sr = self.replace(next_bound=new_next_bound)

        # Reserve the arena rows first: a full arena caps how many of these children can
        # be pushed at all, since a stack entry without its trace row has no path.
        sr, trace_indices, n_written = sr._append_trace(
            ordered.parent_trace, ordered.action, n_push
        )

        return sr.push_packed_batch(
            ordered.state,
            ordered.cost,
            ordered.depth,
            ordered.trail,
            trace_indices,
            n_written,
        )

    def mark_solved(
        self,
        any_solved: chex.Array,
        solved_state: Xtructurable,
        solved_cost: chex.Array,
        solved_trace: chex.Array,
    ) -> "IDSearchResult":
        """
        Mark search as solved and record the arena row of the solution node.

        Only the trace index is stored here -- walking it out to an action sequence is
        done once after the search loop, not on every inner iteration.

        ``solved_idx`` stays -1: the popped node's stack slot is meaningless once the
        stack is reset. It remains a field because ``cli/search_outcome.py`` gates the
        reported cost on its presence.
        """
        # Pure value selection: where instead of lax.cond — on GPU every cond
        # predicate is a host readback (sync) paid once per search iteration.
        first = jnp.logical_and(any_solved, jnp.logical_not(self.solved))
        new_solution_state = xnp.where(first, solved_state, self.solution_state)
        new_solution_cost = jnp.where(first, solved_cost.astype(KEY_DTYPE), self.solution_cost)
        new_solved_trace = jnp.where(first, solved_trace, self.solved_trace_index)

        return self.replace(
            solved=jnp.logical_or(self.solved, any_solved),
            solution_state=new_solution_state,
            solution_cost=new_solution_cost,
            solved_trace_index=new_solved_trace.astype(TRACE_INDEX_DTYPE),
        )

    def reconstruct_solution_actions(self) -> "IDSearchResult":
        """Walk the arena from the solution node and materialise ``solution_actions_arr``.

        Runs once, after the search loop. The chain is emitted goal-first, so it is
        reversed into a start-first, ``ACTION_PAD``-terminated array of ``max_path_len``.
        """
        mpl = self.max_path_len

        def _cond(carry):
            node, step, _ = carry
            return jnp.logical_and(node != TRACE_INVALID, step < mpl)

        def _body(carry):
            node, step, acc = carry
            idx = node.astype(jnp.int32)
            return (
                self.trace_parent[idx],
                step + 1,
                acc.at[step].set(self.trace_action[idx]),
            )

        node0 = jnp.where(self.solved, self.solved_trace_index, TRACE_INVALID).astype(
            TRACE_INDEX_DTYPE
        )
        init = (node0, jnp.array(0, dtype=jnp.int32), jnp.full((mpl,), ACTION_PAD, ACTION_DTYPE))
        _, length, reversed_actions = jax.lax.while_loop(_cond, _body, init)

        # reversed_actions[0] is the last action taken; flip the first `length` entries.
        positions = jnp.arange(mpl, dtype=jnp.int32)
        source = jnp.clip(length - 1 - positions, 0, mpl - 1)
        actions = jnp.where(positions < length, reversed_actions[source], ACTION_PAD)

        # A frontier-solved search never reaches the stack, so it has no arena row; keep
        # the action history the frontier already produced.
        has_trace = self.solved_trace_index != TRACE_INVALID
        return self.replace(
            solution_actions_arr=jnp.where(
                has_trace, actions.astype(ACTION_DTYPE), self.solution_actions_arr
            )
        )

    def initialize_from_frontier(
        self,
        frontier: IDFrontier,
        cost_weight: float,
        eval_fn: callable,
        params: Any,
    ) -> "IDSearchResult":
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
            solved_idx=jnp.array(-1, dtype=jnp.int32),
            solved_trace_index=TRACE_INVALID,
        ).splice_frontier_trace(frontier)

        # Push frontier nodes within bound to stack
        return sr.push_frontier_to_stack(frontier, start_bound), frontier

    @staticmethod
    def detect_solution(
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        states: Puzzle.State,
        costs: chex.Array,
        payload: chex.Array,
        valid_mask: chex.Array,
    ):
        """Find the first solved state. ``payload`` is whatever identifies its path --
        an action history row for the frontier, an arena trace index for the stack."""
        is_solved_mask = puzzle.batched_is_solved(solve_config, states)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        first_idx = jnp.argmax(is_solved_mask)
        solved_state = xnp.take(states, first_idx[jnp.newaxis], axis=0)
        solved_cost = costs[first_idx]

        return any_solved, solved_state, solved_cost, payload[first_idx], first_idx

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
            parent_traces,
            valid_mask,
        ) = self.get_top_batch(batch_size)

        any_solved, solved_state, solved_cost, solved_trace, _ = self.detect_solution(
            puzzle,
            solve_config,
            parents,
            parent_costs,
            parent_traces,
            valid_mask,
        )

        sr_solved = sr.mark_solved(any_solved, solved_state, solved_cost, solved_trace)

        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)

        return (
            sr,
            sr_solved,
            any_solved,
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_traces,
            valid_mask,
            neighbours,
            step_costs,
        )


def apply_standard_deduplication(
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
) -> chex.Array:
    """Apply in-batch deduplication and non-backtracking; return the filtered valid mask."""
    unique_mask = xnp.unique_mask(
        flat_neighbours,
        key=flat_g,
        filled=flat_valid,
    )
    flat_valid = jnp.logical_and(flat_valid, unique_mask)

    return apply_non_backtracking(
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
        # One arena walk for the whole search, instead of a path copy per node.
        return loop_state.search_result.reconstruct_solution_actions()

    jitted_fn = jax.jit(search_fn)

    if show_compile_time:
        print(f"initializing jit for {name}")
        start_t = time.time()

    if warmup_inputs is None:
        warmup_inputs = (puzzle.SolveConfig.default(), puzzle.State.default())
    compile_with_example(jitted_fn, *warmup_inputs)

    if show_compile_time:
        print(f"{name} JIT compile time: {time.time() - start_t:.2f}s")
        print("JIT compiled\n\n")

    warmup_with_example(jitted_fn, *warmup_inputs)
    return jitted_fn


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
    bound_step: float = 0.0,
):
    """Build the IDA* threshold loop.

    ``bound_step`` rounds each new threshold up onto a ``bound_step`` grid. Without it
    the ladder advances to the next *distinct observed* f-value, and since keys are
    ``float16`` and a neural heuristic emits a dense spread of them, the threshold
    creeps by a single ULP (0.015625 around f=16..32) per pass. Each pass re-derives
    the whole tree from the frontier, so on rubikscube-deepcubea at ``cost_weight=1.0``
    that measured 108-178 thresholds where a step of 1.0 needs 5, for 4-22x fewer
    generated nodes at an identical solution cost on 3 of 4 benchmark samples.

    The grid is eps-admissible, not admissible: the returned cost can exceed the
    optimum by less than ``bound_step`` (the 4th sample went 21 -> 23). Set
    ``bound_step=0`` to restore the exact ladder.
    """
    quantize_bound = bound_step > 0.0

    def outer_cond(loop_state: IDLoopState) -> jnp.ndarray:
        sr = loop_state.search_result
        return jnp.logical_and(~sr.solved, jnp.isfinite(sr.bound))

    def outer_body(loop_state: IDLoopState) -> IDLoopState:
        loop_state = jax.lax.while_loop(inner_cond, inner_body, loop_state)

        sr = loop_state.search_result
        new_bound = sr.next_bound
        if quantize_bound:
            new_bound = (jnp.ceil(new_bound / bound_step) * bound_step).astype(KEY_DTYPE)

        reset_sr = sr.replace(
            bound=new_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            stack=sr.stack.replace(size=jnp.array(0, dtype=jnp.uint32)),
        )

        # Re-seed the same frontier for the next threshold; count_generated=False so
        # these seed nodes are not re-tallied every iteration.
        reset_sr = reset_sr.push_frontier_to_stack(
            loop_state.frontier, new_bound, count_generated=False
        )

        return loop_state.replace(search_result=reset_sr)

    return outer_cond, outer_body
