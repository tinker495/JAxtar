"""
Shared core of the A*, A*-D and Q* searches.

Holds the queue value types, the search result buffers and host-side readers that
both expansion families use, and the priority-queue helpers whose semantics must
stay identical across them. The eager family (A*) lives in `search_base_eager.py`; the
deferred family (A*-D, Q*) lives in `search_base_deferred.py`.
"""

from dataclasses import replace
from functools import partial
from typing import Annotated, Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashIdx,
    HashTable,
    Xtructurable,
    XtructurableT,
)
from xtructure.core.dataclass import base_dataclass

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)
from JAxtar.expansion_trace import ExpansionTrace
from JAxtar.solution_trace import (
    SolutionTrace,
    action_pad_int,
)

POP_BATCH_FILLED_RATIO = 0.99  # ratio of batch to be filled before popping

SearchResultT = TypeVar("SearchResultT", bound="SearchResult")
# Priority-queue value of a search family (`Current` or `ParentWithCosts`).
QueueValueT = TypeVar("QueueValueT", bound=Xtructurable)

# Fields every search result keeps as static pytree metadata; families may add more.
SEARCH_RESULT_STATIC_FIELDS = ("action_size", "fixed_batch_pop")


def select_pop_batch(
    priority_queue: BGPQ[XtructurableT],
    key: jax.Array,
    value: XtructurableT,
    *,
    pop_ratio: jax.Array,
    min_pop: int,
    fixed_batch_pop: bool,
) -> tuple[BGPQ[XtructurableT], jax.Array]:
    """Select valid candidates and retain any ratio-rejected entries in the queue."""
    filled = jnp.isfinite(key)
    if fixed_batch_pop:
        return priority_queue, filled

    threshold = key[0] * pop_ratio + 1e-6
    process_mask = jnp.less_equal(key, threshold)

    def _limited_pop(queue: BGPQ[XtructurableT]):
        min_pop_mask = jnp.cumsum(filled) <= min_pop
        selected = filled & (process_mask | min_pop_mask)
        return queue.insert(jnp.where(selected, jnp.inf, key), value), selected

    # Valid entries in the first min_pop slots always satisfy the count-based floor.
    return jax.lax.cond(
        jnp.all(~filled | process_mask | (jnp.arange(key.shape[0]) < min_pop)),
        lambda queue: (queue, filled),
        _limited_pop,
        priority_queue,
    )


def insert_priority_queue_batches(
    search_result: SearchResultT,
    keys,
    vals,
    masks: jax.Array,
    *,
    presorted: bool = False,
    prefix_rows: bool = False,
) -> SearchResultT:
    """Insert action-major candidate batches into the priority queue.

    Shared by both expansion families (single and bidirectional) to keep
    insertion semantics identical before the family's pop.
    ``presorted`` skips BGPQ's per-row sort when the flattened candidate set
    was already sorted before being split into rows.
    ``prefix_rows`` is the caller's promise that the selected entries occupy a
    prefix of the flattened batch (eager A* partitions candidates that way, Q*
    sorts them globally), so the rows with work are the first
    ``ceil(count / row_width)``. They go through BGPQ's multi-row insert, which
    sinks all rows in one batched wavefront instead of one full insert per row.
    """

    def _insert(sr: SearchResultT, key_row, val_row):
        insert = sr.priority_queue.insert_sorted if presorted else sr.priority_queue.insert
        sr.priority_queue = insert(key_row, val_row)
        return sr

    def _scan(sr: SearchResultT, row):
        key_row, val_row, mask_row = row
        sr = jax.lax.cond(
            jnp.any(mask_row),
            _insert,
            lambda current_sr, *args: current_sr,
            sr,
            key_row,
            val_row,
        )
        return sr, None

    if prefix_rows:
        row_width = keys.shape[1]
        n_rows = (jnp.sum(masks, dtype=jnp.int32) + row_width - 1) // row_width
        pq = search_result.priority_queue
        insert_rows = pq.insert_sorted_rows if presorted else pq.insert_rows
        search_result.priority_queue = insert_rows(keys, vals, n_rows)
    else:
        search_result, _ = jax.lax.scan(_scan, search_result, (keys, vals, masks))

    def _update_pq_insert(sr: SearchResultT):
        row_any = jnp.any(masks, axis=1)
        return replace(
            sr,
            xtr_pq_insert_calls=sr.xtr_pq_insert_calls + jnp.sum(row_any).astype(jnp.int32),
            xtr_pq_insert_items=sr.xtr_pq_insert_items + jnp.sum(masks).astype(jnp.int32),
        )

    search_result = jax.lax.cond(
        search_result.xtr_enabled,
        _update_pq_insert,
        lambda sr: sr,
        search_result,
    )
    return search_result


def record_pq_pop(
    search_result: SearchResultT,
    process_mask: jax.Array,
    delete_calls: jax.Array,
    popped_total: jax.Array,
) -> SearchResultT:
    """Add one pop's priority-queue workload counters when the xtr_* counters are enabled."""

    def update(sr: SearchResultT) -> SearchResultT:
        processed = jnp.sum(process_mask).astype(jnp.int32)
        return replace(
            sr,
            xtr_steps=sr.xtr_steps + jnp.array(1, dtype=jnp.int32),
            xtr_pq_delete_calls=sr.xtr_pq_delete_calls + delete_calls,
            xtr_pq_popped_items=sr.xtr_pq_popped_items + popped_total,
            xtr_pq_processed_items=sr.xtr_pq_processed_items + processed,
            xtr_pq_requeued_items=sr.xtr_pq_requeued_items + (popped_total - processed),
            xtr_pq_heap_size_sum=sr.xtr_pq_heap_size_sum
            + sr.priority_queue.heap_size.astype(jnp.int32),
            xtr_pq_buffer_size_sum=sr.xtr_pq_buffer_size_sum
            + sr.priority_queue.buffer_size.astype(jnp.int32),
            xtr_pq_size_samples=sr.xtr_pq_size_samples + jnp.array(1, dtype=jnp.int32),
        )

    return jax.lax.cond(search_result.xtr_enabled, update, lambda sr: sr, search_result)


def insert_start(search_result: "SearchResult", start: Xtructurable) -> HashIdx:
    """Insert the start state with cost 0 (mutates `search_result`) and return its slot."""
    search_result.hashtable, _, hash_idx = search_result.hashtable.insert(start)
    search_result.cost = search_result.cost.at[hash_idx.index].set(0)
    return hash_idx


def search_continues(
    puzzle: Puzzle,
    search_result: "SearchResult",
    solve_config: Puzzle.SolveConfig,
    states: Xtructurable,
    filled: jax.Array,
) -> jax.Array:
    """Loop condition shared by both families: work remains, room remains, nothing solved."""
    size_cond1 = filled.any()  # queue is not empty
    size_cond2 = search_result.hashtable.size < search_result.capacity  # hash table is not full
    size_cond = jnp.logical_and(size_cond1, size_cond2)

    solved = puzzle.batched_is_solved(solve_config, states)
    solved = jnp.logical_and(solved, filled)
    return jnp.logical_and(size_cond, ~solved.any())


@partial(jax.jit, static_argnames=("max_steps",))
def _reconstruct_path_arrays(
    parent_indices: jax.Array,
    costs: jax.Array,
    dists: jax.Array,
    solved_index: jax.Array,
    max_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    parent_indices = parent_indices.astype(jnp.int32)
    path_idx = jnp.full((max_steps,), -1, dtype=jnp.int32)
    path_costs = jnp.full((max_steps,), jnp.nan, dtype=costs.dtype)
    path_dists = jnp.full((max_steps,), jnp.nan, dtype=dists.dtype)

    def _cond(carry):
        idx, step, *_ = carry
        return jnp.logical_and(idx >= 0, step < max_steps)

    def _body(carry):
        idx, step, path_idx, path_costs, path_dists = carry
        path_idx = path_idx.at[step].set(idx)
        path_costs = path_costs.at[step].set(costs[idx])
        path_dists = path_dists.at[step].set(dists[idx])
        next_idx = parent_indices[idx]
        return next_idx, step + 1, path_idx, path_costs, path_dists

    init = (
        solved_index.astype(jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        path_idx,
        path_costs,
        path_dists,
    )
    _, length, path_idx, path_costs, path_dists = jax.lax.while_loop(_cond, _body, init)

    def _detect_loop():
        valid_mask = jnp.arange(max_steps) < length
        sentinel_base = jnp.iinfo(jnp.int32).max
        sentinel = sentinel_base - jnp.arange(max_steps, dtype=jnp.int32)
        safe_used = jnp.where(valid_mask, path_idx, sentinel)
        sorted_used = jnp.sort(safe_used)
        return jnp.any(sorted_used[1:] == sorted_used[:-1])

    def _detect_corruption():
        valid_mask = jnp.arange(max_steps) < length
        pair_valid = jnp.logical_and(valid_mask[1:], valid_mask[:-1])
        next_costs = path_costs[1:]
        prev_costs = path_costs[:-1]
        return jnp.any(jnp.logical_and(next_costs > prev_costs, pair_valid))

    loop_detected = jax.lax.cond(length > 1, _detect_loop, lambda: jnp.array(False))
    corruption_detected = jax.lax.cond(length > 1, _detect_corruption, lambda: jnp.array(False))
    return path_idx, path_costs, path_dists, length, loop_detected, corruption_detected


class Parent(Xtructurable):

    hashidx: Annotated[HashIdx, FieldDescriptor.scalar(dtype=HashIdx)]
    action: Annotated[jax.Array, FieldDescriptor.scalar(dtype=ACTION_DTYPE)]


class Current(Xtructurable):

    hashidx: Annotated[HashIdx, FieldDescriptor.scalar(dtype=HashIdx)]
    cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=KEY_DTYPE)]


@base_dataclass(static_fields=SEARCH_RESULT_STATIC_FIELDS)
class SearchResult(Generic[QueueValueT]):
    """
    Buffers shared by the A*/A*-D/Q* searches; instantiate a family subclass.

    `EagerSearchResult` (search_base_eager.py) queues `Current` states and
    `DeferredSearchResult` (search_base_deferred.py) queues `ParentWithCosts`
    edges. Both share the hash table, the per-state cost/dist/parent buffers,
    path reconstruction, and the xtr_* workload counters defined here.

    Attributes:
        hashtable (HashTable): Stores all encountered states for efficient lookup
        priority_queue (BGPQ): Priority queue for ordering state expansions
        action_size (int): Number of actions available from each state
        pop_ratio (float): The final pop ratio for the search, controlling the
                beam width. It is used to calculate a multiplier `M = max(1.0 + pop_ratio, 1.01)`.
                The search beam will include all nodes with a cost up to `min_cost * M`.
        min_pop (int): Minimum number of states to pop from the priority queue
        cost (jax.Array): Cost array tracking path costs to each state (g value)
        dist (jax.Array): Distance array storing calculated heuristic or Q values
        parent (Parent): Array storing parent state indices for path reconstruction
        solved (jax.Array): Boolean flag indicating if a solution has been found
        solved_idx (Current): Index of the solved state in the hash table
    """

    hashtable: HashTable  # hash table
    priority_queue: BGPQ[QueueValueT]  # priority queue
    action_size: int  # number of actions available from each state
    pop_ratio: jax.Array  # normalized multiplier for the candidate score threshold
    min_pop: int  # minimum number of states to pop from the priority queue
    fixed_batch_pop: bool  # process every valid candidate without a ratio limit
    cost: jax.Array  # cost array - g value
    dist: jax.Array  # distance array - calculated heuristic or Q value
    parent: Parent  # parent array
    solved: jax.Array  # solved array
    solved_idx: Current  # solved index
    pop_generation: jax.Array  # records which pop generation a node was expanded in
    pop_count: jax.Array  # counter for pop_full calls
    xtr_enabled: jax.Array  # enables xtr_* workload counters
    xtr_steps: jax.Array
    xtr_cand_total: jax.Array
    xtr_cand_valid: jax.Array
    xtr_cand_unique: jax.Array
    xtr_accept: jax.Array
    xtr_ht_inserted: jax.Array
    xtr_ht_lookup: jax.Array
    xtr_ht_found: jax.Array
    xtr_pq_insert_calls: jax.Array
    xtr_pq_insert_items: jax.Array
    xtr_pq_delete_calls: jax.Array
    xtr_pq_popped_items: jax.Array
    xtr_pq_processed_items: jax.Array
    xtr_pq_requeued_items: jax.Array
    xtr_pq_heap_size_sum: jax.Array
    xtr_pq_buffer_size_sum: jax.Array
    xtr_pq_size_samples: jax.Array

    @staticmethod
    def _shared_buffers(
        queue_value: type[Xtructurable],
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        action_size: int,
        pop_ratio: float,
        min_pop: int,
        seed: int,
        emit_workload_signature: bool,
    ) -> dict[str, Any]:
        """
        Allocates the shared buffers around a priority queue of `queue_value` entries.

        Returns the constructor fields every family shares; each family's jitted
        `build` constructs itself from them plus its own fields.

        Args:
            queue_value: Priority-queue value type of the family
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
            action_size (int): Number of actions available from each state
            pop_ratio (float): Compile-time search beam width. It is used to calculate
                a multiplier `M = max(1.0 + pop_ratio, 1.01)`. The search beam will
                include all nodes with a cost up to `min_cost * M`.
                - `pop_ratio = 0`: Results in a narrow beam (`M=1.01`).
                - `pop_ratio > 0`: Creates a wider beam. A larger `pop_ratio`
                                 results in a wider beam.
                - `pop_ratio = inf`: Effectively becomes batched A*, processing all
                                 available nodes in the batch.
            min_pop (int): Minimum number of nodes to pop from the priority queue.
            seed (int): Random seed for hash function initialization
            emit_workload_signature (bool): Enables the xtr_* workload counters
        """
        hashtable: HashTable = HashTable.build(
            statecls, seed, max_nodes, CUCKOO_TABLE_N, HASH_SIZE_MULTIPLIER
        )
        priority_queue = BGPQ.build(max_nodes, batch_size, queue_value, KEY_DTYPE)

        # Initialize arrays for tracking costs and state relationships
        # +1 for -1 index as a dummy node
        cost = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        dist = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        parent = Parent.default(hashtable.table.shape.batch)
        solved = jnp.array(False)
        solved_idx = Current.default((1,))
        pop_generation = jnp.full(hashtable.table.shape.batch, -1, dtype=jnp.int32)
        pop_count = jnp.array(0, dtype=jnp.int32)
        xtr_enabled = jnp.array(emit_workload_signature, dtype=jnp.bool_)
        xtr_zero = jnp.array(0, dtype=jnp.int32)

        return dict(
            hashtable=hashtable,
            priority_queue=priority_queue,
            action_size=action_size,
            pop_ratio=jnp.maximum(1.0 + jnp.asarray(pop_ratio), 1.01),
            min_pop=min_pop,
            fixed_batch_pop=pop_ratio == float("inf"),
            cost=cost,
            dist=dist,
            parent=parent,
            solved=solved,
            solved_idx=solved_idx,
            pop_generation=pop_generation,
            pop_count=pop_count,
            xtr_enabled=xtr_enabled,
            xtr_steps=jnp.array(0, dtype=jnp.int32),
            xtr_cand_total=xtr_zero,
            xtr_cand_valid=xtr_zero,
            xtr_cand_unique=xtr_zero,
            xtr_accept=xtr_zero,
            xtr_ht_inserted=xtr_zero,
            xtr_ht_lookup=xtr_zero,
            xtr_ht_found=xtr_zero,
            xtr_pq_insert_calls=xtr_zero,
            xtr_pq_insert_items=xtr_zero,
            xtr_pq_delete_calls=xtr_zero,
            xtr_pq_popped_items=xtr_zero,
            xtr_pq_processed_items=xtr_zero,
            xtr_pq_requeued_items=xtr_zero,
            xtr_pq_heap_size_sum=xtr_zero,
            xtr_pq_buffer_size_sum=xtr_zero,
            xtr_pq_size_samples=xtr_zero,
        )

    @property
    def capacity(self) -> int:
        """Maximum number of states that can be stored."""
        return self.hashtable.capacity

    @property
    def batch_size(self) -> int:
        """Batch size of the search."""
        return self.priority_queue.batch_size

    @property
    def generated_size(self) -> int:
        """Number of states the search generated.

        Eager expansion inserts every child into the hash table, so hash usage is the
        generated count. `DeferredSearchResult` overrides this. Use `hashtable.size`
        directly wherever hash occupancy itself is meant.
        """
        return self.hashtable.size

    def _get_solved_path(search_result) -> list[Parent | Current]:
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        solved_idx = search_result.solved_idx
        solved_index = int(jax.device_get(solved_idx.hashidx.index))
        max_steps = max(1, int(jax.device_get(search_result.hashtable.size)) + 1)

        (
            path_idx,
            path_costs,
            path_dists,
            length,
            loop_detected,
            corruption_detected,
        ) = _reconstruct_path_arrays(
            search_result.parent.hashidx.index,
            search_result.cost,
            search_result.dist,
            jnp.array(solved_index, dtype=jnp.int32),
            max_steps=max_steps,
        )
        path_idx = jax.device_get(path_idx)
        length = int(jax.device_get(length))
        loop_detected = bool(jax.device_get(loop_detected))
        corruption_detected = bool(jax.device_get(corruption_detected))

        path: list[Parent | Current] = [solved_idx]
        steps_added = 0
        for step in range(max(0, length - 1)):
            parent_idx = int(path_idx[step + 1])
            if parent_idx == -1:
                break
            path.append(search_result.parent[int(path_idx[step])])
            steps_added += 1

        path.reverse()

        loop_idx = int(path_idx[length - 1]) if length > 0 else -1
        if loop_detected:
            print(
                "PATH_RECONSTRUCTION_DIAGNOSTIC | "
                f"loop_detected=True | loop_idx={loop_idx} | "
                "corruption_detected=False"
            )

        # Check that costs are monotonically increasing to prevent path corruption
        if corruption_detected:
            path_costs = jax.device_get(path_costs)
            path_dists = jax.device_get(path_dists)
            used_len = steps_added + 1
            costs = [float(val) for val in path_costs[:used_len]]
            dists = [float(val) for val in path_dists[:used_len]]
            costs.reverse()
            dists.reverse()

            first_drop_idx = next(
                (idx for idx in range(1, len(costs)) if costs[idx] < costs[idx - 1]),
                None,
            )
            drop_details = ""
            if first_drop_idx is not None:
                drop_details = (
                    f" | prev_cost={costs[first_drop_idx - 1]}"
                    f" | next_cost={costs[first_drop_idx]}"
                )
            raise AssertionError(
                "PATH_RECONSTRUCTION_DIAGNOSTIC | "
                f"loop_detected={loop_detected} | "
                f"loop_idx={loop_idx} | "
                "corruption_detected=True | "
                f"first_cost_drop_idx={first_drop_idx}"
                f"{drop_details} | "
                f"costs={costs} | "
                f"dists={dists}"
            )

        return path

    def to_solution_trace(
        search_result,
        *,
        puzzle: Puzzle | None = None,
    ) -> SolutionTrace:
        """Return the host-side solution trace for CLI/evaluation adapters."""
        if not bool(jax.device_get(search_result.solved)):
            return SolutionTrace.unsolved()

        action_pad = action_pad_int(ACTION_DTYPE)
        path = search_result._get_solved_path()
        raw_actions = [
            node.action if isinstance(node, Parent) else action_pad for node in path[:-1]
        ]
        states = tuple(search_result.get_state(node) for node in path)
        costs = tuple(float(jax.device_get(search_result.get_cost(node))) for node in path)
        dists = tuple(float(jax.device_get(search_result.get_dist(node))) for node in path)

        return SolutionTrace.from_raw(
            solved=True,
            raw_actions=raw_actions,
            action_pad=action_pad,
            states=states,
            costs=costs,
            dists=dists,
        )

    def to_expansion_trace(search_result) -> ExpansionTrace | None:
        """Return the host-side expansion trace for evaluation adapters."""
        solved_index = None
        if bool(jax.device_get(search_result.solved)):
            solved_index = int(jax.device_get(search_result.solved_idx.hashidx.index))
        return ExpansionTrace.from_raw(
            pop_generation=search_result.pop_generation,
            cost=search_result.cost,
            dist=search_result.dist,
            parent_indices=search_result.parent.hashidx.index,
            solved_index=solved_index,
        )

    def get_state(search_result, idx: HashIdx | Current | Parent) -> Puzzle.State:
        """
        Get the state from the hash table.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.hashtable[idx.hashidx]
        elif isinstance(idx, HashIdx):
            return search_result.hashtable[idx]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_cost(search_result, idx: HashIdx | Current | Parent) -> jax.Array:
        """
        Get the cost of the state from the cost array.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.cost[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.cost[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_dist(search_result, idx: HashIdx | Current | Parent) -> jax.Array:
        """
        Get the distance of the state from the distance array.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.dist[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.dist[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
