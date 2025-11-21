"""
JAxtar Search Base Module
This module implements the core search functionality for A*/Q* algorithms using JAX.
The implementation is designed to be fully parallelizable and GPU-compatible.
Key features:
- Pure JAX implementation for ML research
- Batched operations for GPU optimization
- Generic puzzle-agnostic implementation
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashIdx,
    HashTable,
    Xtructurable,
    base_dataclass,
    xtructure_dataclass,
)

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)

POP_BATCH_FILLED_RATIO = 0.99 # ratio of batch to be filled before popping

@xtructure_dataclass
class Parent:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


@xtructure_dataclass
class Current:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@xtructure_dataclass
class Current_with_Parent:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.

    Attributes:
        parent (Parent): The parent state in the search tree
        current (Current): The current state in the search tree
    """

    parent: FieldDescriptor.scalar(dtype=Parent)
    current: FieldDescriptor.scalar(dtype=Current)


@xtructure_dataclass
class Parant_with_Costs:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.
    """

    parent: FieldDescriptor.scalar(dtype=Parent)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    dist: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@base_dataclass
class SearchResult:
    """
    A dataclass containing the data structures used in the A*/Q* search algorithms.
    This class maintains the state of the search process, including open and closed sets,
    priority queue, and path tracking information.

    Implementation Notes:
    - Uses a HashTable for efficient state storage and lookup
    - Maintains a priority queue (BGPQ) for state expansion ordering
    - Tracks costs and parent relationships for path reconstruction
    - Optimized for GPU execution with batched operations

    Attributes:
        hashtable (HashTable): Stores all encountered states for efficient lookup
        priority_queue (BGPQ): Priority queue for ordering state expansions
        pop_ratio (float): The final pop ratio for the search, controlling the
                beam width. It is used to calculate a multiplier `M = max(1.0 + pop_ratio, 1.01)`.
                The search beam will include all nodes with a cost up to `min_cost * M`.
        min_pop (int): Minimum number of states to pop from the priority queue
        cost (chex.Array): Cost array tracking path costs to each state (g value)
        dist (chex.Array): Distance array storing calculated heuristic or Q values
        parent (Parent): Array storing parent state indices for path reconstruction
        solved (chex.Array): Boolean flag indicating if a solution has been found
        solved_idx (Current): Index of the solved state in the hash table
    """

    hashtable: HashTable  # hash table
    priority_queue: BGPQ  # priority queue
    pop_ratio: float  # ratio of states to pop from the priority queue
    min_pop: int  # minimum number of states to pop from the priority queue
    cost: chex.Array  # cost array - g value
    dist: chex.Array  # distance array - calculated heuristic or Q value
    parent: Xtructurable | Parent  # parent array
    solved: chex.Array  # solved array
    solved_idx: Xtructurable | Current  # solved index
    pop_generation: chex.Array  # records which pop generation a node was expanded in
    pop_count: chex.Array  # counter for pop_full calls

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2), static_argnames=("parant_with_costs",))
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        seed=42,
        parant_with_costs: bool = False,
    ):
        """
        Creates a new instance of SearchResult with initialized data structures.

        Args:
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
            pop_ratio (float): Controls the search beam width. It is used to calculate
                a multiplier `M = max(1.0 + pop_ratio, 1.01)`. The search beam will
                include all nodes with a cost up to `min_cost * M`.
                - `pop_ratio = 0`: Results in a narrow beam (`M=1.01`).
                - `pop_ratio > 0`: Creates a wider beam. A larger `pop_ratio`
                                 results in a wider beam.
                - `pop_ratio = inf`: Effectively becomes batched A*, processing all
                                 available nodes in the batch.
            min_pop (int): Minimum number of nodes to pop from the priority queue.
            seed (int): Random seed for hash function initialization

        Returns:
            SearchResult: A new instance with initialized data structures
        """
        # Initialize the hash table for state storage
        hashtable: HashTable = HashTable.build(
            statecls, seed, max_nodes, CUCKOO_TABLE_N, HASH_SIZE_MULTIPLIER
        )

        if parant_with_costs:
            # Initialize priority queue for state expansion
            priority_queue = BGPQ.build(max_nodes, batch_size, Parant_with_Costs, KEY_DTYPE)
        else:
            priority_queue = BGPQ.build(max_nodes, batch_size, Current_with_Parent, KEY_DTYPE)

        # Initialize arrays for tracking costs and state relationships
        # +1 for -1 index as a dummy node
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
        """Current number of states stored in the founded set.
        This is the number of states in the founded set."""
        return self.hashtable.size

    @property
    def opened_size(self) -> int:
        """Current number of states stored in the opened set.
        This is the number of states in the hash table but not in the closed set."""
        return self.generated_size - self.closed_size

    @property
    def closed_size(self) -> int:
        """Current number of states stored in the closed set.
        This is the number of states in the closed set."""
        return jnp.sum(jnp.isfinite(self.cost))

    def pop_full(search_result) -> tuple["SearchResult", Current, chex.Array]:
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently,
        respecting the pop_ratio to control search width without losing nodes.

        Args:
            search_result (SearchResult): The current search state

        Returns:
            tuple: Contains:
                - Updated SearchResult
                - A batch of the best values to be processed
                - A boolean mask indicating which entries in the batch are valid
        """

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        min_key = search_result.mask_unoptimal(min_key, min_val.current)
        buffer = jnp.full(min_key.shape, jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current_with_Parent.default(min_key.shape)

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, key, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0
            
            # Optimization: Stop if batch is mostly full (e.g. 90%)
            # This prevents excessive sorting/merging for diminishing returns
            fill_ratio = jnp.mean(jnp.isfinite(key))
            not_full_enough = fill_ratio < POP_BATCH_FILLED_RATIO
            
            # For pop_ratio = inf, we don't check threshold and fill the batch completely.
            # For other cases, we can stop early, but the final filtering is done later.
            # This loop is mainly for ensuring we have enough candidates to select from.
            return jnp.logical_and(pq_not_empty, not_full_enough)

        def _body(state):
            search_result, key, val, _, _ = state
            # Pop new nodes from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()
            new_key = search_result.mask_unoptimal(new_key, new_val.current)

            # Merge current batch with new nodes, splitting into main and overflow
            main_keys, main_vals, overflow_keys, overflow_vals = _unique_sort_merge_and_split(
                key, val, new_key, new_val
            )
            return search_result, main_keys, main_vals, overflow_keys, overflow_vals

        # Run the loop until we have a full batch of the best available nodes
        search_result, min_key, min_val, overflow_keys, overflow_vals = jax.lax.while_loop(
            _cond, _body, (search_result, min_key, min_val, buffer, buffer_val)
        )

        # Put overflow nodes back into the PQ so they are never lost
        search_result.priority_queue = jax.lax.cond(
            jnp.any(jnp.isfinite(overflow_keys)),
            lambda: search_result.priority_queue.insert(overflow_keys, overflow_vals),
            lambda: search_result.priority_queue,
        )

        # 3. Apply pop_ratio to the full batch
        filled = jnp.isfinite(min_key)
        # Add a small epsilon for floating point comparisons
        threshold = min_key[0] * search_result.pop_ratio + 1e-6

        # Identify nodes to process now vs. nodes to return to PQ
        process_mask = jnp.less_equal(min_key, threshold)
        base_process_mask = jnp.logical_and(filled, process_mask)

        # Enforce min_pop: ensure that we pop at least min_pop nodes if they are available.
        min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= search_result.min_pop, filled)

        # The final mask includes nodes within the threshold OR nodes to meet min_pop.
        final_process_mask = jnp.logical_or(base_process_mask, min_pop_mask)

        # Separate the nodes to be returned and re-insert them into the PQ
        return_keys = jnp.where(final_process_mask, jnp.inf, min_key)
        search_result.priority_queue = search_result.priority_queue.insert(return_keys, min_val)

        # 4. Update the closed set with the nodes we are processing
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            min_val.current.hashidx.index,
            final_process_mask,
            min_val.current.cost,
        )
        search_result.parent = search_result.parent.at[
            min_val.current.hashidx.index
        ].set_as_condition(final_process_mask, min_val.parent)
        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            min_val.current.hashidx.index,
            final_process_mask,
            search_result.pop_count,
        )
        search_result.pop_count += 1

        return search_result, min_val.current, final_process_mask

    def pop_full_with_actions(search_result, puzzle: Puzzle, solve_config: Puzzle.SolveConfig) -> tuple["SearchResult", Current, chex.Array, chex.Array]:
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently,
        respecting the pop_ratio to control search width without losing nodes.

        This version performs EAGER EXPANSION and S' DEDUPLICATION.
        It expands (S, A) -> S' immediately inside the loop to ensure that the
        returned batch contains only unique S' states, preventing batch starvation
        in highly reversible environments.

        Args:
            search_result (SearchResult): The current search state
            puzzle (Puzzle): The puzzle instance
            solve_config (Puzzle.SolveConfig): Configuration for the puzzle

        Returns:
            tuple: Contains:
                - Updated SearchResult
                - A batch of the best values to be processed (Current)
                - Next states (S')
                - A boolean mask indicating which entries in the batch are valid
        """
        
        def _mask_unoptimal(search_result, key, val):
             idx = val.parent.hashidx.index
             current_cost = search_result.cost[idx]
             is_optimal_path_to_parent = jnp.less_equal(val.cost, current_cost + 1e-6)
             return jnp.where(is_optimal_path_to_parent, key, jnp.inf)

        # Helper to sort and deduplicate EXPANDED nodes (based on S')
        def _unique_sort_expanded(key, states, parents, parent_costs, q_dists, ncosts):
            n = key.shape[-1]
            filled = jnp.isfinite(key)
            
            # Deduplicate based on STATE (S')
            # We keep the one with the lowest key (best f-value)
            mask = xnp.unique_mask(states, key, filled)
            
            key = jnp.where(mask, key, jnp.inf)
            sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
            
            return (
                sorted_key,
                states[sorted_idx],
                parents[sorted_idx],
                parent_costs[sorted_idx],
                q_dists[sorted_idx],
                ncosts[sorted_idx]
            )

        # Helper to merge, sort, and split expanded batches
        def _merge_and_split_expanded(k1, s1, p1, pc1, qd1, nc1, k2, s2, p2, pc2, qd2, nc2):
            # Use the outer batch_size which should be static/concrete if passed correctly
            # Or derive from k1 if k1 has static shape
            # batch_size is available from outer scope
            
            mk = jnp.concatenate([k1, k2])
            ms = xnp.concatenate([s1, s2])
            mp = xnp.concatenate([p1, p2])
            mpc = jnp.concatenate([pc1, pc2])
            mqd = jnp.concatenate([qd1, qd2])
            mnc = jnp.concatenate([nc1, nc2])
            
            sk, ss, sp, spc, sqd, snc = _unique_sort_expanded(mk, ms, mp, mpc, mqd, mnc)
            
            # Split main (keep) and overflow (return to PQ)
            return (
                sk[:batch_size], ss[:batch_size], sp[:batch_size], spc[:batch_size], sqd[:batch_size], snc[:batch_size],
                sk[batch_size:], sp[batch_size:], spc[batch_size:], sqd[batch_size:] # Overflow doesn't need states or ncosts
            )

        # 1. First Pop & Expand (Outside Loop)
        # This allows us to determine shapes and initialize buffers without explicit batch_size
        
        (
            search_result.priority_queue,
            first_key,
            first_val,
        ) = search_result.priority_queue.delete_mins()
        
        # Determine batch_size from the first pop
        batch_size = first_key.shape[0]
        
        # Filter stale paths
        first_key = _mask_unoptimal(search_result, first_key, first_val)
        
        # EAGER EXPANSION
        first_parents = first_val.parent
        first_parent_states = search_result.get_state(first_parents)
        first_actions = first_parents.action
        first_filled = jnp.isfinite(first_key)
        
        first_next_states, first_ncosts = puzzle.batched_get_actions(
            solve_config, first_parent_states, first_actions, first_filled
        )
        
        # Deduplicate the first batch
        (
            unique_k, unique_s, unique_p, unique_pc, unique_qd, unique_nc
        ) = _unique_sort_expanded(
            first_key, first_next_states, first_parents, first_val.cost, first_val.dist, first_ncosts
        )
        
        # 2. Initialize Accumulators using the shape of the first batch
        # We use the first unique batch to initialize, but since it might not be full,
        # we need to be careful. Actually, we can just use full_like on the popped key
        # to get the right size buffers.
        
        # Initialize empty buffers with correct size (batch_size)
        acc_keys = jnp.full_like(first_key, jnp.inf)
        
        # For structured types (State, Parent), we use tree_map to create empty arrays of same structure
        # We can use unique_s as a template since it has the right structure, but we need to ensure size is batch_size
        # unique_s already has size batch_size (from _unique_sort_expanded implementation which returns same size)
        # Wait, _unique_sort_expanded returns size N (same as input).
        # Yes, input to _unique_sort_expanded is size batch_size. Output is also size batch_size.
        
        # So we can just use the first unique batch as the starting point for the accumulator!
        # But we need to handle the "merge" logic.
        # Actually, we can treat the "accumulator" as empty initially, and merge the first batch into it.
        # Or simpler: Initialize accumulator with the first unique batch results.
        
        # However, we need to handle overflow if we were merging. But here we just have one batch.
        # So unique_k, unique_s etc. ARE the initial state of the accumulator.
        # But wait, _unique_sort_expanded just sorts. It doesn't "fill" up to batch_size if we had more.
        # It just processes the batch.
        
        # So:
        acc_keys = unique_k
        acc_states = unique_s
        acc_parents = unique_p
        acc_parent_costs = unique_pc
        acc_q_dists = unique_qd
        acc_ncosts = unique_nc
        
        # We don't have overflow from the first batch because we didn't merge with anything.
        # We just sorted/deduplicated it.
        
        # 3. Loop to fill the batch
        def _cond(state):
            search_result, key, _, _, _, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0
            
            # Check if we have enough valid unique S'
            fill_ratio = jnp.mean(jnp.isfinite(key))
            not_full_enough = fill_ratio < 0.99 # Try to fill as much as possible
            
            return jnp.logical_and(pq_not_empty, not_full_enough)

        def _body(state):
            search_result, ak, ast, ap, apc, aqd, anc, _ = state
            
            # Pop from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()
            
            # Filter stale paths
            new_key = _mask_unoptimal(search_result, new_key, new_val)
            
            # EAGER EXPANSION
            parents = new_val.parent
            parent_states = search_result.get_state(parents)
            actions = parents.action
            filled = jnp.isfinite(new_key)
            
            next_states, ncosts = puzzle.batched_get_actions(
                solve_config, parent_states, actions, filled
            )
            
            # Merge with accumulated batch
            (
                main_k, main_s, main_p, main_pc, main_qd, main_nc,
                over_k, over_p, over_pc, over_qd
            ) = _merge_and_split_expanded(
                ak, ast, ap, apc, aqd, anc,
                new_key, next_states, parents, new_val.cost, new_val.dist, ncosts
            )
            
            # Push overflow back to PQ
            over_val = Parant_with_Costs(parent=over_p, cost=over_pc, dist=over_qd)
            
            search_result.priority_queue = jax.lax.cond(
                jnp.any(jnp.isfinite(over_k)),
                lambda: search_result.priority_queue.insert(over_k, over_val),
                lambda: search_result.priority_queue,
            )
            
            return search_result, main_k, main_s, main_p, main_pc, main_qd, main_nc, jnp.array(0) # dummy

        # Run the loop
        (
            search_result, 
            final_keys, final_states, final_parents, final_parent_costs, final_q_dists, final_ncosts, _
        ) = jax.lax.while_loop(
            _cond, _body, 
            (search_result, acc_keys, acc_states, acc_parents, acc_parent_costs, acc_q_dists, acc_ncosts, jnp.array(0))
        )

        # 4. Apply pop_ratio to the full batch of UNIQUE S'
        filled = jnp.isfinite(final_keys)
        threshold = final_keys[0] * search_result.pop_ratio + 1e-6
        
        process_mask = jnp.less_equal(final_keys, threshold)
        base_process_mask = jnp.logical_and(filled, process_mask)
        min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= search_result.min_pop, filled)
        final_process_mask = jnp.logical_or(base_process_mask, min_pop_mask)
        
        # Return unselected nodes to PQ
        return_keys = jnp.where(final_process_mask, jnp.inf, final_keys)
        return_vals = Parant_with_Costs(parent=final_parents, cost=final_parent_costs, dist=final_q_dists)
        search_result.priority_queue = search_result.priority_queue.insert(return_keys, return_vals)

        # 5. Insert selected UNIQUE S' into Hashtable
        # OPTIMIZATION: Use cached final_ncosts
        
        next_costs = final_parent_costs + final_ncosts
        next_dists = final_q_dists - final_ncosts # Assuming Q(s,a) = h(s') + c(s,a) => h(s') = Q - c
        
        (
            search_result.hashtable,
            new_states_mask,
            cheapest_uniques_mask,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(final_states, final_process_mask, next_costs)

        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            cheapest_uniques_mask,
            next_costs,
        )
        search_result.dist = xnp.update_on_condition(
            search_result.dist,
            hash_idx.index,
            cheapest_uniques_mask,
            next_dists,
        )
        search_result.parent = search_result.parent.at[
            hash_idx.index
        ].set_as_condition(cheapest_uniques_mask, final_parents)

        min_val = Current(hashidx=hash_idx, cost=next_costs)
        
        return search_result, min_val, final_states, cheapest_uniques_mask

    def get_solved_path(search_result) -> list[Parent]:
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        solved_idx = search_result.solved_idx

        path = [solved_idx]
        parent_last = search_result.get_parent(solved_idx)
        visited = set()
        while True:
            idx = int(parent_last.hashidx.index)
            if parent_last.hashidx.index == -1:
                break
            if idx in visited:
                print(f"Loop detected in path reconstruction at index {idx}")
                break
            visited.add(idx)
            path.append(parent_last)
            parent_last = search_result.get_parent(parent_last)
        path.reverse()
        return path

    def get_state(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> Puzzle.State:
        """
        Get the state from the hash table.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.hashtable[idx.current.hashidx]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.hashtable[idx.hashidx]
        elif isinstance(idx, HashIdx):
            return search_result.hashtable[idx]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_cost(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> chex.Array:
        """
        Get the cost of the state from the cost array.
        """
        if isinstance(idx, (Current_with_Parent, Parant_with_Costs)):
            return search_result.cost[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.cost[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.cost[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_dist(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> chex.Array:
        """
        Get the distance of the state from the distance array.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.dist[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.dist[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.dist[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_parent(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> Parent:
        """
        Get the parent action from the parent action array.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.parent[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.parent[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.parent[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def mask_unoptimal(
        search_result, min_key: chex.Array, min_current: Current
    ) -> chex.Array:
        """
        Mask the unoptimal states.
        """
        optimal = jnp.less_equal(min_current.cost, search_result.get_cost(min_current))
        return jnp.where(optimal, min_key, jnp.inf)


def unique_sort(
    key: chex.Array, val: Current_with_Parent
) -> tuple[chex.Array, Current_with_Parent]:
    """
    Sorts the keys and corresponding values.

    Args:
        key (chex.Array): Array of keys.
        val (Current_with_Parent): Array of values.

    Returns:
        tuple:
            - sorted_key (chex.Array): Sorted array of keys.
            - sorted_val (Current_with_Parent): Values corresponding to the sorted keys.
    """
    n = key.shape[-1]
    filled = jnp.isfinite(key)
    mask = xnp.unique_mask(val.current.hashidx, val.current.cost, filled)
    key = jnp.where(mask, key, jnp.inf)
    sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
    sorted_val = val[sorted_idx]
    return sorted_key, sorted_val

# Helper to merge, sort, and split two batches of nodes
def _unique_sort_merge_and_split(k1, v1, k2, v2):
    batch_size = k1.shape[-1]
    merged_key = jnp.concatenate([k1, k2])
    merged_val = xnp.concatenate([v1, v2])

    # Sort and remove duplicates from the combined batch
    sorted_key, sorted_val = unique_sort(merged_key, merged_val)

    # Split into a main batch and an overflow batch
    main_keys = sorted_key[:batch_size]
    main_vals = sorted_val[:batch_size]
    overflow_keys = sorted_key[batch_size:]
    overflow_vals = sorted_val[batch_size:]
    return main_keys, main_vals, overflow_keys, overflow_vals

def unique_sort_parents(
    key: chex.Array, val: Parant_with_Costs
) -> tuple[chex.Array, Parant_with_Costs]:
    """
    Sorts the keys and corresponding values for Parant_with_Costs.
    Deduplicates based on (parent_index, action) pair to minimize redundant expansions.
    Uses Packed ID (parent_idx << 32 | action) as a proxy for the resulting state hash.
    """
    n = key.shape[-1]
    filled = jnp.isfinite(key)

    mask = xnp.unique_mask(val.parent, val.cost, filled)
    
    key = jnp.where(mask, key, jnp.inf)
    sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
    sorted_val = val[sorted_idx]
    return sorted_key, sorted_val

def _unique_sort_merge_and_split_parents(k1, v1, k2, v2):
    batch_size = k1.shape[-1]
    merged_key = jnp.concatenate([k1, k2])
    merged_val = xnp.concatenate([v1, v2])

    # Sort and remove duplicates from the combined batch
    sorted_key, sorted_val = unique_sort_parents(merged_key, merged_val)

    # Split into a main batch and an overflow batch
    main_keys = sorted_key[:batch_size]
    main_vals = sorted_val[:batch_size]
    overflow_keys = sorted_key[batch_size:]
    overflow_vals = sorted_val[batch_size:]
    return main_keys, main_vals, overflow_keys, overflow_vals
