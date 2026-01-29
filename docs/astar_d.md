# A\* Deferred Command (`astar_d`)

The `astar_d` command solves a puzzle using the A\* Deferred search algorithm. This is a variation of A\* where node expansion is deferred, which can be beneficial in certain search spaces or when using specific types of heuristics (e.g., heavy heuristics). It maintains the optimality guarantees of A\* under consistent heuristics.

## Usage

The basic syntax for the `astar_d` command is:

```bash
python main.py astar_d [OPTIONS]
```

Example:

```bash
python main.py astar_d -p rubikscube -nn
```

## Options

The `astar_d` command uses the same option groups as the standard `astar` command.

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment to be solved.

-   `-p, --puzzle`: Specifies the puzzle to solve.
    -   Type: `Choice`
    -   Default: `n-puzzle`
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A comma-separated list of seeds for generating initial puzzle states.
    -   Type: `String`
    -   Default: `"0"`

### Search Options (`@search_options`)

These options control the behavior of the search algorithm.

-   `-m, --max_node_size`: The maximum number of nodes to explore.
    -   Type: `String`
    -   Default: `2e6`
-   `-b, --batch_size`: The number of nodes to process in a single batch.
    -   Type: `Integer`
    -   Default: `10000`
-   `-w, --cost_weight`: The weight `w` for the path cost.
    -   Type: `Float`
    -   Default: `0.9`
-   `-pr, --pop_ratio`: Ratio for popping nodes from the priority queue.
    -   Type: `Float`
    -   Default: `inf`
-   `-vm, --vmap_size`: The number of different initial states to solve in parallel.
    -   Type: `Integer`
    -   Default: `1`
-   `--debug`: Disables JIT compilation.
    -   Type: `Flag`
-   `--profile`: Enables profiler.
    -   Type: `Flag`
-   `--show_compile_time`: Prints compilation time.
    -   Type: `Flag`

### Heuristic Options (`@heuristic_options`)

-   `-nn, --neural_heuristic`: Use a pre-trained neural network as the heuristic function.
    -   Type: `Flag`
-   `--param-path`: Path to the heuristic parameter file.
    -   Type: `String`
-   `--model-type`: Type of the heuristic model.
    -   Type: `String`

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Renders the solution path in the terminal.
    -   Type: `Flag`
-   `-vi, --visualize_imgs`: Generates images and GIF for the solution.
    -   Type: `Flag`
-   `-mt, --max_animation_time`: Max duration for GIF.
    -   Type: `Integer`

## Implementation & Architecture (`JAxtar/stars/astar_d.py`)

A* Deferred is a refined variant designed to handle expensive heuristic evaluations and high branching factors by delaying state generation.

### Key Differences from Standard A*

| Feature | Standard A* (`astar.py`) | A* Deferred (`astar_d.py`) |
| :--- | :--- | :--- |
| **Expansion** | Immediate: Children generated in `loop_body`. | Deferred: Children generated in `pop_full_with_actions` *after* being popped. |
| **PQ Contents** | Fully generated states with $f = g + h$. | Actions/Edges with priorities (often parent's $f$ or look-ahead $f$). |
| **Heuristic Eval** | Every generated child is evaluated immediately. | Only popped or promising children are evaluated (via pruning). |
| **Best For** | Cheap transitions, low branching factor. | Expensive heuristics, wide search trees. |

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (astar_d_builder)"]
    B1["_astar_d_loop_builder"] --> B2["create loop functions"]
    B2 --> B3["jax.jit(astar_d)"]
    B3 --> B4["Warm-up with default config/state"]
  end

  subgraph Run["Run time (astar_d_fn)"]
    R1["call compiled astar_d_fn"] --> R2["init_loop_state"]
    R2 --> R3["jax.lax.while_loop"]
    R3 --> R4["final solved check"]
    R4 --> R5["return SearchResult"]
  end

  subgraph Init["init_loop_state"]
    I1["SearchResult.build (parant_with_costs=True)"] --> I2["insert start state"]
    I2 --> I3["Create LoopStateWithStates (contains start state)"]
  end

  subgraph While["while loop (lax.while_loop)"]
    W1["loop_condition"] --> W2{"Not Solved & Needs More?"}
    W2 -- yes --> W3["loop_body"] --> W1
    W2 -- no --> W4["exit"]
  end

  R2 --> I1
  R3 --> W1
```

### Data Structures At A Glance

The deferred variant uses a modified `SearchResult` and `LoopStateWithStates` that differ from standard A* in key ways:

```mermaid
flowchart TB
  subgraph SR["SearchResult (with parant_with_costs=True)"]
    HT["hashtable\n(state to index mapping)"]
    PQ["priority_queue\n(key to Parant_with_Costs)"]
    COST["cost table\n(g values)"]
    DIST["dist table\n(h values cache)"]
    PARENT["parent table\n(Parent struct: hashidx, action)"]
    POPGEN["pop_generation\n(tracking expansion order)"]
    POPCOUNT["pop_count\n(number of pop operations)"]
  end

  subgraph PC["Parant_with_Costs (PQ Value Type)"]
    PC1["parent: Parent\n(hashidx, action)"]
    PC2["cost: float32\n(g-value at parent)"]
    PC3["dist: float32\n(h-value at child)"]
  end

  subgraph LS["LoopStateWithStates"]
    LS1["search_result: SearchResult"]
    LS2["solve_config: Puzzle.SolveConfig"]
    LS3["params: Any\n(heuristic parameters)"]
    LS4["current: Current\n(hashidx, cost)"]
    LS5["states: Puzzle.State\n(materialized states)"]
    LS6["filled: bool[batch_size]\n(active batch entries)"]
  end

  subgraph CUR["Current (Popped State Reference)"]
    CUR1["hashidx: HashIdx"]
    CUR2["cost: float32 (g)"]
  end

  subgraph PAR["Parent (Backpointer)"]
    PAR1["hashidx: HashIdx"]
    PAR2["action: uint8"]
  end

  PQ -.-> PC
  PARENT -.-> PAR
  LS4 -.-> CUR
  PC1 -.-> PAR
```

**Key Differences from Standard A\***:
- **PQ stores actions/edges**: `Parant_with_Costs` contains the parent state + action + costs, not fully expanded states
- **Materialized states in loop**: `LoopStateWithStates.states` holds the actual states after `pop_full_with_actions`
- **Deferred expansion**: Child states are generated *after* being popped, not during parent expansion

### Loop Body Data Flow (One Iteration)

The deferred implementation in JAxtar defaults to `look_ahead_pruning=True`, which proactively simulates neighbors to avoid inserting non-optimal actions into the priority queue. This is a key optimization that distinguishes this implementation from canonical A* deferred.

The loop body follows a pipeline of five distinct phases:

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Look-ahead Expansion"]
    S1["Input: current states (batch_size)"] --> S2["puzzle.batched_get_neighbours"]
    S2 --> S3["neighbour_look_a_head: [action_size, batch_size]\nncosts: [action_size, batch_size]"]
    S3 --> S4["look_a_head_costs = costs + ncosts\n(tentative g-values)"]
  end

  subgraph Phase2["Phase 2: Deduplicate & Prune"]
    S4 --> S5["flatten to [action_size * batch_size]"]
    S5 --> S6["hashtable.lookup_parallel\n(check if states exist)"]
    S6 --> S7["old_costs = search_result.get_cost(current_hash_idxs)"]
    S7 --> S8["candidate_mask = ~found | (next_g < old_g)"]
    S8 --> S9["optimal_mask = unique_mask & candidate_mask\n(best unique candidates)"]
  end

  subgraph Phase3["Phase 3: Pack & Compute Heuristics"]
    S9 --> S10["need_compute = optimal_mask & ~found\n(new states need h-value)"]
    S10 --> S11["stable_partition_three\n(pack need_compute=True to front)"]
    S11 --> S12["reshape to [action_size, batch_size] chunks"]
    S12 --> S13["jax.lax.scan over chunks"]
    S13 --> S14["_calc_heuristic_chunk:\nvariable_heuristic_batch_switcher"]
    S14 --> S15["heuristic_vals = where(found, old_dists, computed)"]
  end

  subgraph Phase4["Phase 4: Create & Insert Actions"]
    S15 --> S16["neighbour_keys = w*look_a_head_costs + h"]
    S16 --> S17["vals = Parant_with_Costs(\nparent=Parent(hashidx, action),\ncost=costs,\ndist=heuristic_vals)"]
    S17 --> S18["sort by key, keep best candidates"]
    S18 --> S19["jax.lax.scan: insert into PQ\n(conditional on optimal_unique_mask)"]
  end

  subgraph Phase5["Phase 5: Pop & Materialize Next Batch"]
    S19 --> S20["search_result.pop_full_with_actions"]
    S20 --> S21["Inside pop_full_with_actions:\n1. Pop Parant_with_Costs from PQ\n2. Expand (parent, action) -> child states\n3. Deduplicate expanded children\n4. Insert children into hashtable\n5. Update cost/dist/parent tables"]
    S21 --> S22["return new LoopStateWithStates\n(with materialized next states)"]
  end

  S4 --> S5
  S9 --> S10
  S15 --> S16
  S19 --> S20
```

**Key Implementation Details**:
- **Look-ahead simulation**: All actions are simulated *before* PQ insertion to filter non-optimal paths
- **Batch packing**: `stable_partition_three` groups states needing heuristic computation for efficient batching
- **Conditional heuristic**: Only new states (`~found`) compute h-values; existing states reuse cached values
- **Chunked scanning**: Heuristic computation is split into `action_size` chunks to respect `max_batch_size`
- **Deferred materialization**: Full states are only created in `pop_full_with_actions`, not during expansion

### Detailed Phase Breakdowns

#### Phase 1: Look-ahead Expansion

```mermaid
flowchart TD
  A1["Input:\ncurrent states: [batch_size]\ncost: [batch_size]\nfilled: [batch_size]"] --> A2["Tile for all actions"]
  A2 --> A3["idx_tiles: [action_size, batch_size]\naction: [action_size, batch_size] = 0..action_size-1\ncosts: [action_size, batch_size]\nfilled_tiles: [action_size, batch_size]"]
  A3 --> A4["puzzle.batched_get_neighbours(\nsolve_config, states, filled)"]
  A4 --> A5["neighbour_look_a_head: [action_size, batch_size]\nncosts: [action_size, batch_size]"]
  A5 --> A6["look_a_head_costs = costs + ncosts\n(tentative g-values for each action)"]
```

#### Phase 2: Deduplicate & Prune

```mermaid
flowchart TD
  B1["Flatten to [flat_size = action_size * batch_size]:\nflattened_neighbour_look_head\nflattened_look_a_head_costs\nflattened_filled_tiles"] --> B2["hashtable.lookup_parallel(\nflattened_neighbour_look_head,\nflattened_filled_tiles)"]
  B2 --> B3["Returns:\ncurrent_hash_idxs: [flat_size]\nfound: [flat_size] (bool)"]
  B3 --> B4["old_costs = search_result.get_cost(\ncurrent_hash_idxs)"]
  B4 --> B5["candidate_mask =\n~found | (next_g < old_g)"]
  B5 --> B6["optimal_mask = unique_mask(\nflattened_neighbour_look_head,\nflattened_look_a_head_costs,\ncandidate_mask) & candidate_mask"]
  B6 --> B7["Reshape back to [action_size, batch_size]:\nfound_reshaped\noptimal_mask_reshaped"]
```

#### Phase 3: Pack & Compute Heuristics

```mermaid
flowchart TD
  C1["old_dists = search_result.get_dist(\ncurrent_hash_idxs).reshape(\naction_size, batch_size)"] --> C2["need_compute =\noptimal_mask_reshaped & ~found_reshaped"]
  C2 --> C3["Flatten:\nflat_states = neighbour_look_a_head.flatten()\nflat_need_compute = need_compute.flatten()"]
  C3 --> C4["stable_partition_three(\nflat_need_compute, zeros)"]
  C4 --> C5["sorted_indices: [flat_size]\n(need_compute=True entries first)"]
  C5 --> C6["sorted_states = flat_states[sorted_indices]\nsorted_mask = flat_need_compute[sorted_indices]"]
  C6 --> C7["Reshape to chunks:\nsorted_states_chunked: [action_size, batch_size]\nsorted_mask_chunked: [action_size, batch_size]"]
  C7 --> C8["jax.lax.scan(_calc_heuristic_chunk,\nNone, (sorted_states_chunked, sorted_mask_chunked))"]

  subgraph Scan["_calc_heuristic_chunk (per chunk)"]
    S1["states_slice: [batch_size]\ncompute_mask: [batch_size]"] --> S2["variable_heuristic_batch_switcher(\nheuristic_parameters,\nstates_slice,\ncompute_mask)"]
    S2 --> S3["h_val: [batch_size]"]
  end

  C8 --> Scan
  Scan --> C9["h_val_chunks: [action_size, batch_size]"]
  C9 --> C10["Flatten & unsort:\nh_val_sorted = h_val_chunks.reshape(-1)\nflat_h_val[sorted_indices] = h_val_sorted"]
  C10 --> C11["computed_heuristic_vals =\nflat_h_val.reshape(action_size, batch_size)"]
  C11 --> C12["heuristic_vals = where(\nfound_reshaped,\nold_dists,\ncomputed_heuristic_vals)"]
```

#### Phase 4: Create & Insert Actions

```mermaid
flowchart TD
  D1["neighbour_keys =\n(w * look_a_head_costs + heuristic_vals)\n.astype(KEY_DTYPE)"] --> D2["vals = Parant_with_Costs(\nparent=Parent(\nhashidx=idx_tiles.flatten(),\naction=action.flatten()),\ncost=costs.flatten(),\ndist=heuristic_vals.flatten())"]
  D2 --> D3["flattened_vals = vals.flatten()\nflattened_keys = neighbour_keys.flatten()"]
  D3 --> D4["flattened_neighbour_keys =\nwhere(optimal_mask, flattened_keys, inf)"]
  D4 --> D5["jax.lax.sort_key_val(\nflattened_neighbour_keys,\narange(flat_size))"]
  D5 --> D6["sorted_key, sorted_idx\nsorted_vals = flattened_vals[sorted_idx]\nsorted_optimal_unique_mask =\noptimal_mask[sorted_idx]"]
  D6 --> D7["Reshape to [action_size, batch_size]:\nneighbour_keys\nvals\noptimal_unique_mask"]
  D7 --> D8["jax.lax.scan(_scan, search_result,\n(neighbour_keys, vals, optimal_unique_mask))"]

  subgraph Scan["_scan (per action row)"]
    S1["neighbour_keys: [batch_size]\nvals: [batch_size]\nmask: [batch_size]"] --> S2{"any(mask)?"}
    S2 -- yes --> S3["_insert:\nsearch_result.priority_queue.insert(\nneighbour_keys, vals)"]
    S2 -- no --> S4["Skip (no-op)"]
    S3 --> S5["return updated search_result"]
    S4 --> S5
  end

  D8 --> Scan
  Scan --> D9["search_result (with actions inserted)"]
```

#### Phase 5: Pop & Materialize Next Batch

```mermaid
flowchart TD
  E1["search_result.pop_full_with_actions(\npuzzle, solve_config,\nuse_heuristic=True)"] --> E2["Calls:\n_pop_full_with_parent_with_costs"]

  subgraph PopFull["_pop_full_with_parent_with_costs (Eager Expansion)"]
    P1["PQ.delete_mins() ->\nmin_key, min_val (Parant_with_Costs)"] --> P2["_expand_and_filter"]

    subgraph Expand["_expand_and_filter"]
      EX1["Extract:\nparent_states = get_state(val.parent)\nparent_actions = val.parent.action\nparent_costs = get_cost(val.parent)"] --> EX2["puzzle.batched_get_actions(\nsolve_config,\nparent_states,\nparent_actions,\nfilled)"]
      EX2 --> EX3["current_states, ncosts"]
      EX3 --> EX4["current_costs = parent_costs + ncosts\ncurrent_dists = val.dist\n(if use_heuristic)"]
      EX4 --> EX5["unique_mask = unique_mask(\ncurrent_states, current_costs, filled)"]
      EX5 --> EX6["hashtable.lookup_parallel(\ncurrent_states, unique_mask)"]
      EX6 --> EX7["current_hash_idxs, found"]
      EX7 --> EX8["old_costs = get_cost(current_hash_idxs)"]
      EX8 --> EX9["better_cost_mask = current_costs < old_costs\noptimal_mask = unique_mask &\n(~found | better_cost_mask)"]
      EX9 --> EX10["filtered_key = where(optimal_mask, key, inf)"]
    end

    P2 --> P3["current_states, current_costs,\ncurrent_dists, min_key"]
    P3 --> P4["Apply unique_mask to batch\n(maintain invariant)"]
    P4 --> P5["while loop (fill batch to ~99%)"]

    subgraph Loop["Loop Body"]
      L1["PQ.delete_mins() -> new_key, new_val"] --> L2["_expand_and_filter(new_key, new_val)"]
      L2 --> L3["Merge:\nstack_states = concat(current, new)\nstack_costs = concat(current, new)\nstack_dists = concat(current, new)\nstack_key = concat(current, new)\nstack_val = concat(current, new)"]
      L3 --> L4["Deduplicate:\nunique_mask = unique_mask(\nstack_states, stack_costs, stack_filled)"]
      L4 --> L5["stack_key = where(unique_mask, stack_key, inf)"]
      L5 --> L6["Sort and split:\nsorted_key, sorted_idx =\nsort_key_val(stack_key, arange(stack_size))"]
      L6 --> L7["Split:\nmain = sorted[:batch_size]\noverflow = sorted[batch_size:]"]
    end

    P5 --> Loop
    Loop --> P6["Re-insert overflow into PQ"]
    P6 --> P7["Apply pop_ratio threshold:\nthreshold = min_key[0] * pop_ratio + eps\nprocess_mask = (key <= threshold) |\n(cumsum(filled) <= min_pop)"]
    P7 --> P8["Re-insert non-processed into PQ:\nreturn_keys = where(process_mask, inf, min_key)"]
    P8 --> P9["hashtable.parallel_insert(\nfinal_states, final_process_mask)"]
    P9 --> P10["Update tables:\ncost[hash_idx.index] = final_costs\ndist[hash_idx.index] = final_dists\nparent[hash_idx.index] = final_parents"]
    P10 --> P11["return:\nsearch_result,\nfinal_currents (Current),\nfinal_states,\nfinal_process_mask"]
  end

  E2 --> PopFull
  PopFull --> E3["return new LoopStateWithStates(\nsearch_result,\nsolve_config,\nparams,\ncurrent=min_val,\nstates=next_states,\nfilled=filled)"]
```

### pop_full_with_actions: Eager Expansion & Deduplication

Unlike standard A*, the `pop_full_with_actions` method is **critical** to the deferred variant. It performs eager expansion *inside* the pop operation to ensure batch quality:

**Key Steps**:
1. **Pop actions from PQ**: Retrieve `Parant_with_Costs` (parent + action + costs)
2. **Expand to states**: Use `puzzle.batched_get_actions(parent_states, parent_actions)` to materialize child states
3. **Deduplicate**: Apply `unique_mask` and lookup in hashtable to filter non-optimal/duplicate children
4. **Fill batch**: Loop to accumulate enough unique, optimal children (respecting `pop_ratio`)
5. **Insert into hashtable**: Add new states and update cost/dist/parent tables
6. **Return materialized states**: Unlike standard A*, return the actual states (not just hash indices)

**Why Eager Expansion?**

In highly reversible environments (e.g., Rubik's Cube), popping actions can yield many duplicate children (e.g., undoing the previous move). Without eager expansion, the batch would be "starved" — filled with duplicates that provide no useful work.

By expanding and deduplicating *inside* the pop, we guarantee the returned batch contains only unique, optimal states ready for the next iteration.

### Mask Pipeline (Look-ahead Pruning)

The mask pipeline in A* Deferred with look-ahead pruning is more complex than standard A*, as it filters candidates at multiple stages:

```mermaid
flowchart LR
  C0["candidate edges\n(actions from current states)\n[action_size, batch_size]"] --> M1["filled_tiles mask\n(valid entries)"]
  M1 --> M2["Flatten to [flat_size]"]
  M2 --> M3["hashtable.lookup_parallel"]
  M3 --> M3a["found mask\n(state exists in HT)"]
  M3 --> M3b["old_costs\n(stored g-values)"]

  M3a --> M4["candidate_mask =\n~found | (next_g < old_g)"]
  M3b --> M4
  M2 --> M4

  M4 --> M5["optimal_mask =\nunique_mask(...) & candidate_mask"]

  M5 --> M6["need_compute mask =\noptimal_mask & ~found\n(new states need h-value)"]

  M6 --> H1["Trigger heuristic computation\nfor need_compute=True entries"]

  M5 --> M7["Reshape to [action_size, batch_size]:\noptimal_mask_reshaped"]

  M7 --> M8["Sort by key, keep best"]

  M8 --> M9["final optimal_unique_mask\n(after sorting)"]

  M9 --> Q1["Conditional PQ insert\n(only if any(mask) per action row)"]

  M5 --> U1["Used to filter keys\n(where(optimal_mask, key, inf))"]
```

**Mask Semantics**:
- **filled_tiles**: Valid batch entries (finite g-values)
- **found**: State already exists in hashtable (may need cost update)
- **candidate_mask**: States that are either new OR offer better cost
- **optimal_mask**: Best unique candidates (no duplicates within batch)
- **need_compute**: New states requiring heuristic evaluation
- **optimal_unique_mask**: Final mask after sorting (controls PQ insertion)

### Standard A* vs A* Deferred: Data Flow Comparison

```mermaid
flowchart TB
  subgraph Standard["Standard A* (astar.py)"]
    A1["Pop states from PQ"] --> A2["Fetch states from HT"]
    A2 --> A3["batched_get_neighbours\n(expand all actions)"]
    A3 --> A4["Compute h for all children\n(batch heuristic call)"]
    A4 --> A5["parallel_insert into HT\n(deduplicate)"]
    A5 --> A6["Update cost/parent tables"]
    A6 --> A7["Insert states into PQ\nwith f = w*g + h"]
  end

  subgraph Deferred["A* Deferred (astar_d.py) with Look-ahead"]
    D1["Pop actions from PQ\n(Parant_with_Costs)"] --> D2["Expand (parent, action) -> children\nINSIDE pop_full_with_actions"]
    D2 --> D3["Deduplicate expanded children\n(eager expansion)"]
    D3 --> D4["Insert children into HT"]
    D4 --> D5["Update cost/parent tables"]
    D5 --> D6["Return materialized states\n(ready for next iteration)"]
    D6 --> D7["batched_get_neighbours\n(look-ahead simulation)"]
    D7 --> D8["lookup_parallel in HT\n(check if children exist)"]
    D8 --> D9["Compute h only for new states\n(conditional heuristic)"]
    D9 --> D10["Insert actions into PQ\nwith f = w*g + h"]
  end

  subgraph Key["Key Differences"]
    K1["PQ Contents: States (Standard) vs Actions (Deferred)"]
    K2["Expansion: Immediate (Standard) vs Deferred (during pop)"]
    K3["Heuristic: All children (Standard) vs New states only (Deferred)"]
    K4["Deduplication: After expansion (Standard) vs During pop (Deferred)"]
  end

  Standard -.-> Key
  Deferred -.-> Key
```

### JIT Compilation Strategy

Like standard A*, `astar_d_builder(...)` returns a JIT-compiled function (`astar_d_fn = jax.jit(astar_d)`). The compilation strategy is identical:

- **Warm-up compilation**: First call uses `puzzle.SolveConfig.default()` and `puzzle.State.default()` to trigger XLA compilation with simple inputs
- **Cache reuse**: Subsequent calls reuse the compiled program (same shapes/dtypes/static args)
- **Why empty inputs?**: Real puzzles cause extremely long compilation times due to complex traced logic; empty inputs allow fast specialization

**Compilation Time Notes**:
- A* Deferred typically takes **longer to compile** than standard A* due to:
  - More complex loop body (look-ahead simulation + packing)
  - Nested scans (chunked heuristic computation + PQ insertion)
  - Eager expansion logic in `pop_full_with_actions` (expand-deduplicate loop)
- Use `--show_compile_time` flag to measure compilation overhead
