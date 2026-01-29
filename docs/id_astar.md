# Iterative Deepening A\* Command (`id_astar`)

The `id_astar` command solves a puzzle using the Iterative Deepening A\* (IDA\*) search algorithm. IDA\* combines the space efficiency of depth-first search with the optimality of A\*. It performs a series of depth-first searches with increasing cost thresholds.

## Usage

The basic syntax for the `id_astar` command is:

```bash
python main.py id_astar [OPTIONS]
```

Example:

```bash
python main.py id_astar -p rubikscube -nn
```

## Options

The `id_astar` command uses the same option groups as `astar`.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore (per iteration).
-   `-b, --batch_size`: Batch size for GPU processing.
-   `-w, --cost_weight`: Path cost weight.
-   `-pr, --pop_ratio`: Pop ratio.
-   `-vm, --vmap_size`: Parallel solve size via vmap.
-   `--debug`: Disable JIT.
-   `--profile`: Enable profiling.
-   `--show_compile_time`: Print compile time.

### Heuristic Options (`@heuristic_options`)

-   `-nn, --neural_heuristic`: Use neural network heuristic.
-   `--param-path`: Path to heuristic parameters.
-   `--model-type`: Heuristic model type.

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Render path in terminal.
-   `-vi, --visualize_imgs`: Generate images/GIF.
-   `-mt, --max_animation_time`: Max GIF duration.

---

## Implementation Notes (JAxtar/id_stars/id_astar.py)

This section documents the actual control flow and data flow in `JAxtar/id_stars/id_astar.py`.

Iterative Deepening A* (IDA*) is an optimal search algorithm that performs a sequence of depth-first searches with increasing cost thresholds (bounds). JAxtar implements a **Batched IDA*** that leverages JAX to parallelize the depth-first exploration while maintaining memory efficiency.

The core distinction from standard A* is the dual-loop structure:
- Standard A* uses a priority queue with one while loop
- IDA* uses a stack with two nested while loops (outer for bound management, inner for DFS)

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (id_astar_builder)"]
    B1["_id_astar_loop_builder"] --> B2["create init_loop_state, inner_cond, inner_body"]
    B2 --> B3["build_outer_loop (outer_cond, outer_body)"]
    B3 --> B4["jax.jit(search_fn)"]
    B4 --> B5["Warm-up: run with default config/state"]
    B5 --> B6["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (id_astar_fn)"]
    R1["call compiled id_astar_fn"] --> R2["init_loop_state"]
    R2 --> R3["jax.lax.while_loop (OUTER)"]
    R3 --> R4["final solved check"]
    R4 --> R5["return IDSearchBase"]
  end

  B6 --> R1

  subgraph Init["init_loop_state"]
    I1["IDSearchBase.build (allocate stack)"] --> I2["generate_frontier (Parallel BFS)"]
    I2 --> I3["compute frontier f-scores"]
    I3 --> I4["set initial bound = min(f)"]
    I4 --> I5["push frontier to stack"]
    I5 --> I6["create IDLoopState"]
  end

  subgraph Outer["outer loop (bound management)"]
    O1["outer_condition"] --> O2{"Not Solved & Finite Bound?"}
    O2 -- yes --> O3["inner_loop (DFS)"]
    O3 --> O4["update bound = next_bound"]
    O4 --> O5["reset stack & push frontier"]
    O5 --> O1
    O2 -- no --> O6["exit"]
  end

  subgraph Inner["inner loop (DFS expansion)"]
    IN1["inner_condition"] --> IN2{"Stack Not Empty & Not Solved?"}
    IN2 -- yes --> IN3["inner_body"]
    IN3 --> IN1
    IN2 -- no --> IN4["exit to outer loop"]
  end

  R2 --> I1
  R3 --> O1
  O3 --> IN1
```

### Data Structures At A Glance

The IDA* implementation uses a fundamentally different data structure architecture compared to A*:

```mermaid
flowchart LR
  SR["IDSearchBase"] --> STACK["stack (LIFO)\n(IDStackItem entries)"]
  SR --> BOUND["bound\n(current f-limit)"]
  SR --> NEXT["next_bound\n(min f > bound)"]
  SR --> TRACE_P["trace_parent\n(reconstruction)"]
  SR --> TRACE_A["trace_action\n(reconstruction)"]
  SR --> TRACE_R["trace_root\n(frontier tracking)"]
  SR --> SOL["solution info\n(state, cost, actions)"]

  STACK --> ITEM["IDStackItem"]
  ITEM --> STATE["state"]
  ITEM --> COST["cost (g)"]
  ITEM --> DEPTH["depth"]
  ITEM --> TRAIL["trail\n(non-backtracking)"]
  ITEM --> ACT_HIST["action_history"]
  ITEM --> PARENT["parent_index"]
  ITEM --> ROOT["root_index"]
  ITEM --> TRACE["trace_index"]

  LS["IDLoopState"] --> SR
  LS --> CFG["solve_config"]
  LS --> PARAMS["heuristic parameters"]
  LS --> FRONT["frontier\n(for outer loop resets)"]

  FR["IDFrontier"] --> FR_STATES["states"]
  FR --> FR_COSTS["costs"]
  FR --> FR_DEPTHS["depths"]
  FR --> FR_VALID["valid_mask"]
  FR --> FR_F["f_scores"]
  FR --> FR_TRAIL["trail"]
  FR --> FR_SOL["solution info"]
```

#### Key Differences from A*

**A* (SearchResult)**:
- hashtable: Maps states to indices for O(1) deduplication across all iterations
- priority_queue: Orders frontier by f-score
- cost/dist/parent tables: Indexed by hashtable indices

**IDA* (IDSearchBase)**:
- stack: LIFO structure for DFS ordering (no hashtable needed per iteration)
- bound/next_bound: Cost thresholds for pruning
- trace arrays: Parent/action tracking for path reconstruction
- In-batch deduplication only (no global state tracking)

### Data Structure Details

#### `IDFrontier`

A specialized container for the initial frontier generation and outer loop resets.

```mermaid
classDiagram
  class IDFrontier {
    +states: Puzzle.State [batch_size]
    +costs: Array [batch_size]
    +depths: Array [batch_size]
    +valid_mask: Array [batch_size]
    +f_scores: Array [batch_size]
    +trail: Puzzle.State [batch_size, trail_len]
    +action_history: Array [batch_size, max_path_len]
    +solved: bool scalar
    +solution_state: Puzzle.State [1]
    +solution_cost: scalar
    +solution_actions_arr: Array [max_path_len]

    +initialize_from_start() IDFrontier
    +select_top_k() IDFrontier
  }

  note for IDFrontier "Generated by parallel BFS expansion\nStores top-k nodes by f-score\nUsed for outer loop initialization"
```

**Purpose**: The frontier is computed once at the start via batched BFS (`generate_frontier`), then cached in `IDLoopState.frontier`. On each outer loop iteration, the frontier is filtered by the new bound and pushed back onto the stack.

**Trail mechanism**: Each state maintains a fixed-length history of recent parent states. This prevents immediate backtracking without needing a global hashtable.

#### `IDNodeBatch`

A batched representation of candidate nodes during expansion.

```mermaid
classDiagram
  class IDNodeBatch {
    +state: Puzzle.State [flat_size]
    +cost: Array [flat_size] (g-value)
    +depth: Array [flat_size]
    +trail: Puzzle.State [flat_size, trail_len]
    +action_history: Array [flat_size, max_path_len]
    +action: Array [flat_size]
    +parent_index: Array [flat_size]
    +root_index: Array [flat_size]
  }

  note for IDNodeBatch "Temporary structure for expansion\nflat_size = action_size × batch_size\nFiltered by bound before push"
```

**Layout**: Generated by flattening `(action_size, batch_size)` shaped neighbour arrays into `(flat_size,)` for vectorized processing.

#### `IDStackItem`

The actual data stored in the stack.

```mermaid
classDiagram
  class IDStackItem {
    +state: Puzzle.State
    +cost: scalar (g-value)
    +depth: scalar
    +action: scalar
    +parent_index: scalar
    +root_index: scalar
    +trace_index: scalar
    +trail: Puzzle.State [trail_len]
    +action_history: Array [max_path_len]
  }

  note for IDStackItem "Single node entry in stack\ntrace_index: position in trace arrays\nroot_index: link to original frontier"
```

**Trace system**: `trace_index` maps to positions in `trace_parent`, `trace_action`, `trace_root` arrays. This allows efficient path reconstruction when a solution is found.

### Dual-Loop Structure (Detailed)

IDA* uses two nested `jax.lax.while_loop` calls to achieve iterative deepening while remaining JIT-compatible.

#### Outer Loop: Bound Management

```mermaid
flowchart TD
  subgraph OuterLoop["Outer Loop (bound progression)"]
    OC1["Check: ~solved & bound < inf"] --> OC2{"Continue?"}
    OC2 -- no --> OE["Exit: Return solution or failure"]
    OC2 -- yes --> OB1["Execute inner_loop (DFS until bound)"]

    OB1 --> OB2["Extract next_bound from inner result"]
    OB2 --> OB3["Reset search_result:\n- bound ← next_bound\n- next_bound ← inf\n- stack.size ← 0\n- trace_size ← 0"]
    OB3 --> OB4["Re-push frontier filtered by new bound"]
    OB4 --> OB5["Update loop_state"]
    OB5 --> OC1
  end

  style OC2 fill:#e1f5ff
  style OB3 fill:#fff4e1
  style OB4 fill:#e8f5e9
```

**Key insight**: The outer loop doesn't expand nodes itself. It manages the bound threshold and resets the stack for each iteration. The frontier (computed once at init) is re-used across iterations.

#### Inner Loop: Parallel DFS Expansion

```mermaid
flowchart TD
  subgraph InnerLoop["Inner Loop (bounded DFS)"]
    IC1["Check: stack_ptr > 0 & ~solved"] --> IC2{"Continue?"}
    IC2 -- no --> IE["Exit: Return to outer loop"]
    IC2 -- yes --> IB1["Pop batch from stack"]

    IB1 --> IB2["Check for solution in popped batch"]
    IB2 --> IB3{"Found solution?"}
    IB3 -- yes --> IB4["Mark solved, skip expansion"]
    IB3 -- no --> IB5["Get neighbours (batched_get_neighbours)"]

    IB5 --> IB6["Build flat children\n(action_size × batch_size)"]
    IB6 --> IB7["Apply deduplication:\n- unique_mask (in-batch)\n- non_backtracking (trail check)"]
    IB7 --> IB8["Compute heuristic (chunked)"]
    IB8 --> IB9["Compute f = w·g + h"]

    IB9 --> IB10{"f ≤ bound?"}
    IB10 -- yes --> IB11["Keep: Add to push candidates"]
    IB10 -- no --> IB12["Prune: Update next_bound\nnext_bound ← min(next_bound, f)"]

    IB11 --> IB13["Sort by f descending (LIFO order)"]
    IB12 --> IB13
    IB13 --> IB14["Push to stack"]

    IB4 --> IB15["Update loop_state"]
    IB14 --> IB15
    IB15 --> IC1
  end

  style IC2 fill:#e1f5ff
  style IB3 fill:#ffebee
  style IB10 fill:#fff4e1
  style IB14 fill:#e8f5e9
```

**Stack ordering**: Nodes are sorted by f-score in descending order before push. This means the stack operates as a LIFO structure where `pop()` retrieves the lowest f-score nodes first (best-first DFS).

### Loop Body Data Flow (Inner Loop Iteration)

The inner loop body performs one parallel DFS step: pop a batch, expand, prune by bound, and push back.

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Pop & Prepare"]
    S1["sr.get_top_batch(batch_size)"] --> S2["Extract:\n- parents (states)\n- parent_costs (g)\n- parent_depths\n- parent_trails\n- parent_action_histories\n- valid_mask\n- trace_indices"]
  end

  subgraph Phase2["Phase 2: Solution Check"]
    S3["detect_solution(parents)"] --> S4{"Any solved?"}
    S4 -- yes --> S5["mark_solved() & return early"]
    S4 -- no --> S6["Continue expansion"]
  end

  subgraph Phase3["Phase 3: Expand"]
    S7["puzzle.batched_get_neighbours"] --> S8["build_flat_children:\n- Flatten to (flat_size,)\n- Compute child g = parent_g + step_cost\n- Update trail (prepend parent)\n- Update action_history"]
  end

  subgraph Phase4["Phase 4: Deduplicate"]
    S9["unique_mask (in-batch)"] --> S10["apply_non_backtracking (trail check)"]
    S10 --> S11["Final flat_valid mask"]
  end

  subgraph Phase5["Phase 5: Evaluate Heuristic"]
    S12["_chunked_heuristic_eval:\n- Partition valid nodes\n- Reshape to (action_size, batch_size)\n- scan over actions\n- Evaluate h in chunks"] --> S13["Compute f = w·g + h"]
  end

  subgraph Phase6["Phase 6: Prune & Push"]
    S14["Filter by bound:\nkeep_mask = (f ≤ bound)"] --> S15["compact_by_valid(keep_mask)"]
    S15 --> S16["Sort by f descending"]
    S16 --> S17["Update next_bound:\nmin(f where f > bound)"]
    S17 --> S18["push_packed_batch to stack"]
  end

  S2 --> S3
  S6 --> S7
  S8 --> S9
  S11 --> S12
  S13 --> S14
  S18 --> RET["Return updated IDLoopState"]
```

### Bound Progression & Pruning

The key to IDA* optimality is how the bound increases across iterations.

```mermaid
flowchart LR
  subgraph Init["Initialization"]
    I1["Generate frontier (BFS)"] --> I2["Compute frontier f-scores"]
    I2 --> I3["bound ← min(f)"]
    I3 --> I4["next_bound ← inf"]
  end

  subgraph Iter1["Iteration 1 (bound = b₁)"]
    IT1_1["Inner loop expands nodes"] --> IT1_2{"Process node with f"}
    IT1_2 -- "f ≤ b₁" --> IT1_3["Push to stack"]
    IT1_2 -- "f > b₁" --> IT1_4["next_bound ← min(next_bound, f)"]
    IT1_3 --> IT1_5["Continue until stack empty"]
    IT1_4 --> IT1_5
    IT1_5 --> IT1_6["Outer loop: bound ← next_bound = b₂"]
  end

  subgraph Iter2["Iteration 2 (bound = b₂)"]
    IT2_1["Reset stack, re-push frontier"] --> IT2_2["Inner loop with new bound"]
    IT2_2 --> IT2_3["Process nodes: f ≤ b₂ allowed"]
    IT2_3 --> IT2_4["Update next_bound = b₃"]
  end

  subgraph IterN["Iteration N (bound = bₙ)"]
    ITN_1["..."] --> ITN_2["Eventually f_solution ≤ bound"]
    ITN_2 --> ITN_3["Solution found: Exit"]
  end

  I4 --> IT1_1
  IT1_6 --> IT2_1
  IT2_4 --> ITN_1

  style I3 fill:#e8f5e9
  style IT1_6 fill:#fff4e1
  style ITN_3 fill:#e1f5ff
```

**Bound sequence**: b₁ ≤ b₂ ≤ b₃ ≤ ... ≤ f*(solution)

Each bound is the minimum f-value that was pruned in the previous iteration. This ensures no better solution exists below each threshold.

### Optimality Guarantee

```mermaid
flowchart TD
  A["Admissible heuristic: h(n) ≤ h*(n)"] --> B["f(n) = g(n) + h(n)"]
  B --> C["Bound increases monotonically"]
  C --> D["First solution found has f = optimal cost"]

  E["Each iteration exhausts all f ≤ bound"] --> D
  F["Next bound = min(pruned f) in previous iteration"] --> C

  style A fill:#e8f5e9
  style D fill:#e1f5ff
```

IDA* maintains optimality because:
1. The bound increases in discrete steps (minimum pruned f-values)
2. Each iteration completely explores all nodes with f ≤ bound
3. The first solution found must have the optimal cost (no better f exists below that bound)

### Parallel DFS Implementation

Unlike traditional recursive IDA*, JAxtar uses batched parallel processing:

```mermaid
flowchart LR
  subgraph Traditional["Traditional IDA*"]
    T1["Recursive DFS"] --> T2["One node at a time"]
    T2 --> T3["CPU-friendly"]
    T3 --> T4["Low memory"]
  end

  subgraph Batched["JAxtar Batched IDA*"]
    B1["Stack-based DFS"] --> B2["Batch of nodes (e.g., 1024)"]
    B2 --> B3["GPU-friendly"]
    B3 --> B4["Parallel expansion"]
    B4 --> B5["Higher throughput"]
  end

  T4 -.similarity.-> B4

  style Traditional fill:#ffebee
  style Batched fill:#e8f5e9
```

**Batch size**: The `batch_size` parameter controls how many nodes are popped and expanded in parallel. Larger batches improve GPU utilization but may do redundant work if a solution is found mid-batch.

### Memory Efficiency

IDA* is memory-efficient compared to A*:

```mermaid
flowchart TD
  subgraph Memory["Memory Usage Comparison"]
    M1["A*: O(branching_factor^depth)"] --> M2["Stores all generated nodes\nin hashtable + priority queue"]

    M3["IDA*: O(depth × batch_size)"] --> M4["Only stores current search path\n+ one batch in stack"]
  end

  subgraph Trade["Trade-off"]
    T1["IDA* regenerates nodes"] --> T2["More computation"]
    T2 --> T3["But avoids memory explosion"]

    T4["A* expands once"] --> T5["Less computation"]
    T5 --> T6["But can run out of memory"]
  end

  M2 --> T4
  M4 --> T1

  style M3 fill:#e8f5e9
  style M1 fill:#fff4e1
```

**Stack capacity**: The `max_nodes` parameter limits the stack size. If exceeded, the search may fail to find a solution (though this is rare with appropriate settings).

### Non-Backtracking Optimization

The trail mechanism prevents immediate backtracking:

```mermaid
flowchart TD
  subgraph Trail["Trail Structure"]
    TR1["Each node stores trail:\n[parent, grandparent, ..., ancestor_k]"] --> TR2["Length = non_backtracking_steps"]
  end

  subgraph Check["Backtracking Check"]
    C1["Generate child state"] --> C2["Compare child to parent"]
    C2 --> C3{"Match?"}
    C3 -- yes --> C4["Block (set valid=False)"]
    C3 -- no --> C5["Check trail: child == trail[i]?"]
    C5 --> C6{"Match any?"}
    C6 -- yes --> C4
    C6 -- no --> C7["Allow (valid=True)"]
  end

  TR2 --> C1

  style C4 fill:#ffebee
  style C7 fill:#e8f5e9
```

**Parameter tuning**: `non_backtracking_steps=0` disables trail checking (faster but may explore redundant paths). Values of 1-3 typically provide good balance.

### JIT Compilation Strategy

The compilation process mirrors A* but must handle the nested loop structure:

```mermaid
flowchart TD
  subgraph Compile["Compilation Flow"]
    CO1["_id_astar_loop_builder"] --> CO2["Create init/cond/body closures"]
    CO2 --> CO3["build_outer_loop wraps inner_loop"]
    CO3 --> CO4["jax.jit(search_fn)"]
    CO4 --> CO5["Trace with default inputs:\n- puzzle.SolveConfig.default()\n- puzzle.State.default()"]
    CO5 --> CO6["XLA compile both loops"]
    CO6 --> CO7["Cache compiled program"]
  end

  subgraph Reuse["Subsequent Calls"]
    R1["Same shapes/dtypes"] --> R2["Reuse cached XLA"]
    R3["Different shapes"] --> R4["Recompile"]
  end

  CO7 --> R1

  style CO6 fill:#fff4e1
  style R2 fill:#e8f5e9
```

**Compilation time**: IDA* typically compiles faster than A* because there's no hashtable tracing. However, the nested loop structure adds some complexity.

### Chunked Heuristic Evaluation

To avoid memory issues with large flat batches, the heuristic is evaluated in action-sized chunks:

```mermaid
flowchart TD
  subgraph Chunked["Chunked Evaluation"]
    CH1["Flat candidates: (flat_size,)"] --> CH2["Partition valid to front"]
    CH2 --> CH3["Reshape: (action_size, batch_size)"]
    CH3 --> CH4["jax.lax.scan over actions"]
    CH4 --> CH5["Each iteration:\nEvaluate h for one action batch"]
    CH5 --> CH6["Flatten back to (flat_size,)"]
  end

  subgraph Benefit["Benefits"]
    B1["Smaller working set"] --> B2["Better GPU memory usage"]
    B2 --> B3["Can handle larger batch_size"]
  end

  CH3 --> B1

  style CH4 fill:#e8f5e9
```

**Why chunking?** Evaluating `flat_size = action_size × batch_size` states at once can exceed GPU memory. Chunking processes `batch_size` states at a time via scan.
