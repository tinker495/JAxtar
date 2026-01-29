# A\* Command

The `astar` command solves a puzzle using the A\* search algorithm. A\* is a classic graph traversal and path-finding algorithm, which is often used in many fields of computer science due to its completeness, optimality, and optimal efficiency. This implementation is fully JIT-compiled with JAX for high performance on accelerators.

## Usage

The basic syntax for the `astar` command is:

```bash
python main.py astar [OPTIONS]
```

A common use case is to solve a specific puzzle with a neural network heuristic:

```bash
python main.py astar -p rubikscube -nn
```

## Options

The `astar` command uses a combination of option groups to configure the puzzle, search algorithm, heuristic, and visualization.

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment to be solved.

-   `-p, --puzzle`: Specifies the puzzle to solve.
    -   Type: `Choice`
    -   Default: `n-puzzle`
    -   Choices: `n-puzzle`, `rubikscube`, `slidepuzzle`, etc. (depends on configuration).
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
    -   Example: `python main.py astar -pargs '{"size": 4}'`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A comma-separated list of seeds for generating initial puzzle states. Using multiple seeds will run the solver multiple times.
    -   Type: `String`
    -   Default: `"0"`

### Search Options (`@search_options`)

These options control the behavior of the A\* search algorithm itself.

-   `-m, --max_node_size`: The maximum number of nodes to explore. Supports scientific notation.
    -   Type: `String`
-   `-b, --batch_size`: The number of nodes to process in a single batch on the GPU.
    -   Type: `Integer`
-   `-w, --cost_weight`: The weight `w` for the path cost in `f(n) = w * g(n) + h(n)`. A value of `1.0` is standard A\*, while a lower value (< 1.0) prioritizes nodes closer to the goal (greedy search), and a higher value (> 1.0) prioritizes exploring cheaper paths.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio for popping nodes from the priority queue.
    -   Type: `Float`
-   `-vm, --vmap_size`: The number of different initial states to solve in parallel using `jax.vmap`.
    -   Type: `Integer`
-   `--debug`: Disables JIT compilation for easier debugging.
    -   Type: `Flag`
-   `--profile`: Enables `jax.profiler` and saves a trace to `tmp/tensorboard`.
    -   Type: `Flag`
-   `--show_compile_time`: Prints the JIT compilation time.
    -   Type: `Flag`

### Heuristic Options (`@heuristic_options`)

These options determine which heuristic function to use for guiding the search.

-   `-nn, --neural_heuristic`: Use a pre-trained neural network as the heuristic function. If not set, a default, non-ML heuristic is used.
    -   Type: `Flag`
-   `--param-path`: Path to the heuristic parameter file.
    -   Type: `String`
-   `--model-type`: Type of the heuristic model.
    -   Type: `String`

### Visualization Options (`@visualize_options`)

These options control how the final solution path is displayed.

-   `-vt, --visualize_terminal`: Renders the solution path step-by-step in the terminal.
    -   Type: `Flag`
-   `-vi, --visualize_imgs`: Generates an image for each step of the solution and saves them, along with a GIF animation, in a timestamped folder within `tmp/`.
    -   Type: `Flag`
-   `-mt, --max_animation_time`: Sets the maximum duration for the generated GIF animation, in seconds. The frame rate is adjusted to fit this duration.
    -   Type: `Integer`
    -   Default: `10`

## Implementation Notes (JAxtar/stars/astar.py)

This section documents the actual control flow and data flow in `JAxtar/stars/astar.py`.
The implementation is a batched, JIT-compiled A* variant built around two JAX-native data
structures inside `SearchResult`:

- `hashtable`: state deduplication + index assignment (stable IDs for states)
- `priority_queue`: frontier ordering by `key = cost_weight * g + h`

The core loop is built by `_astar_loop_builder(...)` and executed by `jax.lax.while_loop`.

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (astar_builder)"]
    B1["_astar_loop_builder"] --> B2["create init, cond, body functions"]
    B2 --> B3["jax.jit(astar)"]
    B3 --> B4["Warm-up: run with default config/state"]
    B4 --> B5["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (astar_fn)"]
    R1["call compiled astar_fn"] --> R2["init_loop_state"]
    R2 --> R3["jax.lax.while_loop"]
    R3 --> R4["final solved check & result index extraction"]
    R4 --> R5["return SearchResult"]
  end

  B5 --> R1

  subgraph Init["init_loop_state"]
    I1["SearchResult.build (allocate tables)"] --> I2["prepare_heuristic_parameters"]
    I2 --> I3["insert start into hashtable"]
    I3 --> I4["set initial cost & create LoopState"]
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

```mermaid
flowchart LR
  SR["SearchResult"] --> HT["hashtable\n(state to index mapping)"]
  SR --> PQ["priority_queue\n(key to Current)"]
  SR --> COST["cost table\n(g values)"]
  SR --> DIST["dist table\n(h values cache)"]
  SR --> PARENT["parent table\n(Parent struct: hashidx, action)"]

  LS["LoopState"] --> SR
  LS --> CUR["current\n(hashidx and g for expansion)"]
  LS --> FILLED["filled mask\n(active batch entries)"]
  LS --> PARAMS["heuristic parameters"]
```

### Loop Body Data Flow (One Iteration)

The loop body expands the current batch, deduplicates candidate next states in parallel,
updates best-known `g` costs and parent pointers, (re-)inserts improved candidates into the
priority queue using `f = cost_weight * g + h`, then pops the next batch.

Key implementation details from `JAxtar/stars/astar.py`:

- Neighbour generation is fully batched: `puzzle.batched_get_neighbours(solve_config, states, filled)`
- Parent tracking is vectorized (`Parent(hashidx=..., action=...)`) and stored per state index
- Hash table insertion is parallel (`hashtable.parallel_insert`) and returns multiple masks
  used to filter out duplicates and non-optimal paths
- Heuristic is cached in `search_result.dist`; newly inserted states compute `h` via
  `variable_batch_switcher_builder(heuristic.batched_distance, ...)`
- `stable_partition_three(...)` reorders flattened candidates to group useful work early
  (improves efficiency by creating denser batches for subsequent operations)

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Expand"]
    S1["fetch current states"] --> S2["batched_get_neighbours"]
    S2 --> S3["compute next costs & gather parents"]
  end

  subgraph Phase2["Phase 2: Deduplicate & Relax"]
    S4["parallel_insert into hashtable"] --> S5["compute optimal_mask (next_g < stored_g)"]
    S5 --> S6["final_process_mask (unique & optimal)"]
    S6 --> S7["update_on_condition (cost and parent tables)"]
  end

  subgraph Phase3["Phase 3: Group Useful Work"]
    S8["stable_partition_three (new_states_mask, final_process_mask)"] --> S9["reorder candidates to front of batch"]
    S9 --> S10["reshape into action-major rows"]
  end

  subgraph Phase4["Phase 4: Frontier Update (Scan)"]
    S10 --> S11["jax.lax.scan over action rows"]
    S11 --> S12["_scan: Processing one action row"]
  end

  subgraph Phase5["Phase 5: Advance"]
    S12 --> S13["search_result.pop_full()"]
    S13 --> S14["return new LoopState"]
  end

  S3 --> S4
  S7 --> S8

  subgraph Scan["_scan: Processing one action row"]
    A0["Row inputs: vals, neighbours,<br/>new_states_mask, final_process_mask"] --> A1{"any(new_states_mask)?"}
    A1 -- yes --> A2["_new_states: Compute h (NN/heuristic)\n& cache h in search_result.dist"]
    A1 -- no --> A3["_old_states: Reuse cached h\nfrom search_result.dist"]

    A2 --> A4{"any(final_process_mask)?"}
    A3 --> A4

    A4 -- yes --> A5["_inserted: Calculate key (w*g + h)\n& insert into priority_queue"]
    A4 -- no --> A6["Skip insertion"]

    A5 --> A7["Return updated SearchResult"]
    A6 --> A7
  end
```

### Mask Pipeline

```mermaid
flowchart LR
  C0["candidate edges\n(neighbours, next_g)"] --> M1["filleds mask\n(finite g)"]
  M1 --> M2["parallel_insert results"]
  M2 --> M2a["new_states_mask\n(first time seen)"]
  M2 --> M2b["cheapest_uniques_mask\n(best within current batch)"]

  M2b --> M3["optimal_mask\n(next_g < previously stored_g)"]
  M3 --> M4["final_process_mask\n(unique & optimal)"]

  M4 --> U1["update cost & parent tables"]
  M4 --> Q1["eligible for PQ insert"]

  M2a --> H1["trigger heuristic computation"]
```

### JIT Compilation Strategy

`astar_builder(...)` returns a JIT-compiled function (`astar_fn = jax.jit(astar)`).
To avoid extremely long compilation times from tracing complex puzzle logic on real inputs,
it triggers compilation once using `puzzle.SolveConfig.default()` and `puzzle.State.default()`.

This means:

- First call compiles and caches the XLA program.
- Subsequent calls reuse the compiled program as long as shapes/dtypes/static args match.
