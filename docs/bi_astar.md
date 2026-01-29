# Bidirectional A\* Command (`bi_astar`)

The `bi_astar` command solves a puzzle using the Bidirectional A\* search algorithm. This algorithm runs two simultaneous A\* searches: one forward from the initial state and one backward from the goal state. It stops when the two searches meet, often significantly reducing the number of nodes explored compared to standard A\*.

## Usage

The basic syntax for the `bi_astar` command is:

```bash
python main.py bi_astar [OPTIONS]
```

Example:

```bash
python main.py bi_astar -p rubikscube -nn
```

## Options

The `bi_astar` command uses similar option groups to the standard `astar` command.

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
-   `-b, --batch_size`: The number of nodes to process in a single batch.
    -   Type: `Integer`
-   `-w, --cost_weight`: The weight `w` for the path cost.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio for popping nodes from the priority queue.
    -   Type: `Float`
-   `-vm, --vmap_size`: The number of different initial states to solve in parallel.
    -   Type: `Integer`
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

---

## Implementation Notes (JAxtar/bi_stars/bi_astar.py)

Bidirectional A* search explores the state space from both directions simultaneously, aiming to meet at a common state. This strategy typically reduces the number of explored nodes from $O(b^d)$ to $O(b^{d/2})$.

The implementation is built around two parallel SearchResult instances (forward and backward), each with their own hashtable and priority queue. The core loop is constructed by `_bi_astar_loop_builder(...)` and executed by `jax.lax.while_loop`.

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (bi_astar_builder)"]
    B1["build_bi_search_result<br/>(allocate forward & backward tables)"] --> B2["_bi_astar_loop_builder"]
    B2 --> B3["create init, cond, body functions"]
    B3 --> B4["jax.jit(bi_astar)"]
    B4 --> B5["Warm-up: run with default config/state"]
    B5 --> B6["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (bi_astar_fn)"]
    R1["call compiled bi_astar_fn"] --> R2["prepare heuristic params (fwd & bwd)"]
    R2 --> R3["hindsight_transform<br/>(backward solve config)"]
    R3 --> R4["init_loop_state<br/>(initialize both frontiers)"]
    R4 --> R5["jax.lax.while_loop"]
    R5 --> R6["mark solved if meeting found"]
    R6 --> R7["return BiDirectionalSearchResult"]
  end

  B6 --> R1

  subgraph Init["init_loop_state"]
    I1["initialize_bi_loop_common"] --> I2["insert start into forward HT"]
    I2 --> I3["insert goal into backward HT"]
    I3 --> I4["check if start == goal<br/>(immediate solution)"]
    I4 --> I5["create BiLoopState with both currents"]
  end

  subgraph While["while loop (lax.while_loop)"]
    W1["common_bi_loop_condition"] --> W2{"Has Work & Not Terminated?"}
    W2 -- yes --> W3["loop_body<br/>(expand both directions)"] --> W1
    W2 -- no --> W4["exit"]
  end

  R4 --> I1
  R5 --> W1
```

### Data Structures At A Glance

```mermaid
flowchart LR
  BiResult["BiDirectionalSearchResult"] --> FWD["forward: SearchResult"]
  BiResult --> BWD["backward: SearchResult"]
  BiResult --> MEET["meeting: MeetingPoint"]

  FWD --> FHT["hashtable\n(forward state to index)"]
  FWD --> FPQ["priority_queue\n(forward frontier)"]
  FWD --> FCOST["cost table\n(g_fwd values)"]
  FWD --> FDIST["dist table\n(h_fwd cache)"]
  FWD --> FPARENT["parent table\n(Parent: hashidx, action)"]

  BWD --> BHT["hashtable\n(backward state to index)"]
  BWD --> BPQ["priority_queue\n(backward frontier)"]
  BWD --> BCOST["cost table\n(g_bwd values)"]
  BWD --> BDIST["dist table\n(h_bwd cache)"]
  BWD --> BPARENT["parent table\n(Parent: hashidx, action)"]

  MEET --> MDATA["fwd_hashidx, bwd_hashidx<br/>fwd_cost, bwd_cost<br/>total_cost, found"]

  BiLoop["BiLoopState"] --> BiResult
  BiLoop --> CFWD["current_forward<br/>(hashidx and g)"]
  BiLoop --> CBWD["current_backward<br/>(hashidx and g)"]
  BiLoop --> FFWD["filled_forward mask"]
  BiLoop --> FBWD["filled_backward mask"]
  BiLoop --> PARAMS["heuristic parameters<br/>(forward & backward)"]
  BiLoop --> SC["solve_config,<br/>inverse_solveconfig"]
```

### Dual-Frontier Search Structure

The algorithm maintains two independent search trees:
1. **Forward Search**: Starts from the `initial_state` and searches toward the goal using `solve_config`
2. **Backward Search**: Starts from the `goal_state` and searches toward the initial state using `inverse_solveconfig` (created via `puzzle.hindsight_transform`)

Both directions use the same core expansion logic but with direction-specific configurations:
- Forward: `puzzle.batched_get_neighbours`
- Backward: `puzzle.batched_get_inverse_neighbours`

### Loop Condition (common_bi_loop_condition)

The search continues while:
1. **Has Work**: At least one direction has nodes to expand AND hashtable capacity
   - `(fwd_has_nodes AND fwd_not_full) OR (bwd_has_nodes AND bwd_not_full)`
2. **Not Terminated**: Based on termination mode:
   - `terminate_on_first_solution=True` (default): Continue until any meeting point found
   - `terminate_on_first_solution=False`: Continue until optimality proven via f-value bounds

```mermaid
flowchart TD
  Start["loop_condition called"] --> Check1["Check forward frontier<br/>(has nodes & not full)"]
  Check1 --> Check2["Check backward frontier<br/>(has nodes & not full)"]
  Check2 --> HasWork{"At least one<br/>direction active?"}

  HasWork -- no --> RetFalse["return False<br/>(terminate)"]
  HasWork -- yes --> TermMode{"terminate_on<br/>first_solution?"}

  TermMode -- yes --> CheckFound{"meeting.found?"}
  CheckFound -- yes --> RetFalse2["return False<br/>(terminate)"]
  CheckFound -- no --> RetTrue["return True<br/>(continue)"]

  TermMode -- no --> CalcF["get_min_f_value<br/>(both directions)"]
  CalcF --> CheckOpt{"meeting.found AND<br/>w*total_cost <= min(f_fwd, f_bwd)?"}
  CheckOpt -- yes --> RetFalse3["return False<br/>(optimal found)"]
  CheckOpt -- no --> RetTrue2["return True<br/>(continue)"]
```

### Loop Body Data Flow (One Iteration)

Each iteration expands batches from both directions (if available), checks for intersections with the opposite frontier, and updates the meeting point if a better path is found.

```mermaid
flowchart TD
  subgraph LoopBody["loop_body (one iteration)"]
    L1["Check forward capacity"] --> L2{"fwd_has_nodes<br/>AND fwd_not_full?"}
    L2 -- yes --> L3["_expand_direction(forward)"]
    L2 -- no --> L4["skip forward expansion"]

    L3 --> L5["Check backward capacity"]
    L4 --> L5

    L5 --> L6{"bwd_has_nodes<br/>AND bwd_not_full?"}
    L6 -- yes --> L7["_expand_direction(backward)"]
    L6 -- no --> L8["skip backward expansion"]

    L7 --> L9["create new BiLoopState"]
    L8 --> L9
    L9 --> L10["return updated loop state"]
  end

  subgraph ExpandDir["_expand_direction (per direction)"]
    direction expand_detail
      E1["Phase 1: Expand"] --> E2["Phase 2: Deduplicate & Relax"]
      E2 --> E3["Phase 3: Intersection Check"]
      E3 --> E4["Phase 4: Group Useful Work"]
      E4 --> E5["Phase 5: Frontier Update"]
      E5 --> E6["Phase 6: Pop Next Batch"]
  end

  L3 -.-> expand_detail
  L7 -.-> expand_detail
```

### Detailed Loop Body Data Flow (_expand_direction)

The expansion of one direction follows a 6-phase pipeline similar to unidirectional A*, with an added intersection check phase.

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Expand Current Batch"]
    P1S1["get_state(current)"] --> P1S2["batched_get_neighbours<br/>or batched_get_inverse_neighbours"]
    P1S2 --> P1S3["compute next costs & parent info"]
    P1S3 --> P1S4["flatten: [action_size × batch_size]"]
  end

  subgraph Phase2["Phase 2: Deduplicate & Relax"]
    P2S1["parallel_insert into this direction's HT"] --> P2S2["get masks:<br/>new_states_mask, cheapest_uniques_mask"]
    P2S2 --> P2S3["compute optimal_mask<br/>(next_g < stored_g)"]
    P2S3 --> P2S4["final_process_mask<br/>(unique & optimal)"]
    P2S4 --> P2S5["update cost & parent tables"]
  end

  subgraph Phase3["Phase 3: Intersection Check"]
    P3S1["lookup_parallel in opposite HT"] --> P3S2["found_mask:<br/>states in both frontiers"]
    P3S2 --> P3S3["get opposite g-values"]
    P3S3 --> P3S4["compute total_costs<br/>(this_g + opposite_g)"]
    P3S4 --> P3S5["update_meeting_point<br/>(if better path found)"]
  end

  subgraph Phase4["Phase 4: Group Useful Work"]
    P4S1["stable_partition_three<br/>(new_states, final_process)"] --> P4S2["reorder candidates<br/>(new states first)"]
    P4S2 --> P4S3["reshape into action-major rows"]
  end

  subgraph Phase5["Phase 5: Frontier Update (Scan)"]
    P5S1["jax.lax.scan over action rows"] --> P5S2["for each row: _scan"]
  end

  subgraph Phase6["Phase 6: Pop Next Batch"]
    P6S1["pop_full() from PQ"] --> P6S2["return updated bi_result,<br/>new_current, new_filled"]
  end

  P1S4 --> P2S1
  P2S5 --> P3S1
  P3S5 --> P4S1
  P4S3 --> P5S1
  P5S2 --> P6S1

  subgraph Scan["_scan: Processing one action row"]
    S1["Row inputs: vals, neighbours,<br/>new_states_mask, final_process_mask"] --> S2{"any(new_states_mask)?"}
    S2 -- yes --> S3["_new_states: Compute h<br/>(via heuristic.batched_distance)"]
    S2 -- no --> S4["_old_states: Reuse cached h<br/>(from search_result.dist)"]

    S3 --> S5["cache h in dist table"]
    S4 --> S6["retrieve h from dist table"]

    S5 --> S7{"any(final_process_mask)?"}
    S6 --> S7

    S7 -- yes --> S8["_inserted: Calculate key<br/>(w*g + h) & insert into PQ"]
    S7 -- no --> S9["Skip insertion"]

    S8 --> S10["Return updated SearchResult"]
    S9 --> S10
  end

  P5S2 -.-> S1
```

### Intersection Detection Mechanism

The intersection check is the key differentiator from unidirectional A*. After expanding states in one direction, we immediately check if those states exist in the opposite direction's hashtable.

```mermaid
flowchart LR
  subgraph Input["Input to check_intersection"]
    I1["expanded_states<br/>(just generated)"]
    I2["expanded_costs<br/>(g from this direction)"]
    I3["expanded_mask<br/>(valid states)"]
    I4["opposite_sr<br/>(opposite SearchResult)"]
  end

  subgraph Process["Intersection Detection"]
    P1["lookup_parallel in<br/>opposite.hashtable"] --> P2["get opposite_hashidx<br/>and found mask"]
    P2 --> P3["get_cost(opposite_hashidx)<br/>(g from opposite direction)"]
    P3 --> P4["total_costs =<br/>expanded_costs + opposite_costs"]
    P4 --> P5["found_mask =<br/>found AND expanded_mask"]
  end

  subgraph Output["Output"]
    O1["found_mask<br/>(states in both HTs)"]
    O2["opposite_hashidx<br/>(location in opposite HT)"]
    O3["opposite_costs<br/>(g_opposite values)"]
    O4["total_costs<br/>(potential path cost)"]
  end

  I1 --> P1
  I2 --> P4
  I3 --> P5
  I4 --> P1

  P5 --> O1
  P2 --> O2
  P3 --> O3
  P4 --> O4

  subgraph Update["update_meeting_point"]
    U1["Find best total_cost<br/>among found intersections"] --> U2{"better than<br/>current meeting.total_cost?"}
    U2 -- yes --> U3["Update MeetingPoint:<br/>fwd_hashidx, bwd_hashidx,<br/>fwd_cost, bwd_cost,<br/>total_cost, found=True"]
    U2 -- no --> U4["Keep current MeetingPoint<br/>(update found flag if any found)"]
  end

  O1 --> U1
  O4 --> U1
```

### Path Reconstruction (reconstruct_bidirectional_path)

Unlike unidirectional A*, bidirectional search reconstructs the path by tracing from the meeting point in both directions, then concatenating the paths.

```mermaid
flowchart TD
  Start["reconstruct_bidirectional_path"] --> Check{"meeting.found?"}
  Check -- no --> Empty["return []"]

  Check -- yes --> Fwd["Forward Half: Trace from start"]

  subgraph ForwardTrace["Forward Path Tracing"]
    F1["Start at meeting.fwd_hashidx"] --> F2["Follow parent pointers backward"]
    F2 --> F3["_trace_root_to_target<br/>(indices, actions)"]
    F3 --> F4["Reverse to get start→meeting"]
    F4 --> F5["Extract states from forward HT"]
  end

  Fwd --> ForwardTrace

  ForwardTrace --> Bwd["Backward Half: Trace from meeting"]

  subgraph BackwardTrace["Backward Path Tracing"]
    B1["Start at meeting.bwd_hashidx"] --> B2["Follow parent pointers backward"]
    B2 --> B3["_trace_target_to_root<br/>(indices, actions)"]
    B3 --> B4["Already in meeting→goal order"]
    B4 --> B5["Extract states from backward HT"]
  end

  Bwd --> BackwardTrace

  BackwardTrace --> Merge["Concatenate Paths"]

  subgraph Concatenation["Path Merging"]
    M1["fwd_states: [s_start, ..., s_meeting]"] --> M2["bwd_states: [s_meeting, ..., s_goal]"]
    M2 --> M3["Drop duplicate meeting state"]
    M3 --> M4["states = fwd_states + bwd_states[1:]"]
    M4 --> M5["actions = fwd_actions + bwd_actions"]
    M5 --> M6["Build (action, state) pairs<br/>starting with (-1, start)"]
  end

  Merge --> Concatenation
  Concatenation --> Result["return complete path"]
```

### Key Implementation Details

From `JAxtar/bi_stars/bi_astar.py` and `JAxtar/bi_stars/bi_search_base.py`:

- **Hindsight Transformation**: `inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)` transforms the backward search configuration so it treats the start state as its target
- **Intersection Check Timing**: Performed after every expansion using all valid neighbors (`flatten_filleds`), not just newly inserted states, to catch states added to the opposite frontier in the current iteration
- **Meeting Point Update**: Uses stored g-values from hashtables (`this_costs = search_result.get_cost(hash_idx)`), not candidate costs, to ensure correctness with duplicate discoveries
- **Backward Heuristic**: Can be disabled for fixed heuristics (`use_backward_heuristic = not heuristic.is_fixed`) to save computation
- **Termination Modes**:
  - `terminate_on_first_solution=True` (default): Fast termination, may not guarantee optimality
  - `terminate_on_first_solution=False`: Continues until `w * total_cost <= min(f_fwd, f_bwd)` for optimality proof
- **Path Reconstruction Convention**: Backward search stores forward actions in parent table (puxle convention), so no action inversion needed during reconstruction

### Optimality Considerations

To ensure an optimal path with `terminate_on_first_solution=False`:
1. Heuristic must be admissible in both directions
2. `cost_weight` should be 1.0 or very close to it (default: `1.0 - 1e-6`)
3. Search continues until the weighted meeting cost is proven optimal via f-value bounds:
   - `w * meeting.total_cost <= f_fwd_min` (all unexpanded forward states have higher f)
   - `w * meeting.total_cost <= f_bwd_min` (all unexpanded backward states have higher f)
