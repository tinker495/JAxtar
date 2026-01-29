# Bidirectional A\* Deferred Command (`bi_astar_d`)

The `bi_astar_d` command solves a puzzle using the Bidirectional A\* Deferred search algorithm. This combines bidirectional search (forward and backward) with deferred node expansion. It is useful for reducing the search space in complex problems where node expansion is costly.

## Usage

The basic syntax for the `bi_astar_d` command is:

```bash
python main.py bi_astar_d [OPTIONS]
```

Example:

```bash
python main.py bi_astar_d -p rubikscube -nn
```

## Options

The `bi_astar_d` command uses the same option groups as `bi_astar`.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore.
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

## Implementation Notes (JAxtar/bi_stars/bi_astar_d.py)

Bidirectional A* Deferred (`bi_astar_d`) is an advanced search algorithm that combines the search space reduction of bidirectional search with the computational efficiency of deferred expansion. This implementation is fully JIT-compiled with JAX for high performance on accelerators.

### Key Strategy: Deferred Meeting Detection

Unlike standard bidirectional search, `bi_astar_d` stores actions (edges) in the priority queue rather than fully materialized states. States are only generated when a node is popped from the priority queue.

1. **Pop & Materialize**: Both forward and backward directions pop node batches using `pop_full_with_actions`, which materializes child states from (parent, action) pairs during the pop operation.
2. **Intersection Check**: After materializing states during the pop step, each direction checks if the new states exist in the opposite direction's hash table using `check_intersection`.
3. **Edge-only Meeting**: With `look_ahead_pruning=True`, meeting points can be detected early by looking ahead at neighbors. If a meeting is found via an edge not yet in the hash table, it's recorded as (parent_hashidx, action) for later materialization.

The core loop is built by `_bi_astar_d_loop_builder(...)` and executed by `jax.lax.while_loop`.

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (bi_astar_d_builder)"]
    B1["build_bi_search_result<br/>(allocate forward/backward HT/PQ)"] --> B2["_bi_astar_d_loop_builder"]
    B2 --> B3["create init, cond, body functions"]
    B3 --> B4["jax.jit(bi_astar_d)"]
    B4 --> B5["Warm-up: run with default config/state"]
    B5 --> B6["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (bi_astar_d_fn)"]
    R1["call compiled bi_astar_d_fn"] --> R2["init_loop_state"]
    R2 --> R3["jax.lax.while_loop"]
    R3 --> R4["materialize_meeting_point_hashidxs"]
    R4 --> R5["mark forward/backward as solved"]
    R5 --> R6["return BiDirectionalSearchResult"]
  end

  B6 --> R1

  subgraph Init["init_loop_state"]
    I1["initialize_bi_loop_common"] --> I2["insert start into forward HT"]
    I2 --> I3["insert goal into backward HT"]
    I3 --> I4["check if start == goal"]
    I4 --> I5["create BiLoopStateWithStates"]
  end

  subgraph While["while loop (lax.while_loop)"]
    W1["loop_condition"] --> W2{"Not Solved & Has Work?"}
    W2 -- yes --> W3["loop_body"] --> W1
    W2 -- no --> W4["exit"]
  end

  R2 --> I1
  R3 --> W1
```

### Data Structures At A Glance

```mermaid
flowchart LR
  subgraph BiResult["BiDirectionalSearchResult"]
    FWD["forward: SearchResult"]
    BWD["backward: SearchResult"]
    MEET["meeting: MeetingPoint"]
  end

  subgraph SearchResult["SearchResult (per direction)"]
    HT["hashtable<br/>(state to index mapping)"]
    PQ["priority_queue<br/>(key to Parant_with_Costs)"]
    COST["cost table<br/>(g values)"]
    DIST["dist table<br/>(h values cache)"]
    PARENT["parent table<br/>(Parent: hashidx, action)"]
  end

  subgraph MeetingPt["MeetingPoint"]
    MP1["fwd_hashidx, bwd_hashidx<br/>(meeting indices in both HTs)"]
    MP2["fwd_cost, bwd_cost, total_cost<br/>(path costs)"]
    MP3["fwd_has_hashidx, bwd_has_hashidx<br/>(materialization flags)"]
    MP4["fwd_parent_hashidx, fwd_parent_action<br/>bwd_parent_hashidx, bwd_parent_action<br/>(edge representation for unmaterialized)"]
  end

  subgraph LoopState["BiLoopStateWithStates"]
    LS1["bi_result"]
    LS2["current_forward, current_backward<br/>(Current: hashidx, cost)"]
    LS3["states_forward, states_backward<br/>(materialized states)"]
    LS4["filled_forward, filled_backward<br/>(valid batch entries)"]
    LS5["params_forward, params_backward<br/>(heuristic parameters)"]
  end

  BiResult --> FWD
  BiResult --> BWD
  BiResult --> MEET
  FWD --> SearchResult
  BWD --> SearchResult
  MEET --> MeetingPt
  LoopState --> LS1
  LS1 --> BiResult
```

### Loop Body Data Flow (One Iteration)

The loop body expands both forward and backward directions in sequence. Each direction:
1. Pops the current batch and generates candidate edges (parent, action) pairs
2. Materializes child states for look-ahead
3. Performs deduplication and cost-based filtering
4. Detects meeting points (both materialized and edge-only)
5. Inserts optimal edges into the priority queue
6. Pops and materializes the next batch for the next iteration

Key implementation details from `JAxtar/bi_stars/bi_astar_d.py`:

- **Deferred Expansion**: `_expand_direction_deferred` generates (parent, action) pairs and inserts them into the PQ with keys based on parent's f-value
- **Look-ahead Pruning**: When enabled, neighbors are materialized early for cost-based filtering and early meeting detection
- **Backward Value Lookahead**: Optional optimization for backward direction that uses 1-step Q-value backup (V(s) = min_a [1 + h(s')]) to improve pruning
- **Edge-only Meeting Detection**: `update_meeting_point_best_only_deferred` can record meeting points without inserting into hash table
- **Eager Pop Expansion**: `pop_full_with_actions` performs state materialization during pop to ensure unique states in the batch

```mermaid
flowchart TD
  subgraph LoopBody["loop_body: Expand Both Directions"]
    START["BiLoopStateWithStates"] --> FWD_CHECK{"forward filled<br/>& not full?"}
    FWD_CHECK -- yes --> FWD_EXPAND["_expand_direction_deferred<br/>(forward)"]
    FWD_CHECK -- no --> FWD_SKIP["skip forward"]

    FWD_EXPAND --> BWD_CHECK{"backward filled<br/>& not full?"}
    FWD_SKIP --> BWD_CHECK

    BWD_CHECK -- yes --> BWD_EXPAND["_expand_direction_deferred<br/>(backward)"]
    BWD_CHECK -- no --> BWD_SKIP["skip backward"]

    BWD_EXPAND --> UPDATE["create new BiLoopStateWithStates"]
    BWD_SKIP --> UPDATE
    UPDATE --> END["return updated state"]
  end

  subgraph ExpandDir["_expand_direction_deferred: One Direction"]
    D1["Tile parent hashidx & costs<br/>(action_size x batch_size)"] --> D2["Generate action indices<br/>(0 to action_size-1)"]
    D2 --> D3{"look_ahead_pruning?"}

    D3 -- yes --> LA1["Look-Ahead Path"]
    D3 -- no --> NLA1["No Look-Ahead Path"]
  end

  subgraph LookAhead["Look-Ahead Path (Recommended)"]
    LA1["batched_get_neighbours<br/>(materialize children)"] --> LA2["compute look_a_head_costs<br/>(parent.g + step_cost)"]
    LA2 --> LA3["unique_mask: deduplicate<br/>by (state, cost)"]
    LA3 --> LA4["lookup_parallel in this HT<br/>(check if child exists)"]
    LA4 --> LA5["compute candidate_mask<br/>(new OR improved cost)"]
    LA5 --> LA6{"use_heuristic?"}

    LA6 -- yes --> LA7["compute/reuse h-values"]
    LA6 -- no --> LA8["use h=0"]

    LA7 --> LA9{"backward &<br/>value_lookahead?"}
    LA8 --> LA10["compute neighbour_keys<br/>(w*g + h)"]

    LA9 -- yes --> LA9A["backup: V(s) = min_a[1+h(s')]<br/>(top-k candidates only)"]
    LA9 -- no --> LA10
    LA9A --> LA10

    LA10 --> LA11["update_meeting_point_best_only_deferred<br/>(edge-only meeting detection)"]
    LA11 --> LA12["Sort & Insert Optimal Edges into PQ"]
  end

  subgraph NoLookAhead["No Look-Ahead Path"]
    NLA1["Use parent's h-value"] --> NLA2["optimal_mask = filled_tiles"]
    NLA2 --> NLA3["compute neighbour_keys<br/>(w*parent.g + parent.h)"]
    NLA3 --> NLA4["Insert All Edges into PQ"]
  end

  subgraph PopMaterialize["Pop & Materialize Next Batch"]
    LA12 --> PM1["pop_full_with_actions<br/>(pop edges, materialize states)"]
    NLA4 --> PM1
    PM1 --> PM2["Update HT: insert new states,<br/>update costs & parents"]
    PM2 --> PM3["check_intersection with opposite HT<br/>(detect materialized meetings)"]
    PM3 --> PM4["update_meeting_point<br/>(update if better path found)"]
    PM4 --> PM5["return: bi_result, current, states, filled"]
  end

  FWD_EXPAND -.-> D1
  BWD_EXPAND -.-> D1
```

### Deferred Expansion Details

```mermaid
flowchart TD
  subgraph DeferredConcept["Deferred Expansion Concept"]
    DC1["Standard A*: Insert states into PQ,<br/>compute h when inserting"] --> DC2["Deferred A*: Insert (parent, action) pairs into PQ,<br/>compute h when popping"]
    DC2 --> DC3["Benefit: Fewer heuristic evaluations<br/>(only for popped nodes)"]
  end

  subgraph PQValues["Priority Queue Values"]
    PQ1["Parant_with_Costs {<br/>  parent: Parent(hashidx, action)<br/>  cost: parent.g<br/>  dist: parent.h or lookahead.h<br/>}"]
    PQ2["Key = w * cost + dist"]
  end

  subgraph PopExpansion["pop_full_with_actions: Eager Expansion"]
    PE1["Pop batch of (parent, action) pairs"] --> PE2["For each pair: materialize child<br/>using puzzle.batched_get_actions"]
    PE2 --> PE3["Insert children into HT"]
    PE3 --> PE4["Deduplicate: ensure unique children in batch<br/>(prevents batch starvation in reversible envs)"]
    PE4 --> PE5{"use_heuristic?"}
    PE5 -- yes --> PE6["Compute h for new children"]
    PE5 -- no --> PE7["Use h=0"]
    PE6 --> PE8["Cache h in dist table"]
    PE7 --> PE8
    PE8 --> PE9["Update cost & parent tables"]
    PE9 --> PE10["Return: current (hashidx, cost), states, filled"]
  end

  DC3 --> PQ1
  PQ1 --> PQ2
  PQ2 -.used by.-> PE1
```

### Deferred Meeting Detection Mechanism

```mermaid
flowchart TD
  subgraph MeetingTypes["Two Types of Meeting Detection"]
    MT1["1. Materialized Meeting:<br/>State exists in both HTs"] --> MT2["Detected by check_intersection<br/>after pop_full_with_actions"]
    MT3["2. Edge-only Meeting:<br/>State exists in opposite HT,<br/>but not yet in this HT"] --> MT4["Detected by update_meeting_point_best_only_deferred<br/>during look-ahead"]
  end

  subgraph EdgeOnlyMeeting["Edge-only Meeting Detection"]
    EOM1["Look-ahead generates candidates"] --> EOM2["unique_mask: keep best cost per state"]
    EOM2 --> EOM3["lookup_parallel in opposite HT"]
    EOM3 --> EOM4{"candidate exists<br/>in opposite HT?"}
    EOM4 -- yes --> EOM5["Compute total_cost<br/>= this_cost + opposite_cost"]
    EOM4 -- no --> EOM6["Skip this candidate"]
    EOM5 --> EOM7{"total_cost <<br/>current best?"}
    EOM7 -- yes --> EOM8["Update MeetingPoint"]
    EOM7 -- no --> EOM9["Keep current best"]
  end

  subgraph MeetingPointUpdate["MeetingPoint Update (Edge-only)"]
    MPU1["Check if candidate exists in this HT"] --> MPU2{"exists in<br/>this HT?"}
    MPU2 -- yes --> MPU3["Store: hashidx<br/>Set: has_hashidx = True"]
    MPU2 -- no --> MPU4["Store: parent_hashidx, parent_action<br/>Set: has_hashidx = False"]
    MPU3 --> MPU5["Update meeting.fwd/bwd fields<br/>(depending on direction)"]
    MPU4 --> MPU5
    MPU5 --> MPU6["Update meeting.total_cost"]
  end

  MT2 -.uses.-> MaterializedMeeting
  MT4 -.uses.-> EOM1
  EOM8 -.triggers.-> MPU1

  subgraph MaterializedMeeting["Materialized Meeting Detection"]
    MM1["Popped states from this direction"] --> MM2["lookup_parallel in opposite HT"]
    MM2 --> MM3{"found in<br/>opposite HT?"}
    MM3 -- yes --> MM4["Compute total_cost<br/>= this.g + opposite.g"]
    MM3 -- no --> MM5["No meeting"]
    MM4 --> MM6{"total_cost <<br/>current best?"}
    MM6 -- yes --> MM7["update_meeting_point<br/>(both hashidx are valid)"]
    MM6 -- no --> MM8["Keep current best"]
  end
```

### materialize_meeting_point_hashidxs Flow

After the search loop terminates, the meeting point may be represented via an edge (parent_hashidx, action) on one or both sides. This function ensures that the meeting state has valid hashidx entries in both hash tables for path reconstruction.

```mermaid
flowchart TD
  subgraph Main["materialize_meeting_point_hashidxs"]
    M1["Input: bi_result with meeting"] --> M2{"meeting.found?"}
    M2 -- no --> M3["Return unchanged"]
    M2 -- yes --> M4["_pick_meeting_state"]
  end

  subgraph PickState["_pick_meeting_state: Retrieve Meeting State"]
    PS1{"meeting.fwd_has_hashidx?"}
    PS1 -- yes --> PS2["Get from forward HT<br/>using fwd_hashidx"]
    PS1 -- no --> PS3{"meeting.bwd_has_hashidx?"}
    PS3 -- yes --> PS4["Get from backward HT<br/>using bwd_hashidx"]
    PS3 -- no --> PS5["Compute from fwd edge<br/>or bwd edge"]
  end

  subgraph ComputeState["Compute State from Edge"]
    CS1["Extract parent state from HT"] --> CS2["Apply action to parent<br/>using puzzle.batched_get_actions"]
    CS2 --> CS3["Return child state"]
  end

  subgraph MaterializeSide["_materialize_side: Insert into HT"]
    MS1["lookup meeting_state in HT"] --> MS2{"exists?"}
    MS2 -- yes --> MS3["Use existing hashidx"]
    MS2 -- no --> MS4["Insert into HT,<br/>get new hashidx"]
    MS3 --> MS5["Update cost & parent<br/>if this path is better"]
    MS4 --> MS5
    MS5 --> MS6["Return: updated SearchResult, hashidx"]
  end

  M4 --> PS1
  PS2 --> M5
  PS4 --> M5
  PS5 --> CS1
  CS3 --> M5

  M5{"meeting.fwd_has_hashidx?"}
  M5 -- no --> M6["_materialize_side<br/>(forward)"]
  M5 -- yes --> M7{"meeting.bwd_has_hashidx?"}
  M6 --> M7
  M7 -- no --> M8["_materialize_side<br/>(backward)"]
  M7 -- yes --> M9["Refresh costs from HTs"]
  M8 --> M9
  M9 --> M10["Return: bi_result with<br/>both hashidx materialized"]

  M6 -.uses.-> MS1
  M8 -.uses.-> MS1
```

### Loop Condition & Termination

```mermaid
flowchart TD
  subgraph LoopCond["common_bi_loop_condition"]
    LC1["Check forward filled & not full"] --> LC2["Check backward filled & not full"]
    LC2 --> LC3{"has_work = any direction<br/>can expand?"}
    LC3 -- no --> LC4["Stop: no work"]
    LC3 -- yes --> LC5["get_min_f_value for both directions"]
  end

  subgraph MinFValue["get_min_f_value: Compute Lower Bound"]
    MF1["f = w * current.cost + sr.get_dist(current)"] --> MF2["Mask invalid entries with inf"]
    MF2 --> MF3["Return: min(f_values)"]
  end

  subgraph Termination["bi_termination_condition"]
    T1{"terminate_on<br/>_first_solution?"}
    T1 -- yes --> T2["Stop if meeting.found"]
    T1 -- no --> T3["weighted_meeting_cost<br/>= w * meeting.total_cost"]
    T3 --> T4{"weighted_meeting_cost<br/><= fwd_min_f?"}
    T4 -- yes --> T5{"weighted_meeting_cost<br/><= bwd_min_f?"}
    T4 -- no --> T6["Continue"]
    T5 -- yes --> T7["Stop: optimal solution found"]
    T5 -- no --> T6
  end

  LC5 --> MF1
  MF3 --> LC6["bi_termination_condition"]
  LC6 --> T1
  T2 --> LC7["should_terminate"]
  T7 --> LC7
  T6 --> LC8["should_not_terminate"]

  LC7 --> LC9{"has_work &<br/>~should_terminate?"}
  LC8 --> LC9
  LC9 -- yes --> LC10["Continue loop"]
  LC9 -- no --> LC11["Exit loop"]
```

### Mandatory Optimization: look_ahead_pruning

In `bi_astar_d`, `look_ahead_pruning` is **forced to True**. This is critical because:

1. **Accurate f-values**: Without look-ahead, the PQ key uses the parent's f-value, which doesn't reflect the true cost of the child. This breaks the termination condition.
2. **Edge-only Meeting Detection**: Look-ahead enables early meeting detection without inserting into the hash table, reducing total expansions.
3. **Better Pruning**: Filtering candidates before PQ insertion prevents unnecessary work.

The builder function enforces this with a warning if `look_ahead_pruning=False` is passed.
