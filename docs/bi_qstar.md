# Bidirectional Q\* Command (`bi_qstar`)

The `bi_qstar` command solves a puzzle using the Bidirectional Q\* search algorithm. It performs bidirectional search using Q-values to guide the expansion in both forward and backward directions. This allows for more informed search steps compared to standard heuristic-based search.

## Usage

The basic syntax for the `bi_qstar` command is:

```bash
python main.py bi_qstar [OPTIONS]
```

Example:

```bash
python main.py bi_qstar -p rubikscube -nn
```

## Options

The `bi_qstar` command uses similar option groups to the `qstar` command.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore.
-   `-b, --batch_size`: Batch size.
-   `-w, --cost_weight`: Path cost weight.
-   `-pr, --pop_ratio`: Pop ratio.
-   `-vm, --vmap_size`: Parallel solve size via vmap.
-   `--debug`: Disable JIT.
-   `--profile`: Enable profiling.
-   `--show_compile_time`: Print compile time.

### Q-Function Options (`@qfunction_options`)

-   `-nn, --neural_qfunction`: Use neural network Q-function.
-   `--param-path`: Path to Q-function parameters.
-   `--model-type`: Q-function model type.

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Render path in terminal.
-   `-vi, --visualize_imgs`: Generate images/GIF.
-   `-mt, --max_animation_time`: Max GIF duration.

---

## Implementation Notes (JAxtar/bi_stars/bi_qstar.py)

Bidirectional Q* (`bi_qstar`) applies the action-value guidance of $Q(s, a)$ to both forward and backward searches. This algorithm is highly specialized for environments where a Q-function has been trained for both forward and inverse dynamics.

The implementation is built around two `SearchResult` instances (forward/backward) wrapped in a `BiDirectionalSearchResult`, each with its own hashtable and priority queue. The core loop is executed by `jax.lax.while_loop` with alternating expansions from both directions.

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (bi_qstar_builder)"]
    B1["_bi_qstar_loop_builder"] --> B2["create init, cond, body functions"]
    B2 --> B3["build_bi_search_result<br/>(allocate forward/backward tables)"]
    B3 --> B4["jax.jit(bi_qstar)"]
    B4 --> B5["Warm-up: run with default config/state"]
    B5 --> B6["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (bi_qstar_fn)"]
    R1["call compiled bi_qstar_fn"] --> R2["prepare_q_parameters<br/>(forward & backward)"]
    R2 --> R3["init_loop_state<br/>(initialize both frontiers)"]
    R3 --> R4["jax.lax.while_loop"]
    R4 --> R5["materialize_meeting_point_hashidxs<br/>(ensure meeting in both HTs)"]
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
    W1["common_bi_loop_condition"] --> W2{"Has Work & Not Terminated?"}
    W2 -- yes --> W3["loop_body<br/>(expand both directions)"] --> W1
    W2 -- no --> W4["exit"]
  end

  R3 --> I1
  R4 --> W1
```

### Data Structures for Bidirectional Q*

```mermaid
flowchart LR
  subgraph BiResult["BiDirectionalSearchResult"]
    FWD["forward: SearchResult"]
    BWD["backward: SearchResult"]
    MEET["meeting: MeetingPoint"]
  end

  subgraph FwdSR["Forward SearchResult"]
    FHT["hashtable (state→idx)"]
    FPQ["priority_queue<br/>(key→Parant_with_Costs)"]
    FCOST["cost table (g-values)"]
    FDIST["dist table<br/>(Q-values cache)"]
    FPARENT["parent table (Parent)"]
  end

  subgraph BwdSR["Backward SearchResult"]
    BHT["hashtable (state→idx)"]
    BPQ["priority_queue<br/>(key→Parant_with_Costs)"]
    BCOST["cost table (g-values)"]
    BDIST["dist table<br/>(Q-values or V-values cache)"]
    BPARENT["parent table (Parent)"]
  end

  subgraph MeetStruct["MeetingPoint"]
    M1["fwd_hashidx, bwd_hashidx"]
    M2["fwd_cost, bwd_cost, total_cost"]
    M3["found: bool"]
    M4["Deferred fields:<br/>fwd_has_hashidx, bwd_has_hashidx<br/>fwd_parent_hashidx, fwd_parent_action<br/>bwd_parent_hashidx, bwd_parent_action"]
  end

  FWD --> FwdSR
  BWD --> BwdSR
  MEET --> MeetStruct

  subgraph LoopState["BiLoopStateWithStates"]
    LS1["bi_result"]
    LS2["solve_config, inverse_solveconfig"]
    LS3["params_forward, params_backward"]
    LS4["current_forward, current_backward"]
    LS5["states_forward, states_backward"]
    LS6["filled_forward, filled_backward"]
  end

  LS1 --> BiResult
```

### Backward Direction Scoring Modes

Scoring a backward transition $(s_{next} \to s_{prev})$ using a forward-oriented Q-function is non-trivial. JAxtar provides several `backward_mode` options:

| Mode | Description | Requirements |
| :--- | :--- | :--- |
| **`edge_q`** | Uses $Q(s, a)$ of the *forward equivalent* edge. | `inverse_action_map` |
| **`value_v`** | Uses $V(s) = \min_a Q(s, a)$ as a node heuristic. | None (Auto-fallback) |
| **`dijkstra`** | Ignores Q-values in the backward direction. | None |
| **`auto`** | Uses `edge_q` if possible, otherwise `value_v`. | Recommended |

```mermaid
flowchart TD
  subgraph ModeSelection["Backward Mode Selection (not is_forward)"]
    START["use_backward_q=True"] --> CHECK{"backward_mode?"}
    CHECK -- "auto" --> AUTO{"puzzle.inverse_action_map<br/>exists?"}
    AUTO -- yes --> EDGE_Q["use edge_q:<br/>Q(parent, inverse_action_map[a])"]
    AUTO -- no --> VALUE_V["use value_v:<br/>V(child) = min_a Q(child, a)"]

    CHECK -- "edge_q" --> CHECK2{"inverse_action_map?"}
    CHECK2 -- yes --> EDGE_Q
    CHECK2 -- no --> FALLBACK["fallback to value_v<br/>(with warning)"]
    FALLBACK --> VALUE_V

    CHECK -- "value_v" --> VALUE_V
    CHECK -- "dijkstra" --> DIJKSTRA["use_q=False:<br/>use true step costs"]
  end

  EDGE_Q --> COMP1["Priority key:<br/>f = w*g + Q(parent, mapped_action)"]
  VALUE_V --> COMP2["Priority key:<br/>f = w*g(parent->child) + V(child)"]
  DIJKSTRA --> COMP3["Priority key:<br/>f = w*g + step_cost"]
```

### Inverse Action Mapping (edge_q mode)

The `edge_q` mode requires `puzzle.inverse_action_map` to translate backward action indices to their forward equivalents, enabling Q-value reuse:

```mermaid
flowchart LR
  subgraph Forward["Forward Direction"]
    FS["Parent State s"] --> FA["Action a<br/>(forward)"]
    FA --> FCHILD["Child State s'"]
    FQ["Q(s, a)"] -.-> FA
  end

  subgraph Backward["Backward Direction (edge_q)"]
    BCHILD["Parent State s'<br/>(was child)"] --> BINV["Inverse Neighbour [i]<br/>(returns s)"]
    BINV --> BS["Child State s<br/>(was parent)"]
    BMAP["inverse_action_map[i] → a"] -.-> BINV
    BQ["Q(s', inverse_action_map[i])<br/>≠ Q(s, a)<br/>(wrong direction!)"] -.x BINV
    CORRECT["Use Q(s, a) via mapping:<br/>Q(parent_s, mapped_action)"] -.-> BINV
  end

  subgraph Mapping["inverse_action_map"]
    MAP["For each inverse action i:<br/>map[i] = forward action<br/>that produces inverse[i]"]
  end

  BMAP --> MAP

  style FQ fill:#90EE90
  style BQ fill:#FFB6C1
  style CORRECT fill:#87CEEB
```

### Loop Body Data Flow (One Iteration)

The loop body performs expansions in both directions, using Q-functions to score edges and detect intersections between frontiers.

```mermaid
flowchart TD
  subgraph Start["Loop State Input"]
    IN1["BiLoopStateWithStates:<br/>- bi_result<br/>- current_forward, states_forward, filled_forward<br/>- current_backward, states_backward, filled_backward<br/>- params_forward, params_backward"]
  end

  subgraph FwdExpand["Forward Expansion"]
    F1["Check: filled_forward.any() AND<br/>forward not at capacity"] --> F2{"Expand Forward?"}
    F2 -- yes --> F3["_expand_direction_q<br/>(is_forward=True, use_q=True)"]
    F2 -- no --> F4["Skip, keep current forward state"]
  end

  subgraph BwdExpand["Backward Expansion"]
    B1["Check: filled_backward.any() AND<br/>backward not at capacity"] --> B2{"Expand Backward?"}
    B2 -- yes --> B3["_expand_direction_q<br/>(is_forward=False, use_q=use_backward_q)"]
    B2 -- no --> B4["Skip, keep current backward state"]
  end

  subgraph Output["Loop State Output"]
    OUT1["BiLoopStateWithStates:<br/>- updated bi_result (with new meeting)<br/>- new current/states/filled for both directions"]
  end

  IN1 --> F1
  IN1 --> B1
  F3 --> B1
  F4 --> B1
  B3 --> OUT1
  B4 --> OUT1

  subgraph ExpandDetail["_expand_direction_q (one direction)"]
    E1["Fetch current states/costs"] --> E2["Tile states, actions, costs<br/>for all actions"]
    E2 --> E3{"Backward Mode?"}

    E3 -- "value_v" --> V1["Generate neighbours<br/>(look-ahead)"]
    V1 --> V2["Packed+Chunked Q-evaluation:<br/>Q(child, :) for all children"]
    V2 --> V3["V(child) = min_a Q(child, a)"]
    V3 --> V4["Priority key:<br/>f = w*g(parent->child) + V(child)"]
    V4 --> V5["dist = V(child)"]

    E3 -- "edge_q" --> Q1["Q-evaluation on parent:<br/>Q(parent, :)"]
    Q1 --> Q2["Map actions via inverse_action_map:<br/>q_vals = q_vals[inv_map, :]"]
    Q2 --> Q3["Priority key:<br/>f = w*g(parent) + Q(parent, mapped_a)"]
    Q3 --> Q4["dist = Q(parent, mapped_a)"]

    E3 -- "forward or dijkstra" --> D1["Q-evaluation on parent:<br/>Q(parent, :)"]
    D1 --> D2["Priority key:<br/>f = w*g(parent) + Q(parent, a)"]
    D2 --> D3["dist = Q(parent, a)"]

    V5 --> PRUNE{"look_ahead_pruning?"}
    Q4 --> PRUNE
    D3 --> PRUNE

    PRUNE -- yes --> P1["Look up candidates in this HT"]
    P1 --> P2["optimal_mask: unique & better cost"]
    P2 --> P3["Early meeting detection:<br/>update_meeting_point_best_only_deferred"]

    PRUNE -- no --> P4["optimal_mask = filled"]

    P3 --> INS["Insert optimal candidates into PQ"]
    P4 --> INS

    INS --> POP["Pop next batch with actions:<br/>pop_full_with_actions"]
    POP --> INTER["Check intersection with opposite HT"]
    INTER --> UPD["update_meeting_point<br/>(post-pop intersection)"]
    UPD --> RET["Return updated bi_result,<br/>new current/states/filled"]
  end

  F3 --> ExpandDetail
  B3 --> ExpandDetail
```

### Detailed Forward Direction Q-Evaluation

```mermaid
flowchart TD
  subgraph FwdQEval["Forward Q-Evaluation (use_q=True)"]
    FQ1["Parent states from pop:<br/>states, filled"] --> FQ2["Q-function evaluation:<br/>q_vals = Q(states, :)<br/>[batch_size, action_size]"]
    FQ2 --> FQ3["Transpose to action-major:<br/>q_vals.T → [action_size, batch_size]"]
    FQ3 --> FQ4["Priority keys:<br/>key[a,i] = w*g[i] + Q(state[i], a)"]
    FQ4 --> FQ5["Dist cache:<br/>dist = Q(parent, a)"]
  end

  subgraph FwdLookAhead["Forward Look-Ahead Pruning"]
    LA1["Generate neighbours:<br/>neighbour_look_ahead, ncosts"] --> LA2["Look-ahead costs:<br/>look_ahead_costs = g + ncosts"]
    LA2 --> LA3["Distinct score for uniqueness:<br/>distinct = look_ahead_costs ± 1e-5*Q"]
    LA3 --> LA4["unique_mask: unique by distinct score"]
    LA4 --> LA5["Lookup in forward HT"]
    LA5 --> LA6["Reconstruct old Q-values:<br/>Q_old = old_dist + step_cost"]
    LA6 --> LA7{"pessimistic_update?"}
    LA7 -- yes --> LA8["dist = max(Q_new, Q_old)"]
    LA7 -- no --> LA9["dist = min(Q_new, Q_old)"]
    LA8 --> LA10["optimal_mask: unique & better cost"]
    LA9 --> LA10
  end

  FQ5 --> LA1
  LA10 --> FWDINSERT["Insert into forward PQ"]
  FWDINSERT --> FWDPOP["Pop next forward batch"]
```

### Detailed Backward Direction Q-Evaluation (edge_q vs value_v)

```mermaid
flowchart TD
  subgraph BwdEdgeQ["Backward edge_q Mode"]
    EQ1["Parent states (backward direction):<br/>states, filled"] --> EQ2["Q-function evaluation:<br/>q_vals = Q(states, :)<br/>[batch_size, action_size]"]
    EQ2 --> EQ3["Remap via inverse_action_map:<br/>q_vals = q_vals[inv_map, :]"]
    EQ3 --> EQ4["Transpose to action-major:<br/>q_vals.T → [action_size, batch_size]"]
    EQ4 --> EQ5["Priority keys:<br/>key[a,i] = w*g[i] + Q_remapped(state[i], a)"]
    EQ5 --> EQ6["Dist cache:<br/>dist = Q_remapped(parent, a)"]
    EQ6 --> EQ7["Continue with look-ahead pruning<br/>(same as forward)"]
  end

  subgraph BwdValueV["Backward value_v Mode"]
    VV1["Parent states (backward direction):<br/>states, filled"] --> VV2["Generate inverse neighbours:<br/>neighbour_look_ahead, ncosts<br/>[action_size, batch_size]"]
    VV2 --> VV3["Flatten to candidates:<br/>[action_size * batch_size]"]
    VV3 --> VV4["Packed reordering:<br/>stable_partition_three<br/>(valid entries first)"]
    VV4 --> VV5["Reshape to chunks:<br/>[action_size, batch_size]"]
    VV5 --> VV6["Chunked Q-evaluation via scan:<br/>for each chunk:<br/>  Q(chunk_states, :) → [b, a]<br/>  V = min_a Q(chunk, a)"]
    VV6 --> VV7["Reverse permutation:<br/>restore original order"]
    VV7 --> VV8["V-values for all candidates:<br/>V(child) for each neighbour"]
    VV8 --> VV9["Priority keys:<br/>key = w*(g + step_cost) + V(child)"]
    VV9 --> VV10["Dist cache:<br/>dist = V(child)"]
    VV10 --> VV11["Relaxed meeting mask:<br/>unique by cost only"]
    VV11 --> VV12["Look-ahead pruning:<br/>optimal_mask & meeting detection"]
  end

  subgraph BwdDijkstra["Backward dijkstra Mode (use_q=False)"]
    DJ1["use_q=False"] --> DJ2["Generate neighbours:<br/>neighbour_look_ahead, ncosts"]
    DJ2 --> DJ3["Use step costs as 'Q-values':<br/>q_vals = ncosts"]
    DJ3 --> DJ4["Priority keys:<br/>key = w*g + step_cost"]
    DJ4 --> DJ5["Dist cache:<br/>dist = 0 (step_cost - step_cost)"]
    DJ5 --> DJ6["Continue with standard logic"]
  end
```

### Meeting Point Detection and Materialization

Bidirectional Q* uses two mechanisms for detecting meeting points:

```mermaid
flowchart TD
  subgraph EarlyMeeting["Early Meeting Detection (Look-Ahead)"]
    EM1["During expansion, before PQ insert"] --> EM2["Candidate neighbours generated<br/>but not yet in this_sr HT"]
    EM2 --> EM3["Check candidates in opposite_sr HT"]
    EM3 --> EM4{"Found in opposite?"}
    EM4 -- yes --> EM5["Compute total_cost = this_cost + opposite_cost"]
    EM5 --> EM6{"Better than current meeting?"}
    EM6 -- yes --> EM7["update_meeting_point_best_only_deferred:<br/>Store edge representation<br/>(parent_hashidx, action)"]
    EM6 -- no --> EM8["Skip"]
    EM4 -- no --> EM8
  end

  subgraph PostPopMeeting["Post-Pop Meeting Detection"]
    PM1["After pop_full_with_actions"] --> PM2["Newly popped states now in this_sr HT"]
    PM2 --> PM3["Check popped states in opposite_sr HT"]
    PM3 --> PM4{"Found in opposite?"}
    PM4 -- yes --> PM5["Compute total_cost = this_cost + opposite_cost"]
    PM5 --> PM6{"Better than current meeting?"}
    PM6 -- yes --> PM7["update_meeting_point:<br/>Store both hashidxs<br/>(fwd_hashidx, bwd_hashidx)"]
    PM6 -- no --> PM8["Skip"]
    PM4 -- no --> PM8
  end

  subgraph Materialization["Meeting Materialization (After Loop)"]
    MAT1["materialize_meeting_point_hashidxs"] --> MAT2{"meeting.found?"}
    MAT2 -- yes --> MAT3{"Both fwd/bwd_has_hashidx?"}
    MAT3 -- no --> MAT4["Pick meeting state:<br/>- from existing hashidx (if available)<br/>- or compute from edge (parent_hashidx, action)"]
    MAT4 --> MAT5["Insert missing meeting state<br/>into HT with proper g-value and parent"]
    MAT5 --> MAT6["Update meeting with both hashidxs"]
    MAT3 -- yes --> MAT7["Already materialized"]
    MAT2 -- no --> MAT8["No meeting found"]
  end

  EM7 --> MAT1
  PM7 --> MAT1
```

### Loop Condition and Termination

```mermaid
flowchart TD
  subgraph Condition["common_bi_loop_condition"]
    C1["Check frontier status"] --> C2{"At least one direction<br/>has nodes AND capacity?"}
    C2 -- no --> C3["STOP: No work remaining"]
    C2 -- yes --> C4["Compute min f-values:<br/>fwd_min_f, bwd_min_f"]
    C4 --> C5["bi_termination_condition"]
  end

  subgraph Termination["bi_termination_condition"]
    T1{"terminate_on_first_solution?"}
    T1 -- yes --> T2{"meeting.found?"}
    T2 -- yes --> T3["STOP: Solution found"]
    T2 -- no --> T4["CONTINUE"]

    T1 -- no --> T5["Weighted meeting cost:<br/>w * meeting.total_cost"]
    T5 --> T6{"w*total_cost ≤ fwd_min_f<br/>AND<br/>w*total_cost ≤ bwd_min_f?"}
    T6 -- yes --> T7["STOP: Optimal solution proven"]
    T6 -- no --> T8["CONTINUE: May find better path"]
  end

  C5 --> T1
  T3 --> STOP["Exit loop"]
  T7 --> STOP
  T4 --> CONT["Continue loop"]
  T8 --> CONT
```

### Key Optimizations

#### 1. Packed Value-Heuristic (`value_v`)
When using `value_v` mode, computing $V(s) = \min_a Q(s, a)$ for every child state in a batch can be expensive. JAxtar uses a **packed+chunked** approach:
- Child states are filtered and packed contiguously using `stable_partition_three`.
- Q-evaluations are processed in smaller chunks to avoid memory spikes.
- $V(s)$ is extracted as the minimum across the action dimension of the Q-output.

#### 2. Inverse Action Support (`edge_q`)
If the puzzle defines an `inverse_action_map`, the backward search can interpret an inverse action ID as its corresponding forward action. This allows it to use the exact $Q(s, a)$ training for the edge, making the backward search as "smart" as the forward one.

#### 3. Deferred Meeting Detection
Like other bidirectional deferred variants, `bi_qstar` can detect meeting points during look-ahead without inserting states into the hash table. The meeting point stores edge information `(parent_hashidx, action)` for deferred states, which are materialized via `materialize_meeting_point_hashidxs` after the loop completes.

#### 4. Pessimistic vs Optimistic Updates
When a state is revisited via multiple edges, Q* must decide how to combine Q-values:
- **Pessimistic** (`pessimistic_update=True`): Use `max(Q_new, Q_old)` — conservative, prevents over-optimism
- **Optimistic** (`pessimistic_update=False`): Use `min(Q_new, Q_old)` — aggressive, may find better paths faster

The reconstructed Q-value for comparison is: $Q_{old} = dist_{stored} + step\_cost$
