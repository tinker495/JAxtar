# Iterative Deepening Q\* Command (`id_qstar`)

The `id_qstar` command solves a puzzle using the Iterative Deepening Q\* (ID-Q\*) search algorithm. This is an experimental algorithm that adapts the principles of IDA\* to use learned Q-values. It performs iterative deepening search guided by Q-value estimates.

## Usage

The basic syntax for the `id_qstar` command is:

```bash
python main.py id_qstar [OPTIONS]
```

Example:

```bash
python main.py id_qstar -p rubikscube -nn
```

## Options

The `id_qstar` command uses similar option groups to the `qstar` command.

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

## Implementation Notes (JAxtar/id_stars/id_qstar.py)

This section documents the actual control flow and data flow in `JAxtar/id_stars/id_qstar.py`.
ID-Q* is an experimental algorithm that brings Q-function guidance to the Iterative Deepening framework. It ranks transitions using action-values, allowing the search to prioritize paths that a reinforcement learning model deems most promising.

Unlike A* which uses a priority queue, ID-Q* uses a **stack-based DFS** within bounded cost thresholds, where Q-values replace heuristics for guidance.

The implementation is built around:
- `IDSearchBase`: Stack-based search state (LIFO for DFS)
- `IDFrontier`: Initial frontier built by Q-optimized ranking
- Dual-loop structure: outer loop increases bounds, inner loop performs bounded DFS

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (id_qstar_builder)"]
    B1["_id_qstar_loop_builder"] --> B2["create init, cond, body functions"]
    B2 --> B3["_id_qstar_frontier_builder<br/>(Q-optimized frontier)"]
    B3 --> B4["jax.jit(id_qstar)"]
    B4 --> B5["Warm-up: run with default config/state"]
    B5 --> B6["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (id_qstar_fn)"]
    R1["call compiled id_qstar_fn"] --> R2["init_loop_state"]
    R2 --> R3["Outer Loop (jax.lax.while_loop)"]
    R3 --> R4["final solved check & path extraction"]
    R4 --> R5["return IDSearchBase"]
  end

  B6 --> R1

  subgraph Init["init_loop_state"]
    I1["IDSearchBase.build (allocate stack)"] --> I2["prepare_q_parameters"]
    I2 --> I3["generate_frontier (Q-optimized)"]
    I3 --> I4["initialize_from_frontier with min-Q bound"]
    I4 --> I5["create IDLoopState"]
  end

  subgraph OuterLoop["Outer Loop (Iterative Deepening)"]
    O1["outer_cond"] --> O2{"Solved or Max Nodes?"}
    O2 -- no --> O3["Inner Loop (Bounded DFS)"]
    O3 --> O4["inner_cond"] --> O5{"Solved or Stack Empty?"}
    O5 -- no --> O6["inner_body (expand & push)"]
    O6 --> O4
    O5 -- yes --> O7["Increase bound to next_bound"]
    O7 --> O8["Regenerate frontier with new bound"]
    O8 --> O1
    O2 -- yes --> O9["exit"]
  end

  R2 --> I1
  R3 --> O1
```

### Data Structures At A Glance

```mermaid
flowchart LR
  SR["IDSearchBase"] --> STACK["stack\n(LIFO DFS storage)"]
  SR --> BOUND["bound & next_bound\n(cost thresholds)"]
  SR --> TRACE["trace arrays\n(parent, action, root)"]
  SR --> SOL["solution tracking\n(state, cost, actions)"]

  STACK --> ITEM["IDStackItem:\n- state\n- cost (g)\n- depth\n- action\n- parent_index\n- root_index\n- trace_index\n- trail (non-backtrack)\n- action_history"]

  LS["IDLoopState"] --> SR
  LS --> FRONTIER["IDFrontier\n(Q-ranked initial batch)"]
  LS --> CONFIG["solve_config"]
  LS --> PARAMS["Q-function parameters"]

  FRONTIER --> FSTATES["states batch"]
  FRONTIER --> FCOSTS["costs (g values)"]
  FRONTIER --> FSCORES["f_scores (Q-ranked)"]
  FRONTIER --> FTRAIL["trail (anti-backtrack)"]
```

### Q-Optimized Frontier Builder Detail

Unlike standard IDA* which ranks nodes by $h(s')$, ID-Q* evaluates $Q(s, a)$ on **parent states** to build and rank its initial frontier. This makes the very first batch of nodes extremely well-filtered according to the learned policy.

```mermaid
flowchart TD
  subgraph FrontierBuild["_id_qstar_frontier_builder (Initialization)"]
    F1["Initialize frontier with start state"] --> F2["while_loop condition:<br/>Not solved & has capacity & within limit"]
    F2 -- yes --> F3["Expand current batch"]
    F3 --> F4["batched_get_neighbours<br/>(generate all actions)"]
    F4 --> F5["Evaluate Q(parent_states, all_actions)"]
    F5 --> F6["Compute f = w*g + Q(s,a) for each child"]
    F6 --> F7["Apply deduplication (unique_mask)"]
    F7 --> F8["Apply non_backtracking filter"]
    F8 --> F9["select_top_k by f-score"]
    F9 --> F10["Update frontier with best candidates"]
    F10 --> F2
    F2 -- no --> F11["Return final IDFrontier"]
  end

  subgraph QEval["Q-Value Evaluation Details"]
    Q1["variable_q_parent_switcher<br/>(batched Q-function call)"] --> Q2["Q-values shape: [batch_size, action_size]"]
    Q2 --> Q3["Mask invalid with inf"]
    Q3 --> Q4["Ensure non-negative:<br/>max(0.0, Q)"]
    Q4 --> Q5["Transpose to [action_size, batch_size]"]
    Q5 --> Q6["Broadcast: f = w*g + Q"]
    Q6 --> Q7["Flatten to [flat_size]"]
  end

  F5 --> Q1
```

**Key Insight**: The frontier builder runs a mini-search using Q-values to pre-filter the search space before the main iterative deepening loop begins.

### Detailed Dual-Loop Structure

ID-Q* uses a nested loop structure: the outer loop increases the cost bound when the inner DFS exhausts all nodes within the current bound.

```mermaid
flowchart TD
  subgraph OuterLoop["Outer Loop (build_outer_loop)"]
    OC["outer_cond:<br/>Not solved & generated_count < max_nodes"] --> OB["outer_body"]

    OB --> IL["Run Inner Loop<br/>(while_loop with inner_cond/inner_body)"]
    IL --> ILR["Inner loop returns when:<br/>solved OR stack empty"]

    ILR --> CHECK{"Check result"}
    CHECK -- "Solved" --> RET1["Return with solution"]
    CHECK -- "Not Solved" --> INBOUND["Increase bound:<br/>bound = next_bound"]
    INBOUND --> REGEN["Regenerate frontier<br/>(with new bound)"]
    REGEN --> REINIT["Re-initialize from frontier:<br/>initialize_from_frontier"]
    REINIT --> OC
  end

  subgraph InnerLoop["Inner Loop (inner_body)"]
    IC["inner_cond:<br/>Not solved & stack not empty"] --> IB["inner_body"]

    IB --> POP["prepare_for_expansion<br/>(pop batch from stack)"]
    POP --> EXP["Expand nodes:<br/>batched_get_neighbours"]
    EXP --> QEVAL["Evaluate Q(parent, actions)"]
    QEVAL --> FCOMP["Compute f = w*g + Q(s,a)"]
    FCOMP --> PRUNE["Prune by bound:<br/>f > bound?"]
    PRUNE --> UPDATE["Update next_bound<br/>(track minimum pruned f)"]
    UPDATE --> DEDUP["apply_standard_deduplication"]
    DEDUP --> PUSH["expand_and_push<br/>(push valid children to stack)"]
    PUSH --> IC
  end

  OB --> IC
```

### Detailed Loop Body (Inner Body) Data Flow

The inner body is where the core Q-guided bounded DFS happens. Each iteration:
1. Pops a batch from the stack
2. Generates successors
3. Evaluates Q-values for action ranking
4. Prunes based on f-value vs bound
5. Deduplicates
6. Pushes valid successors back to stack

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Pop & Expand"]
    S1["prepare_for_expansion:<br/>Pop batch from stack"] --> S2["Extract parent states, costs, depths, trails"]
    S2 --> S3["batched_get_neighbours<br/>(all actions from parents)"]
    S3 --> S4["build_flat_children:<br/>Flatten to [batch_size * action_size]"]
  end

  subgraph Phase2["Phase 2: Q-Value Evaluation"]
    S5["variable_q_parent_switcher:<br/>Evaluate Q(parents, all_actions)"] --> S6["Q-values: [batch_size, action_size]"]
    S6 --> S7["Mask invalid states with inf"]
    S7 --> S8["Ensure non-negative: max(0.0, Q)"]
    S8 --> S9["Transpose: [action_size, batch_size]"]
    S9 --> S10["Compute f = w*g + Q(s,a)"]
    S10 --> S11["Flatten: flat_f [flat_size]"]
  end

  subgraph Phase3["Phase 3: Action-Level Pruning"]
    S12["f_prune_mask = flat_f > bound + eps"] --> S13["Find minimum among pruned:<br/>min_f_pruned"]
    S13 --> S14["Update next_bound:<br/>min(sr.next_bound, min_f_pruned)"]
    S14 --> S15["Filter valid:<br/>flat_valid &= (flat_f <= bound + eps)"]
  end

  subgraph Phase4["Phase 4: Deduplication"]
    S16["apply_standard_deduplication"] --> S17["unique_mask:<br/>First occurrence within batch"]
    S17 --> S18["apply_non_backtracking:<br/>Avoid cycles using trail"]
    S18 --> S19["Update flat_valid with masks"]
  end

  subgraph Phase5["Phase 5: Push to Stack"]
    S20["Build IDNodeBatch:<br/>Pack states, costs, depths, trails, etc."] --> S21["expand_and_push:<br/>Push valid children to stack"]
    S21 --> S22["Increment generated_count"]
    S22 --> S23["Return updated IDSearchBase"]
  end

  S4 --> S5
  S11 --> S12
  S15 --> S16
  S19 --> S20

  subgraph Conditional["Solution Detection"]
    C1["Check if any expanded node is solved"] --> C2{"any_solved?"}
    C2 -- yes --> C3["Record solution & return"]
    C2 -- no --> C4["Continue to Phase 5"]
  end

  S4 --> C1
  C4 --> S20
```

### Min-Q Bound Initialization

For the root of each iteration, ID-Q* calculates:
$$V(s) = \min_a Q(s, a)$$
and uses this value to initialize the iterative deepening bound, ensuring that the search space expands logically according to the best estimated path.

```mermaid
flowchart LR
  subgraph MinQ["Min-Q Initialization (_min_q)"]
    M1["variable_q_parent_switcher:<br/>Evaluate Q(states, all_actions)"] --> M2["Q-values: [batch_size, action_size]"]
    M2 --> M3["Mask invalid with inf"]
    M3 --> M4["Ensure non-negative: max(0.0, Q)"]
    M4 --> M5["Min over actions:<br/>jnp.min(q_vals, axis=-1)"]
    M5 --> M6["Return: [batch_size] min Q-values"]
  end

  subgraph Init["initialize_from_frontier"]
    I1["Receive IDFrontier from generator"] --> I2["Call _min_q on frontier.states"]
    I2 --> I3["Set initial bound:<br/>bound = w * g + min_q"]
    I3 --> I4["Initialize stack from frontier"]
    I4 --> I5["Set next_bound = inf"]
  end

  M6 --> I2
```

**Why Min-Q?** Using $\min_a Q(s,a)$ gives the most optimistic (lowest cost) estimate for reaching the goal from state $s$, making it an admissible lower bound for iterative deepening.

### Comparison: IDA* vs ID-Q* Decision Making

```mermaid
flowchart LR
  subgraph IDA["IDA* Decision"]
    A1["Parent state s"] --> A2["Expand all actions"]
    A2 --> A3["For each child s':<br/>Evaluate h(s')"]
    A3 --> A4["Compute f = w*g + h(s')"]
    A4 --> A5["Prune if f > bound"]
    A5 --> A6["State-level pruning:<br/>All or nothing"]
  end

  subgraph IDQ["ID-Q* Decision"]
    Q1["Parent state s"] --> Q2["Evaluate Q(s, all_actions)<br/>(action-dependent)"]
    Q2 --> Q3["For each action a:<br/>Q-value already computed"]
    Q3 --> Q4["Compute f = w*g + Q(s,a)"]
    Q4 --> Q5["Prune if f > bound"]
    Q5 --> Q6["Action-level pruning:<br/>Selective per-action"]
  end

  subgraph Key["Key Differences"]
    K1["IDA*: h(s') evaluated on children<br/>after expansion"]
    K2["ID-Q*: Q(s,a) evaluated on parent<br/>before generating child"]
    K3["IDA*: Heuristic guides child selection"]
    K4["ID-Q*: Q-function guides action selection"]
    K5["IDA*: Pruning at state level"]
    K6["ID-Q*: Pruning at action level"]
  end

  A6 -.-> K1
  Q6 -.-> K2
  A3 -.-> K3
  Q2 -.-> K4
  A5 -.-> K5
  Q5 -.-> K6
```

### Action-Dependent Pruning Details

In ID-Q*, pruning is fine-grained. Instead of pruning an entire state, the search can prune specific **actions** from a parent state if their $Q(s, a)$ exceeds the current bound. This results in a highly sparse and efficient search tree.

```mermaid
flowchart TD
  subgraph ActionPrune["Action-Level Pruning Process"]
    P1["Parent batch: [batch_size] states"] --> P2["Q(s, all_actions): [batch_size, action_size]"]
    P2 --> P3["Flatten to [flat_size] where<br/>flat_size = batch_size * action_size"]
    P3 --> P4["For each (parent, action) pair:<br/>f = w*g + Q(s,a)"]
    P4 --> P5{"f > bound?"}
    P5 -- yes --> P6["Mark action invalid:<br/>flat_valid[i] = False"]
    P5 -- no --> P7["Keep action valid:<br/>flat_valid[i] = True"]
    P6 --> P8["Track min pruned f:<br/>Update next_bound"]
    P7 --> P9["Generate child state:<br/>Push to stack"]
  end

  subgraph Example["Example: 3 actions, bound=10"]
    E1["State s, g=5"] --> E2["Q(s,a1)=2, Q(s,a2)=8, Q(s,a3)=3"]
    E2 --> E3["f1=5*w+2, f2=5*w+8, f3=5*w+3"]
    E3 --> E4["If w=1: f1=7, f2=13, f3=8"]
    E4 --> E5["Action a1: f1=7 <= 10 ✓<br/>Action a2: f2=13 > 10 ✗ (pruned)<br/>Action a3: f3=8 <= 10 ✓"]
    E5 --> E6["Only a1 and a3 children pushed to stack"]
    E6 --> E7["next_bound tracks min(13) for next iteration"]
  end
```

### Deduplication and Non-backtracking

Like ID-A*, ID-Q* employs `apply_standard_deduplication` and `non_backtracking_steps` to handle cycles and local oscillations, which is critical for search efficiency in complex puzzle environments.

```mermaid
flowchart TD
  subgraph Dedup["apply_standard_deduplication"]
    D1["Flat candidates: [flat_size]"] --> D2["unique_mask:<br/>Keep first occurrence by g-value"]
    D2 --> D3["apply_non_backtracking:<br/>Check against trail"]
    D3 --> D4["Final valid mask"]
  end

  subgraph NonBacktrack["Non-backtracking Logic"]
    N1["Each node stores trail:<br/>last k states visited"] --> N2["For each candidate child"]
    N2 --> N3["Check if child == parent"]
    N3 --> N4["Check if child in parent's trail"]
    N4 --> N5{"Match found?"}
    N5 -- yes --> N6["Block this child (cycle)"]
    N5 -- no --> N7["Allow this child"]
  end

  D3 --> N1
```

**Trail Management**: Each node maintains a trail of the last `non_backtracking_steps` states. When expanding, children that match any state in the trail are filtered out, preventing local cycles and redundant exploration.

### JIT Compilation Strategy

`id_qstar_builder(...)` returns a JIT-compiled function (`id_qstar_fn = jax.jit(id_qstar)`).
To avoid extremely long compilation times from tracing complex puzzle logic on real inputs,
it triggers compilation once using `puzzle.SolveConfig.default()` and `puzzle.State.default()`.

This means:

- First call compiles and caches the XLA program.
- Subsequent calls reuse the compiled program as long as shapes/dtypes/static args match.
