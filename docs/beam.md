# Beam Search Command (`beam`)

The `beam` command solves a puzzle using Beam Search. Beam Search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. It is like a breadth-first search that optimizes memory usage by only storing a fixed number of best states at each level (the "beam width"). It is not guaranteed to find the optimal solution but is often much faster and more memory-efficient than A*.

## Usage

The basic syntax for the `beam` command is:

```bash
python main.py beam [OPTIONS]
```

Example:

```bash
python main.py beam -p rubikscube -nn -b 10000
```

## Options

The `beam` command uses option groups similar to `astar`, but the search behavior is governed by beam search principles.

### Puzzle Options (`@puzzle_options`)

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

For Beam Search, some options have specific implications:

-   `-b, --batch_size`: **Critical for Beam Search.** This effectively sets the **Beam Width** (or a multiple of it, depending on implementation details). It limits the number of nodes kept at each step.
    -   Type: `Integer`
    -   Default: `10000`
-   `-m, --max_node_size`: The maximum number of nodes to explore.
    -   Type: `String`
-   `-w, --cost_weight`: The weight `w` for the path cost.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio for popping nodes.
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

## Implementation Notes (JAxtar/beamsearch/heuristic_beam.py)

This section documents the actual control flow and data flow in `JAxtar/beamsearch/heuristic_beam.py`.
Beam Search is a greedy BFS variant that only keeps the top-B best nodes (where B is the `beam_width`) at each depth. This prevents search space explosion while maintaining a high chance of finding a solution.

Unlike A*, beam search operates **without a global hash table or priority queue**. Instead, it uses:

- `beam`: Current set of B active states (only current depth, not historical)
- `trace_*` tables: History buffers for path reconstruction across all depths
- `select_beam()`: Top-k selection by score for beam pruning
- `non_backtracking_mask()`: Local cycle prevention without global deduplication

The core loop is built by `_heuristic_beam_loop_builder(...)` and executed by `jax.lax.while_loop`.

### High-Level Control Flow

```mermaid
flowchart TD
  subgraph Build["Build time (beam_builder)"]
    B1["_heuristic_beam_loop_builder"] --> B2["create init, cond, body functions"]
    B2 --> B3["jax.jit(beam)"]
    B3 --> B4["Warm-up: run with default config/state"]
    B4 --> B5["XLA Compilation & Caching"]
  end

  subgraph Run["Run time (beam_fn)"]
    R1["call compiled beam_fn"] --> R2["init_loop_state"]
    R2 --> R3["jax.lax.while_loop"]
    R3 --> R4["final solved check & index extraction"]
    R4 --> R5["return BeamSearchResult"]
  end

  B5 --> R1

  subgraph Init["init_loop_state"]
    I1["BeamSearchResult.build (allocate buffers)"] --> I2["prepare_heuristic_parameters"]
    I2 --> I3["seed beam with start state"]
    I3 --> I4["compute initial h & score"]
    I4 --> I5["initialize trace_* tables (depth 0)"]
    I5 --> I6["set active_trace[0] = 0"]
    I6 --> I7["create BeamSearchLoopState"]
  end

  subgraph While["while loop (lax.while_loop)"]
    W1["loop_condition"] --> W2{"Depth < max_depth &<br/>Beam has states &<br/>Not solved?"}
    W2 -- yes --> W3["loop_body"] --> W1
    W2 -- no --> W4["exit"]
  end

  R2 --> I1
  R3 --> W1
```

### Data Structures At A Glance

```mermaid
flowchart TB
  subgraph BSR["BeamSearchResult"]
    direction TB

    subgraph Current["Current Beam (live states at depth d)"]
      BEAM["beam\n(B states)"]
      COST["cost\n(g values, size B)"]
      DIST["dist\n(h values, size B)"]
      SCORES["scores\n(f = w*g + h, size B)"]
      PARENT["parent_index\n(index within previous beam, size B)"]
      ACTIVE["active_trace\n(pointer into trace tables, size B)"]
    end

    subgraph Meta["Search Metadata"]
      DEPTH["depth\n(current iteration)"]
      SOLVED["solved\n(boolean)"]
      SOLVED_IDX["solved_idx\n(index in beam)"]
      GEN_SIZE["generated_size\n(total nodes expanded)"]
    end

    subgraph Trace["Trace System (history for path reconstruction)"]
      TRACE_STATE["trace_state\n(all states ever in beam, capacity = (max_depth+1)*B)"]
      TRACE_PARENT["trace_parent\n(parent trace ID, same capacity)"]
      TRACE_ACTION["trace_action\n(action from parent, same capacity)"]
      TRACE_COST["trace_cost\n(g value, same capacity)"]
      TRACE_DIST["trace_dist\n(h value, same capacity)"]
      TRACE_DEPTH["trace_depth\n(depth when recorded, same capacity)"]
    end

    subgraph Static["Static Configuration"]
      BW["beam_width\n(B)"]
      MD["max_depth"]
      AS["action_size"]
    end
  end

  subgraph LS["BeamSearchLoopState"]
    LSR["search_result: BeamSearchResult"]
    LSC["solve_config: Puzzle.SolveConfig"]
    LSP["params: heuristic_parameters"]
  end

  LS --> BSR

  ACTIVE -.link.-> TRACE_PARENT
  ACTIVE -.link.-> TRACE_ACTION
  ACTIVE -.link.-> TRACE_STATE

  style Current fill:#e3f2fd
  style Trace fill:#fff3e0
  style Meta fill:#f3e5f5
  style Static fill:#e8f5e9
```

Key differences from A*:
- **No global hashtable**: Beam search does not maintain a global mapping of states to indices. States are deduplicated only within each batch of candidates before beam selection.
- **No priority queue**: Instead of a heap-based frontier, beam search uses `select_beam()` (top-k selection) to pick the B best candidates by score.
- **Trace system**: Since old beams are discarded, a trace history is maintained to enable path reconstruction. `active_trace[i]` points to an index in `trace_*` arrays, which store the full lineage.

### Loop Body Data Flow (One Iteration)

The loop body expands all B states in the current beam, deduplicates candidates in-batch,
applies non-backtracking filtering, selects the top-B candidates by score, updates the trace
history, and advances to the next depth.

Key implementation details from `JAxtar/beamsearch/heuristic_beam.py`:

- Neighbor generation is fully batched: `puzzle.batched_get_neighbours(solve_config, beam_states, filled_mask)`
- Candidate costs are computed: `child_costs = base_costs + transition_cost`
- **In-batch deduplication**: `xnp.unique_mask(flat_states, key=flat_cost, filled=flat_valid)` keeps only the best instance of each state within the current candidates
- **Non-backtracking filter**: `non_backtracking_mask()` rejects candidates that match any of their N most recent ancestors in the trace
- **Heuristic computation**: Candidates are reordered with `stable_partition_three()` to group valid work, then heuristics are computed in chunks via `jax.lax.fori_loop` over action rows
- **Beam selection**: `select_beam(scores, beam_width, pop_ratio, min_keep)` performs top-k selection with optional ratio-based pruning
- **Final deduplication**: After selection, a second `xnp.unique_mask()` ensures no duplicate states in the new beam
- **Trace update**: New beam states are recorded in `trace_*` tables at indices `(depth+1) * beam_width + slot`

```mermaid
flowchart TD
  subgraph Phase1["Phase 1: Expand Current Beam"]
    S1["fetch beam states (B states)"] --> S2["batched_get_neighbours<br/>(generates B x action_size candidates)"]
    S2 --> S3["compute child_costs = base_costs + transition_cost"]
    S3 --> S4["compute child_valid mask<br/>(filled & finite cost)"]
  end

  subgraph Phase2["Phase 2: In-Batch Deduplication"]
    S5["flatten candidates to (B * action_size,)"] --> S6["xnp.unique_mask<br/>(keep cheapest duplicate within batch)"]
    S6 --> S7["update child_valid with unique_mask"]
  end

  subgraph Phase3["Phase 3: Non-Backtracking Filter"]
    S8["broadcast parent_trace IDs<br/>(active_trace for each candidate)"] --> S9["non_backtracking_mask<br/>(scan through ancestor chain)"]
    S9 --> S10["reject candidates matching<br/>any of N recent ancestors"]
    S10 --> S11["update child_valid with allowed_mask"]
  end

  subgraph Phase4["Phase 4: Reorder & Compute Heuristics"]
    S12["stable_partition_three<br/>(move valid candidates to front)"] --> S13["reshape into action-major rows<br/>(action_size x B)"]
    S13 --> S14["jax.lax.fori_loop over action rows"]
    S14 --> S15["_compute_chunk:<br/>batched_distance for valid row members"]
    S15 --> S16["scatter results back to flattened order"]
  end

  subgraph Phase5["Phase 5: Scoring & Selection"]
    S17["compute scores = w*g + h"] --> S18["select_beam<br/>(top-k by score with pop_ratio pruning)"]
    S18 --> S19["gather selected states/costs/dists/actions/parents"]
    S19 --> S20["final xnp.unique_mask<br/>(deduplicate within selected B)"]
    S20 --> S21["update selected_valid with unique_valid"]
  end

  subgraph Phase6["Phase 6: Trace Update & Advance"]
    S22["gather parent_trace_ids<br/>(from previous beam's active_trace)"] --> S23["compute next_trace_ids<br/>(depth+1) * beam_width + slot"]
    S23 --> S24["update trace_parent, trace_action,<br/>trace_cost, trace_dist, trace_depth, trace_state<br/>at next_trace_ids"]
    S24 --> S25["set active_trace = next_trace_ids<br/>(or TRACE_INVALID if not valid)"]
    S25 --> S26["update beam, cost, dist, scores, parent_index"]
    S26 --> S27["increment depth & generated_size"]
    S27 --> S28["return new BeamSearchLoopState"]
  end

  S4 --> S5
  S7 --> S8
  S11 --> S12
  S16 --> S17
  S21 --> S22
```

### Detailed Phase Breakdowns

#### Phase 4: Heuristic Computation with Chunking

Since candidates are sparse (many invalid entries), the implementation reorders them to pack valid work into the front of batches, improving GPU utilization.

```mermaid
flowchart TB
  subgraph Reorder["Reordering for Efficiency"]
    R1["flat_states (B*action_size)"] --> R2["stable_partition_three<br/>(valid candidates first)"]
    R2 --> R3["ordered_states (valid work at front)"]
    R3 --> R4["reshape into (action_size, B) chunks"]
  end

  subgraph Loop["fori_loop over action rows"]
    L1["for i in range(action_size):"] --> L2["row_mask = chunk_valid[i]"]
    L2 --> L3{"any(row_mask)?"}
    L3 -- yes --> L4["variable_heuristic_batch_switcher<br/>(compute h for valid entries)"]
    L3 -- no --> L5["skip (keep inf)"]
    L4 --> L6["write dist_row to chunk_dists[i]"]
    L5 --> L6
    L6 --> L1
  end

  subgraph Scatter["Scatter back to original order"]
    SC1["ordered_dists (reordered)"] --> SC2["flat_dists[global_perm] = ordered_dists"]
    SC2 --> SC3["flat_dists (original order)"]
    SC3 --> SC4["reshape to (action_size, B)"]
  end

  R4 --> L1
  L6 --> SC1
```

#### Phase 5: Beam Selection (select_beam)

The `select_beam` function performs top-k selection with optional ratio-based pruning to focus on promising candidates.

```mermaid
flowchart TB
  SEL1["flat_scores (B*action_size candidates)"] --> SEL2["top_k(-scores, beam_width)<br/>(negate for min-heap behavior)"]
  SEL2 --> SEL3["selected_scores, topk_idx"]
  SEL3 --> SEL4["best_score = selected_scores[0]"]
  SEL4 --> SEL5["threshold = best_score * pop_ratio + ε"]
  SEL5 --> SEL6["within_ratio = score <= threshold"]
  SEL6 --> SEL7["forced_keep = idx < min(min_keep, valid_count)"]
  SEL7 --> SEL8["keep_mask = (within_ratio OR forced_keep) AND valid"]
  SEL8 --> SEL9["return (selected_scores, topk_idx, keep_mask)"]

  style SEL1 fill:#e3f2fd
  style SEL9 fill:#c8e6c9
```

- `pop_ratio`: Allows pruning candidates much worse than the best. E.g., `pop_ratio=1.1` keeps only candidates within 10% of the best score.
- `min_keep`: Ensures at least N candidates survive (if available), even if they exceed the ratio threshold.

#### Phase 6: Trace System Update

The trace system maintains a history of all states that have ever been in the beam, enabling path reconstruction without a global hash table.

```mermaid
flowchart LR
  subgraph Depth0["Depth 0 (Start)"]
    D0_T0["trace_idx = 0<br/>state = start<br/>parent = INVALID<br/>action = PAD"]
    D0_B0["beam[0] = start<br/>active_trace[0] = 0"]
  end

  subgraph Depth1["Depth 1 (First Expansion)"]
    D1_T0["trace_idx = B+0<br/>parent = 0<br/>action = a0"]
    D1_T1["trace_idx = B+1<br/>parent = 0<br/>action = a1"]
    D1_TN["trace_idx = B+(B-1)<br/>parent = ?<br/>action = ?"]
    D1_B["beam = selected states<br/>active_trace[i] = B+i"]
  end

  subgraph Depth2["Depth 2 (Second Expansion)"]
    D2_T0["trace_idx = 2*B+0<br/>parent = B+j<br/>action = ak"]
    D2_TN["..."]
    D2_B["beam = new selected states<br/>active_trace[i] = 2*B+i"]
  end

  D0_T0 --> D1_T0
  D0_T0 --> D1_T1
  D1_T0 --> D2_T0
  D1_TN --> D2_TN

  D0_B0 -.active_trace.-> D0_T0
  D1_B -.active_trace.-> D1_T0
  D1_B -.active_trace.-> D1_TN
  D2_B -.active_trace.-> D2_T0

  style D0_T0 fill:#fff3e0
  style D1_T0 fill:#fff3e0
  style D2_T0 fill:#fff3e0
  style D0_B0 fill:#e3f2fd
  style D1_B fill:#e3f2fd
  style D2_B fill:#e3f2fd
```

Key trace mechanics:
1. **Trace capacity**: `(max_depth + 1) * beam_width` preallocated slots
2. **Trace index formula**: `trace_idx = depth * beam_width + slot_in_beam`
3. **Parent linking**: `trace_parent[trace_idx]` points to parent's trace_idx from previous depth
4. **Action recording**: `trace_action[trace_idx]` stores the action taken from parent
5. **State storage**: `trace_state[trace_idx]` stores the actual state for non-backtracking checks and path reconstruction
6. **Active mapping**: `active_trace[i]` maps current beam slot i to its trace_idx

When a solution is found, `get_solved_path()` uses `_reconstruct_trace_indices()` to walk backward through `trace_parent` links, collecting states and actions.

### Non-Backtracking Filter

Beam search can get stuck in local loops (e.g., A -> B -> A -> B -> ...) without global deduplication. The `non_backtracking_steps` parameter prevents this by rejecting candidates that match any of their N most recent ancestors.

```mermaid
flowchart TB
  NB1["candidate_states (current children)"] --> NB2["parent_trace_ids (from active_trace)"]
  NB2 --> NB3["jax.lax.scan for N steps backward"]

  subgraph Scan["Scan Body (_scan_fn)"]
    SC1["trace_ids (current ancestor level)"] --> SC2["gather ancestor_states = trace_state[trace_ids]"]
    SC2 --> SC3["matches = _batched_state_equal<br/>(candidate_states, ancestor_states)"]
    SC3 --> SC4["blocked = blocked OR matches"]
    SC4 --> SC5["trace_ids = trace_parent[trace_ids]<br/>(move to next ancestor level)"]
    SC5 --> SC1
  end

  NB3 --> Scan
  Scan --> NB4["return NOT blocked"]
  NB4 --> NB5["allowed_mask (candidates not matching ancestors)"]

  style NB1 fill:#e3f2fd
  style NB5 fill:#c8e6c9
  style Scan fill:#fff3e0
```

Algorithm:
1. Start with `parent_trace_ids` (immediate parents of candidates)
2. For each lookback step:
   - Fetch ancestor states from `trace_state[trace_ids]`
   - Check equality with `candidate_states`
   - Mark matches as blocked
   - Move to next ancestor level via `trace_parent[trace_ids]`
3. Return mask of candidates that don't match any ancestor

Example with `non_backtracking_steps=3`:
- Candidate C is rejected if it equals parent P, grandparent GP, or great-grandparent GGP
- This prevents loops like A -> B -> A (2-cycle) or A -> B -> C -> A (3-cycle)
- Does NOT prevent longer cycles (requires global hashtable for that)

### Trace System Path Reconstruction

```mermaid
flowchart TB
  REC1["solved_idx (index in final beam)"] --> REC2["start_node = active_trace[solved_idx]"]
  REC2 --> REC3["_reconstruct_trace_indices<br/>(while_loop backward through trace_parent)"]

  subgraph ReconLoop["Reconstruction while_loop"]
    RL1["path_idx array (preallocated)"] --> RL2{"idx != TRACE_INVALID &<br/>step < max_steps?"}
    RL2 -- yes --> RL3["path_idx[step] = idx"]
    RL3 --> RL4["idx = trace_parent[idx]"]
    RL4 --> RL5["step += 1"]
    RL5 --> RL2
    RL2 -- no --> RL6["return path_idx, length"]
  end

  REC3 --> ReconLoop
  ReconLoop --> REC4["gather states/actions from trace_* tables"]
  REC4 --> REC5["reverse arrays (path was built backward)"]
  REC5 --> REC6["return path"]

  style REC1 fill:#e3f2fd
  style REC6 fill:#c8e6c9
  style ReconLoop fill:#fff3e0
```

The reconstruction logic:
1. **Find starting point**: `active_trace[solved_idx]` gives the trace_idx of the solution
2. **Walk backward**: Follow `trace_parent` links until reaching `TRACE_INVALID` (the root)
3. **Collect path**: Record trace indices in `path_idx` array
4. **Gather data**: Use `path_idx` to index into `trace_state`, `trace_action`, etc.
5. **Reverse**: Path was built from goal to start, so reverse it

This approach avoids storing full paths for every beam state, trading off storage for reconstruction time.

### JIT Compilation Strategy

`beam_builder(...)` returns a JIT-compiled function (`beam_fn = jax.jit(beam)`).
To avoid extremely long compilation times from tracing complex puzzle logic on real inputs,
it triggers compilation once using `puzzle.SolveConfig.default()` and `puzzle.State.default()`.

This means:

- First call compiles and caches the XLA program.
- Subsequent calls reuse the compiled program as long as shapes/dtypes/static args match.
