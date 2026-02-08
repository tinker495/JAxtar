# A\* Deferred Command (`astar_d`)

The `astar_d` command solves a puzzle using the A\* Deferred search algorithm. This is a variation of A\* where node expansion is deferred, which can be beneficial in certain search spaces or when using specific types of heuristics (e.g., heavy heuristics). It maintains the optimality guarantees of A\* under consistent heuristics.

## Search Correctness Notes

- Deferred pop selection is computed from the merged final key batch, not from the pre-merge pop batch.
- Candidate insertion into the priority queue uses a shared action-major batch helper so deferred variants follow identical insertion semantics.
- Path reconstruction failures raise a structured diagnostic message (`PATH_RECONSTRUCTION_DIAGNOSTIC`) with loop/cost-drop details.

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
