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
