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
-   `-ps, --puzzle_size`: A simpler way to set the size for puzzles that support it.
    -   Type: `String`
    -   Default: `default`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A comma-separated list of seeds for generating initial puzzle states. Using multiple seeds will run the solver multiple times.
    -   Type: `String`
    -   Default: `"0"`

### Search Options (`@search_options`)

These options control the behavior of the A\* search algorithm itself.

-   `-m, --max_node_size`: The maximum number of nodes to explore. Supports scientific notation.
    -   Type: `String`
    -   Default: `2e6`
-   `-b, --batch_size`: The number of nodes to process in a single batch on the GPU.
    -   Type: `Integer`
    -   Default: `10000`
-   `-w, --cost_weight`: The weight `w` for the path cost in `f(n) = w * g(n) + h(n)`. A value of `1.0` is standard A\*, while a lower value (< 1.0) prioritizes nodes closer to the goal (greedy search), and a higher value (> 1.0) prioritizes exploring cheaper paths.
    -   Type: `Float`
    -   Default: `0.9`
-   `--pop-ratio`: Controls the search beam width. Nodes are expanded if their cost is within `pop_ratio` percent of the best node's cost (e.g., 0.1 allows a 10% margin). A value of `inf` corresponds to a fixed-width beam search determined by the batch size.
    -   Type: `Float`
    -   Default: `inf`
-   `-vm, --vmap_size`: The number of different initial states to solve in parallel using `jax.vmap`.
    -   Type: `Integer`
    -   Default: `1`
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

### Visualization Options (`@visualize_options`)

These options control how the final solution path is displayed.

-   `-vt, --visualize_terminal`: Renders the solution path step-by-step in the terminal.
    -   Type: `Flag`
-   `-vi, --visualize_imgs`: Generates an image for each step of the solution and saves them, along with a GIF animation, in a timestamped folder within `tmp/`.
    -   Type: `Flag`
-   `-mt, --max_animation_time`: Sets the maximum duration for the generated GIF animation, in seconds. The frame rate is adjusted to fit this duration.
    -   Type: `Integer`
    -   Default: `10`
