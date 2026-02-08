# Q\* Command

The `qstar` command solves a puzzle using the Q\* search algorithm. Q\* is a variation of A\* that is particularly useful in reinforcement learning contexts, where a Q-function is learned to estimate the cost-to-go. This implementation is fully JIT-compiled with JAX for high performance on accelerators.

## Search Correctness Notes

- Deferred-style Q* queueing paths use a shared action-major batch insertion helper for consistent PQ insertion behavior across single and bidirectional variants.
- Deferred pop selection applies `pop_ratio`/`min_pop` to the final merged key batch.
- Path reconstruction consistency checks raise a structured diagnostic payload when monotonic-cost assumptions are violated.

## Usage

The basic syntax for the `qstar` command is:

```bash
python main.py qstar [OPTIONS]
```

A common use case is to solve a specific puzzle with a neural Q-function:

```bash
python main.py qstar -p rubikscube -nn
```

## Options

The `qstar` command uses a combination of option groups to configure the puzzle, search algorithm, Q-function, and visualization.

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment to be solved.

-   `-p, --puzzle`: Specifies the puzzle to solve.
    -   Type: `Choice`
    -   Default: `n-puzzle`
    -   Choices: `n-puzzle`, `rubikscube`, `slidepuzzle`, etc. (depends on configuration).
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
    -   Example: `python main.py qstar -pargs '{"size": 4}'`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A comma-separated list of seeds for generating initial puzzle states. Using multiple seeds will run the solver multiple times.
    -   Type: `String`
    -   Default: `"0"`

### Search Options (`@search_options`)

These options control the behavior of the Q\* search algorithm itself.

-   `-m, --max_node_size`: The maximum number of nodes to explore. Supports scientific notation.
    -   Type: `String`
-   `-b, --batch_size`: The number of nodes to process in a single batch on the GPU.
    -   Type: `Integer`
-   `-w, --cost_weight`: The weight `w` for the path cost in the search priority calculation, which is analogous to `f(n) = w * g(n) + Q(s,a)`. A lower value prioritizes states with better Q-values, making the search greedier.
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

### Q-Function Options (`@qfunction_options`)

These options determine which Q-function to use for guiding the search.

-   `-nn, --neural_qfunction`: Use a pre-trained neural network as the Q-function. If not set, a default, non-ML Q-function is used.
    -   Type: `Flag`
-   `--param-path`: Path to the Q-function parameter file.
    -   Type: `String`
-   `--model-type`: Type of the Q-function model.
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
