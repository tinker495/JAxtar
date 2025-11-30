# Q-Beam Search Command (`qbeam`)

The `qbeam` command solves a puzzle using a variation of Beam Search guided by a Q-function (Q-Beam). Instead of a standard heuristic $h(n)$, it uses the learned Q-values $Q(s, a)$ to score and select nodes. This allows for beam search strategies driven by reinforcement learning models.

## Usage

The basic syntax for the `qbeam` command is:

```bash
python main.py qbeam [OPTIONS]
```

Example:

```bash
python main.py qbeam -p rubikscube -nn -b 5000
```

## Options

The `qbeam` command combines beam search mechanics with Q-function options.

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

For Q-Beam Search:

-   `-b, --batch_size`: **Critical.** Sets the **Beam Width** (or capacity) for the search.
    -   Type: `Integer`
    -   Default: `10000`
-   `-m, --max_node_size`: The maximum number of nodes to explore.
    -   Type: `String`
-   `-w, --cost_weight`: The weight `w` for the path cost in the priority calculation.
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

### Q-Function Options (`@qfunction_options`)

-   `-nn, --neural_qfunction`: Use a pre-trained neural network as the Q-function.
    -   Type: `Flag`
-   `--param-path`: Path to the Q-function parameter file.
    -   Type: `String`
-   `--model-type`: Type of the Q-function model.
    -   Type: `String`

### Visualization Options (`@visualize_options`)

-   `-vt, --visualize_terminal`: Renders the solution path in the terminal.
    -   Type: `Flag`
-   `-vi, --visualize_imgs`: Generates images and GIF for the solution.
    -   Type: `Flag`
-   `-mt, --max_animation_time`: Max duration for GIF.
    -   Type: `Integer`
