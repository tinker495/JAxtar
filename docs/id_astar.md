# Iterative Deepening A\* Command (`id_astar`)

The `id_astar` command solves a puzzle using the Iterative Deepening A\* (IDA\*) search algorithm. IDA\* combines the space efficiency of depth-first search with the optimality of A\*. It performs a series of depth-first searches with increasing cost thresholds.

## Usage

The basic syntax for the `id_astar` command is:

```bash
python main.py id_astar [OPTIONS]
```

Example:

```bash
python main.py id_astar -p rubikscube -nn
```

## Options

The `id_astar` command uses the same option groups as `astar`.

### Puzzle Options (`@puzzle_options`)

-   `-p, --puzzle`: Specifies the puzzle to solve.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle arguments.
-   `-h, --hard`: Use a hard version of the puzzle.
-   `-s, --seeds`: Comma-separated list of seeds.

### Search Options (`@search_options`)

-   `-m, --max_node_size`: Max nodes to explore (per iteration).
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
