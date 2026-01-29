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
