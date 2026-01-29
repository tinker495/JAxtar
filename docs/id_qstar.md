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
