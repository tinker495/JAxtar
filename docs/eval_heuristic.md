# Heuristic Evaluation Command (`eval heuristic`)

The `eval heuristic` command is used to systematically evaluate the performance of a trained neural network heuristic. It runs the A* search algorithm over a large number of puzzle instances, collecting statistics on success rate, search time, path length, and the number of nodes expanded. This is a crucial tool for assessing the quality of a trained heuristic model.

## Usage

The basic syntax for the `eval heuristic` command is:

```bash
python main.py eval heuristic -p <puzzle_name> [OPTIONS]
```

Example: Evaluate the `n-puzzle` heuristic on 200 puzzles starting from seed 1000.

```bash
python main.py eval heuristic -p n-puzzle --num-eval 200 -s 1000
```

## How It Works

This command always uses the pre-configured **neural network heuristic** for the specified puzzle. It pairs the neural heuristic with the **A\* search algorithm** to solve the puzzles. The `-nn` flag from other commands is not needed here as its usage is implied.

The evaluation can be run in two ways depending on the `-s, --seeds` option:
1.  **Single Seed**: If one seed is provided (e.g., `-s 1000`), it serves as the starting point. The command will evaluate on a sequence of puzzles with seeds from `1000` to `1000 + num_eval - 1`.
2.  **Multiple Seeds**: If a comma-separated list of seeds is provided (e.g., `-s 1,5,10,42`), the command will evaluate only on the puzzles corresponding to those specific seeds.

By default, this command attempts to use the "hard" version of puzzles if one is defined in the puzzle's configuration.

## Options

The `eval heuristic` command combines options for evaluation, puzzle selection, and the underlying heuristic.

### Evaluation Options (`@eval_options`)

These options control the evaluation process itself. They override the default evaluation settings.

-   `--num-eval`: The number of puzzles to evaluate when a single starting seed is provided.
    -   Type: `Integer`
    -   Default: `200`
-   `--batch-size`: The batch size for the A* search algorithm.
    -   Type: `Integer`
    -   Default: `10000`
-   `--max-node-size`: The maximum number of nodes to explore during search.
    -   Type: `Integer`
    -   Default: `20,000,000`
-   `--cost-weight`: The weight `w` for the path cost in `f(n) = w * g(n) + h(n)`.
    -   Type: `Float`
    -   Default: `0.6`

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment for the evaluation.

-   `-p, --puzzle`: **(Required)** Specifies the puzzle whose heuristic is to be evaluated.
    -   Type: `Choice`
    -   Choices: `n-puzzle`, `rubikscube`, etc.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-ps, --puzzle_size`: Sets the size for puzzles that support it (e.g., `4` for a 15-puzzle).
    -   Type: `String`
    -   Default: `default`
-   `-h, --hard`: Use a "hard" version of the puzzle if available. For evaluation commands, this is often the default behavior if a hard version is defined.
    -   Type: `Flag`
-   `-s, --seeds`: A single seed (to use with `--num-eval`) or a comma-separated list of seeds.
    -   Type: `String`
    -   Default: `"0"`

### Heuristic Options (`@heuristic_options`)

For the `eval heuristic` command, these options are mostly implicit.

-   `-nn, --neural_heuristic`: This flag is **not required**. The command is specifically designed to evaluate the neural heuristic defined in the puzzle's configuration.
