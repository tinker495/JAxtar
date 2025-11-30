# Q-Function Evaluation Commands (`eval qstar`, `eval qbeam`)

These commands are used to systematically evaluate the performance of trained neural network Q-functions using different search algorithms. They run the specified search algorithm over a large number of puzzle instances, collecting statistics on success rate, search time, path length, and nodes expanded.

## Commands

-   **`eval qstar`**: Evaluates using Q* Search.
-   **`eval qbeam`**: Evaluates using Q-Beam Search (Beam Search with Q-values).

## Usage

The basic syntax follows the pattern:

```bash
python main.py eval <algorithm> -p <puzzle_name> [OPTIONS]
```

Examples:

```bash
# Evaluate Q*
python main.py eval qstar -p rubikscube --num-eval 200

# Evaluate Q-Beam
python main.py eval qbeam -p rubikscube --num-eval 200 --batch-size 5000
```

## Options

These commands share common evaluation and puzzle options, tailored to the specific search algorithm.

### Evaluation Options (`@eval_options`)

-   `-ne, --num-eval`: The number of puzzles to evaluate.
    -   Type: `Integer`
    -   Default: `200`
-   `-b, --batch-size`: Batch size for the search. For **Q-Beam**, this determines the **beam width**.
    -   Type: `Integer`
    -   Default: `10000`
-   `-m, --max-node-size`: The maximum number of nodes to explore.
    -   Type: `String`
    -   Default: `20,000,000`
-   `-w, --cost-weight`: Weight for cost in search (w * g + Q).
    -   Type: `Float`
    -   Default: `0.6`
-   `-pr, --pop_ratio`: Ratio(s) for popping nodes from the priority queue.
    -   Type: `String`
    -   Default: `inf`
-   `-rn, --run-name`: Name of the evaluation run.
    -   Type: `String`
-   `--use-early-stopping`: Enable early stopping based on success rate.
    -   Type: `Boolean`
-   `--early-stop-patience`: Number of samples to check for early stopping.
    -   Type: `Integer`
-   `--early-stop-threshold`: Minimum success rate threshold.
    -   Type: `Float`

### Puzzle Options (`@eval_puzzle_options`)

-   `-p, --puzzle`: **(Required)** Specifies the puzzle.
    -   Type: `Choice`
-   `-pargs, --puzzle_args`: JSON string for puzzle arguments.
    -   Type: `String`
-   `-h, --hard`: Use "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: Single seed or comma-separated list of seeds.
    -   Type: `String`
    -   Default: `"0"`

### Q-Function Options (`@qfunction_options`)

These commands automatically use the neural Q-function defined in the puzzle configuration.

-   `--param-path`: Path to override the Q-function parameter file.
    -   Type: `String`
-   `--model-type`: Type of the Q-function model.
    -   Type: `String`
