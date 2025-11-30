# Heuristic Evaluation Commands (`eval astar`, `eval astar_d`, `eval beam`)

These commands are used to systematically evaluate the performance of trained neural network heuristics using different search algorithms. They run the specified search algorithm over a large number of puzzle instances, collecting statistics on success rate, search time, path length, and nodes expanded.

## Commands

-   **`eval astar`**: Evaluates using A* Search.
-   **`eval astar_d`**: Evaluates using A* Deferred Search.
-   **`eval beam`**: Evaluates using Beam Search.

## Usage

The basic syntax follows the pattern:

```bash
python main.py eval <algorithm> -p <puzzle_name> [OPTIONS]
```

Examples:

```bash
# Evaluate A*
python main.py eval astar -p n-puzzle --num-eval 200

# Evaluate A* Deferred
python main.py eval astar_d -p rubikscube --num-eval 200

# Evaluate Beam Search
python main.py eval beam -p rubikscube --num-eval 200 --batch-size 5000
```

## Options

These commands share common evaluation and puzzle options, tailored to the specific search algorithm.

### Evaluation Options (`@eval_options`)

-   `-ne, --num-eval`: The number of puzzles to evaluate.
    -   Type: `Integer`
    -   Default: `200`
-   `-b, --batch-size`: Batch size for the search. For **Beam Search**, this determines the **beam width**.
    -   Type: `Integer`
    -   Default: `10000`
-   `-m, --max-node-size`: The maximum number of nodes to explore.
    -   Type: `String`
    -   Default: `20,000,000`
-   `-w, --cost-weight`: Weight for cost in search.
    -   Type: `Float`
    -   Default: `0.6` (may vary by preset)
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

### Heuristic Options (`@heuristic_options`)

These commands automatically use the neural heuristic defined in the puzzle configuration.

-   `--param-path`: Path to override the heuristic parameter file.
    -   Type: `String`
-   `--model-type`: Type of the heuristic model.
    -   Type: `String`
