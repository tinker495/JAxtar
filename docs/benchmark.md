# Benchmark Command

The `benchmark` command is designed to evaluate search algorithms against standardized benchmark datasets. This ensures consistent and comparable results across different experiments and configurations. It allows you to run any of the supported search algorithms (`astar`, `astar_d`, `beam`, `qstar`, `qbeam`) on a registered benchmark.

## Usage

The basic syntax for the `benchmark` command is:

```bash
python main.py benchmark <algorithm> [OPTIONS]
```

Example: Run A* benchmark on the default benchmark dataset.

```bash
python main.py benchmark astar
```

Example: Run Q* benchmark on the `sokoban` benchmark with a specific sample limit.

```bash
python main.py benchmark qstar --benchmark sokoban --sample-limit 100
```

## Subcommands

The available subcommands correspond to the search algorithms:

-   `astar`: Benchmark A* Search.
-   `astar_d`: Benchmark A* Deferred Search.
-   `qstar`: Benchmark Q* Search.
-   `beam`: Benchmark Beam Search.
-   `qbeam`: Benchmark Q-Beam Search.

## Options

The `benchmark` command uses a set of options to select the benchmark dataset and configure the evaluation.

### Benchmark Options (`@benchmark_options`)

-   `--benchmark`: The key of the benchmark dataset to evaluate.
    -   Type: `Choice`
    -   Default: (depends on registered bundles, e.g., `n-puzzle`)
-   `--benchmark-args`: JSON string containing keyword arguments for the benchmark constructor.
    -   Type: `String`
-   `--sample-limit`: The maximum number of samples to evaluate from the benchmark dataset.
    -   Type: `Integer`
-   `--sample-ids`: A comma-separated list of specific sample IDs to evaluate. This overrides `--sample-limit`.
    -   Type: `String`

### Evaluation Options (`@eval_options`)

These options control the evaluation process, similar to the `eval` command.

-   `-ne, --num-eval`: Number of puzzles to evaluate (often overridden by benchmark settings, but can be used to limit the run).
    -   Type: `Integer`
-   `-b, --batch-size`: Batch size for the search.
    -   Type: `Integer`
-   `-m, --max-node-size`: Maximum number of nodes to explore.
    -   Type: `String`
-   `-w, --cost-weight`: Weight for cost in search.
    -   Type: `Float`
-   `-pr, --pop_ratio`: Ratio(s) for popping nodes from the priority queue.
    -   Type: `String`
-   `-rn, --run-name`: Name of the evaluation run.
    -   Type: `String`
-   `--use-early-stopping`: Enable early stopping based on success rate.
    -   Type: `Boolean`
-   `--early-stop-patience`: Number of samples to check for early stopping.
    -   Type: `Integer`
-   `--early-stop-threshold`: Minimum success rate threshold.
    -   Type: `Float`

### Model Options

Depending on the algorithm (`astar` vs `qstar`), you can specify model options.

-   `--param-path`: Optional override for the parameter file path.
    -   Type: `String`
-   `--model-type`: Type of the model (heuristic or Q-function) to load from the benchmark configuration.
    -   Type: `String`
    -   Default: `default`
-   `--output-dir`: Directory to store run artifacts (defaults to `runs/<timestamp>`).
    -   Type: `Path`

## Results and Logging

For detailed information on where results are stored, file formats (`results.csv`, `config.json`), and how to generate comparison reports, please refer to the [**Benchmark Logging Documentation**](./benchmark_logging.md).
