# Benchmark Results & Logging

When running benchmarks or evaluations in `JAxtar`, the results are automatically logged to the file system. This document explains the structure of these logs, the file formats, and how to interpret the generated metrics.

## Output Directory Structure

By default, all run artifacts are stored in the `runs/` directory. You can override this using the `--output-dir` option.

### Single Run

A standard evaluation run produces a directory named with the pattern `<puzzle>_<algo>_<timestamp>` (e.g., `rubikscube_astar_2023-10-27_10-00-00`).

```text
runs/rubikscube_astar_2023-10-27_10-00-00/
├── config.json             # Full configuration of the run
├── results.csv             # Detailed per-sample results
├── console.log             # Capture of the terminal output
├── heuristic_accuracy.png  # (If heuristic used) Accuracy plots
├── path_cost_distribution.png
├── search_time_by_path_cost.png
└── ... (other plots)
```

### Parameter Sweep

If you provide multiple values for options like `--pop_ratio` or `--cost_weight` (e.g., `-w 0.8,0.9`), `JAxtar` performs a parameter sweep. The directory structure is slightly different:

```text
runs/rubikscube_astar_sweep_<timestamp>/
├── comparison_report.png   # Aggregated plots comparing sub-runs
├── pr_inf_cw_0.8_bs_10000/ # Sub-run 1
│   ├── config.json
│   ├── results.csv
│   └── ...
└── pr_inf_cw_0.9_bs_10000/ # Sub-run 2
    ├── config.json
    ├── results.csv
    └── ...
```

## Key Output Files

### `config.json`

Contains the exact parameters used for the run, including puzzle settings, model metadata, and search options. This ensures reproducibility.

### `results.csv`

This is the raw data file containing one row per evaluated puzzle sample. Key columns include:

-   `seed`: The seed or sample ID.
-   `solved`: Boolean indicating if a solution was found.
-   `search_time_s`: Time taken in seconds.
-   `nodes_generated`: Total number of nodes generated/explored.
-   `path_cost`: The cost of the found solution.
-   `path_action_count`: Number of actions in the solution.
-   `benchmark_optimal_path_cost`: (If available) The known optimal cost from the benchmark.
-   `matches_optimal_path`: (If available) Whether the found solution matches the optimal path/cost.

## Visualizations

`JAxtar` automatically generates several plots to help analyze performance:

-   **Path Cost Distribution**: Histogram of solution costs.
-   **Search Time vs. Path Cost**: Scatter plot showing the relationship between difficulty (cost) and time.
-   **Nodes Generated vs. Path Cost**: Scatter plot showing search effort vs. difficulty.
-   **Heuristic Accuracy**: (If applicable) Plots showing the correlation between heuristic estimates and actual costs-to-go.
-   **Benchmark Comparison**: (If applicable) Plots comparing found solution lengths against optimal benchmark lengths.

## Comparing Runs

You can compare multiple completed runs using the `eval compare` command. This generates a report highlighting configuration differences and performance metrics.

```bash
# Compare two specific run directories
python main.py eval compare runs/run_A runs/run_B

# Compare all sub-runs within a sweep directory
python main.py eval compare runs/my_sweep_dir
```

## Metrics Glossary

-   **Success Rate**: The percentage of puzzles solved within the resource limits (node limit, batch size).
-   **Optimal Rate**: The percentage of solved puzzles where the found solution cost matches the known optimal cost (requires benchmark data).
-   **Avg Nodes Generated**: The average search effort required. Lower is better.
-   **Avg Time**: The average wall-clock time to solve.
-   **Cost Gap**: The average difference between the found path cost and the optimal path cost.
