# Benchmark Results & Logging

When running `benchmark` or training evaluations in `JAxtar`, results and plots are written automatically to the file system.

## Output Directory Structure

By default, artifacts are stored under `runs/<puzzle>_<algo>_<timestamp>` (or `runs/<run_name>` when `--run-name` is set). Override the root path with `--output-dir`.

### Single Run

```text
runs/rubikscube_astar_2023-10-27_10-00-00/
├── config.json                     # Full configuration and runtime_provenance
├── results.csv                     # Per-sample evaluation results
├── console.log                     # Captured terminal output
├── path_cost_distribution.png
├── search_time_by_path_cost.png
├── nodes_generated_by_path_cost.png
├── heuristic_accuracy.png          # (Or heuristic_accuracy_optimal_path.png)
├── benchmark_path_comparison.png   # (When evaluating an exact benchmark)
├── path_states/                    # Optional saved path state arrays
└── expansion_plots/                # Per-seed expansion & tree plots (up to --max-expansion-plots)
```

### Parameter Sweep

On the CLI, `-pr`/`--pop_ratio` parses comma-separated lists for sweeps (e.g., `-pr inf,0.3,0.1`), whereas `-w`/`--cost-weight` and `-b`/`--batch-size` are scalar CLI flags (list-valued sweeps across `cost_weight` or `batch_size` come from presets such as training `EvalOptions`). The parent directory keeps the `<puzzle>_<algo>_<timestamp>` pattern, and each combination writes to a `pr_<pr>_cw_<cw>_bs_<bs>` subdirectory where `inf` is formatted as `Infinity`:

```text
runs/rubikscube_astar_2023-10-27_10-00-00/
├── path_cost_comparison.png
├── search_time_comparison.png
├── nodes_generated_comparison.png
├── nodes_vs_time_scatter.png
├── nodes_vs_path_cost_scatter.png
├── pr_Infinity_cw_0.6_bs_10000/
│   ├── config.json
│   ├── results.csv
│   └── ...
└── pr_0.3_cw_0.6_bs_10000/
    ├── config.json
    ├── results.csv
    └── ...
```

## Key Output Files

### `config.json`

Records puzzle/model metadata, search/eval options, heuristic/benchmark summary metrics, and `runtime_provenance` (`jax_version`, `jax_backend`, `jax_devices`, and Git revision plus dirty state for JAxtar, PuXle, and Xtructure — recording `N/A` if Git metadata is unavailable).

### `results.csv`

One row per evaluated puzzle sample. Key columns include:

- `seed`: Seed or benchmark sample ID.
- `solved`: Whether a solution was found.
- `search_time_s`: Synchronized wall-clock search time in seconds.
- `nodes_generated`: Generated/explored node count (or beam slots for beam search).
- `node_metric_label`: Metric label (`"Nodes Generated"` or `"Beam Slots"`).
- `path_cost`: Cost of the found solution (`0` if unsolved).
- `path_state_count`, `path_action_count`: Number of states and actions along the path.
- `benchmark_optimal_path_cost`, `benchmark_optimal_action_count`: Known optimal references when using an exact benchmark.
- `matches_optimal_path`: Whether the found solution matches the benchmark optimal reference.

## Visualizations

- **Path Cost Distribution** (`path_cost_distribution.png`): Histogram of solved path costs with mean and median lines.
- **Search Time vs. Path Cost** (`search_time_by_path_cost.png`): Boxplot of search time grouped by path cost, overlaid with a dashed median pointplot.
- **Nodes Generated vs. Path Cost** (`nodes_generated_by_path_cost.png`): Boxplot of generated nodes grouped by path cost, overlaid with a dashed median pointplot.
- **Heuristic Accuracy** (`heuristic_accuracy.png`): Boxplot and median curve of estimated distance vs. actual cost-to-go with $R^2$ and $\rho_c$ (CCC).
- **Benchmark Comparison** (`benchmark_path_comparison.png`): Scatter plots comparing found vs. optimal cost and action count.

## Comparing Runs

Use `benchmark compare` to compare completed runs or all sub-runs in a sweep directory:

```bash
uv run benchmark compare runs/run_A runs/run_B
uv run benchmark compare runs/my_sweep_dir
```

Multiple run paths write individual comparison PNGs (`path_cost_comparison.png`, `search_time_comparison.png`, `nodes_generated_comparison.png`, `nodes_vs_time_scatter.png`, `nodes_vs_path_cost_scatter.png`) to `runs/comparison_<timestamp>/`; passing a single sweep directory writes them directly into that directory.

## Metrics Glossary

- **Success Rate**: Percentage of evaluated instances solved within budget (`max_node_size`, `batch_size`).
- **Optimal Rate**: Percentage of solved instances matching the benchmark optimal reference.
- **Avg Nodes Generated**: Average search effort (`generated_size`: hash occupancy for eager search, expansions × branching factor for deferred search).
- **Avg Time**: Mean wall-clock search time per instance.
- **Cost Gap**: Mean difference between found path cost and benchmark optimal cost.
