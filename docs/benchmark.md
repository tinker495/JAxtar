# Benchmark Command (`benchmark`)

Evaluates search algorithms on registered exact benchmark datasets or generated puzzle workloads, and compares completed runs.

## Usage

```bash
uv run benchmark <subcommand> [OPTIONS]
uv run benchmark qstar --benchmark rubikscube-deepcubea --sample-limit 100
uv run benchmark qstar --puzzle rubikscube --num-eval 200
uv run benchmark compare runs/run_A runs/run_B
uv run benchmark --help
```

`benchmark` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit preset defaults. Run `uv run benchmark <subcommand> --help` for full subcommand options.

## Subcommands

- **Heuristic search**: `astar`, `astar-d`, `bi-astar`, `bi-astar-d`, `id-astar`, `beam`
- **Q-function search**: `qstar`, `bi-qstar`, `id-qstar`, `qbeam`
- **Run comparison**: `compare`

## Key Semantics

- **Target mutual exclusion**: `--benchmark` (defaults to `rubikscube-deepcubea` when neither target is passed) and `-p, --puzzle` (generates hard puzzle instances) are mutually exclusive. `--benchmark-args`, `--sample-ids`, and `--sample-limit` require `--benchmark`; `--puzzle-args` requires `--puzzle`.
- **CLI sweeps**: `-pr, --pop-ratio` accepts comma-separated sweep lists on the CLI (e.g., `-pr inf,0.3,0.1`); `-w, --cost-weight` and `-b, --batch-size` are scalar CLI flags.
- **Output path**: `--output-dir` defaults to `runs/<puzzle>_<algo>_<timestamp>` (or `runs/<run_name>` when `--rn, --run-name` is set). See [**Benchmark Logging**](./benchmark_logging.md) for output schemas and plots.
