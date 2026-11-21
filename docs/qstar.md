# Q\* Command (`qstar`)

Runs JIT-compiled batched Q\* search, queuing `(parent, action)` edges scored from parent Q-values (`f = w * g(parent) + Q(parent, action)`) and deferring child generation until pop.

## Usage

```bash
uv run qstar [OPTIONS]
uv run qstar -p rubikscube -nn
uv run qstar --help
```

`qstar` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit defaults (`puzzle="n-puzzle"`, puzzle/search presets). Note that `-h` means `--hard`; use `--help` for the full option catalog.

## Key Semantics

- **`-w, --cost-weight`** (default `0.6`): Weights path cost in `f = w * g + Q(s, a)`. Values `< 1.0` make search greedier; `1.0` restores unweighted Q\*.
- **`-pr, --pop-ratio`** (default `inf`): Expands popped candidates scoring within `best_score * (1 + pop_ratio)` (subject to `min_pop`); `inf` disables ratio filtering and pops full batches.
- **`--emit-workload-signature`**: Records `xtr_*` hash-table and priority-queue counters (supported only on `astar`, `astar_d`, and `qstar`).
- **Artifacts**: `--profile` writes a `jax.profiler` trace to `tmp/tensorboard`; `-vi` / `--visualize-imgs` saves step PNGs and `animation.gif` under `tmp/<puzzle>_<timestamp>/`.
