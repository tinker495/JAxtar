# A\* Deferred Command (`astar_d`)

Runs JIT-compiled batched A\* Deferred search, queuing `(parent, action)` edges and delaying child state generation and heuristic evaluation until edges are popped. Deferring reduces wasted work when heuristics are expensive or branching factors are large.

## Usage

```bash
uv run astar_d [OPTIONS]
uv run astar_d -p rubikscube -nn
uv run astar_d --help
```

`astar_d` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit defaults (`puzzle="n-puzzle"`, puzzle/search presets). Note that `-h` means `--hard`; use `--help` for the full option catalog.

## Key Semantics

- **`-w, --cost-weight`** (default `0.6`): Weights path cost in `f(n) = w * g(n) + h(n)`. Values `< 1.0` make search greedier; `1.0` restores unweighted search.
- **`-pr, --pop-ratio`** (default `inf`): Expands popped candidates scoring within `best_score * (1 + pop_ratio)` (subject to `min_pop`); `inf` disables ratio filtering and pops full batches.
- **`--emit-workload-signature`**: Records `xtr_*` hash-table and priority-queue counters (supported only on `astar`, `astar_d`, and `qstar`).
- **Artifacts**: `--profile` writes a `jax.profiler` trace to `tmp/tensorboard`; `-vi` / `--visualize-imgs` saves step PNGs and `animation.gif` under `tmp/<puzzle>_<timestamp>/`.
