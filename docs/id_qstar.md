# Iterative Deepening Q\* Command (`id_qstar`)

Runs JIT-compiled batched Iterative Deepening Q\* (ID-Q\*), performing bounded depth-first expansions from a batched initial frontier using parent Q-values for `f = w * g + Q(s, a)` without per-node heuristic caching.

## Usage

```bash
uv run id_qstar [OPTIONS]
uv run id_qstar -p rubikscube -nn
uv run id_qstar --help
```

`id_qstar` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit defaults (`puzzle="n-puzzle"`, puzzle/search presets). Note that `-h` means `--hard`; use `--help` for the full option catalog.

## Key Semantics

- **`--bound-step`** (default `1.0`): Rounds each iterative-deepening threshold up onto a grid of this size. Setting `0` restores the exact `float16` ladder, which under a real-valued Q-function can advance by a single `float16` ULP per pass and re-derive the search tree from the frontier each time.
- **`--max-path-len`** (default `256`): Per-node action-history length for the frontier and hard depth cutoff (`depth <= max_path_len`), dominating frontier/trace buffer sizing. Lower it only above the puzzle's solution depth; setting it too low silently turns deeper solvable instances into unsolved.
- **`-w, --cost-weight`** (default `0.6`): Weights path cost in `f = w * g(n) + Q(s, a)`.
- **Artifacts**: `--profile` writes a `jax.profiler` trace to `tmp/tensorboard`; `-vi` / `--visualize-imgs` saves step PNGs and `animation.gif` under `tmp/<puzzle>_<timestamp>/`.
