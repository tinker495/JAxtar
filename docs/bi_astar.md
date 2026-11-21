# Bidirectional A\* Command (`bi_astar`)

Runs JIT-compiled Bidirectional A\* search simultaneously forward from the start state and backward from the goal state against a shared hash table. Requires puzzles implementing `batched_get_inverse_neighbours`.

## Usage

```bash
uv run bi_astar [OPTIONS]
uv run bi_astar -p rubikscube -nn
uv run bi_astar --help
```

`bi_astar` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit defaults (`puzzle="n-puzzle"`, puzzle/search presets). Note that `-h` means `--hard`; use `--help` for the full option catalog.

## Key Semantics

- **`-m, --max-node-size`** (default `2M`): Combined capacity of the shared hash table across both directions, not a per-direction budget.
- **`-w, --cost-weight`** (default `0.6`): Weights path cost in `f(n) = w * g(n) + h(n)` for each direction's priority queue.
- **`-pr, --pop-ratio`** (default `inf`): Expands popped candidates scoring within `best_score * (1 + pop_ratio)` (subject to `min_pop`); `inf` pops full batches.
- **Artifacts**: `--profile` writes a `jax.profiler` trace to `tmp/tensorboard`; `-vi` / `--visualize-imgs` saves step PNGs and `animation.gif` under `tmp/<puzzle>_<timestamp>/`.

## Related Commands

```bash
uv run benchmark bi-astar --benchmark rubikscube-deepcubea [OPTIONS]
uv run distance_train heuristic --eval-search-metric bi_astar
```
