# Beam Search Command (`beam`)

Runs JIT-compiled batched Beam Search without hash tables or priority queues, retaining only a fixed-width frontier at each depth for predictable memory usage.

## Usage

```bash
uv run beam [OPTIONS]
uv run beam -p rubikscube -nn -b 10000
uv run beam --help
```

`beam` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted flags inherit defaults (`puzzle="n-puzzle"`, puzzle/search presets). Note that `-h` means `--hard`; use `--help` for the full option catalog.

## Key Semantics

- **`-b, --batch-size`** (default `10000`): Sets the **Beam Width** (and caps depth at `ceil(max_node_size / beam_width)`).
- **`-w, --cost-weight`** (default `0.6`): Weights path cost in `f(n) = w * g(n) + h(n)`.
- **`-pr, --pop-ratio`** (default `inf`): Applies multiplicative slack `best_score * pop_ratio` when pruning the beam (preserving at least `min_keep` valid candidates up to `beam_width`); `inf` keeps the top `beam_width`.
- **Artifacts**: `--profile` writes a `jax.profiler` trace to `tmp/tensorboard`; `-vi` / `--visualize-imgs` saves step PNGs and `animation.gif` under `tmp/<puzzle>_<timestamp>/`.
