# Heuristic Training Command (`distance_train heuristic`)

Trains a neural heuristic model to estimate cost-to-go distances via Dynamic A\* Value Iteration (DAVI) or trajectory diffusion.

## Usage

```bash
uv run distance_train heuristic [OPTIONS]
uv run distance_train heuristic -p rubikscube -pre davi -s 10000 -db 1000 -dmb 200 -tmb 200
uv run distance_train heuristic --help
```

`distance_train heuristic` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted options inherit defaults (`puzzle="rubikscube"`, `hard=True` internally, `--optimizer muon`, and the chosen `--preset`). Use `--help` for the full option catalog.

## Key Semantics

- **Presets (`-pre, --preset`)**: `davi` (default), `quality`, `largebatch`, `diffusion_distance`, `diffusion_distance_warmup_davi`.
- **Target generation (`--label`)**:
  - `td` (default, DAVI): Bootstrap TD targets always min-capped by deduplicated trajectory (diffusion) distances to suppress overestimation.
  - `diffusion`: Trajectory Bellman propagation directly without bootstrap targets.
  - `warmup_td`: Uses diffusion targets for the first `--warmup-ratio` (default `0.2`) fraction of steps, then switches to `td`.
- **Multi-device scaling (`-md, --multi-device`, default `True`)**: `-ui, --update-interval` and `-fui, --force-update-interval` are single-device-equivalent outer intervals. Across `n_devices > 1`, hard-update cadence and outer `steps` are divided by `n_devices`, and soft updates use the compounded Polyak rate `1 - (1 - 1 / update_interval) ** n_devices`.
- **Training evaluation**: Controlled via `-ec, --eval-count` and `--eval-search-metric` (`astar`, `astar_d`, `bi_astar`, `bi_astar_d`, `beam`).
