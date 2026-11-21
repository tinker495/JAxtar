# Q-Function Training Command (`distance_train qfunction`)

Trains a neural Q-function model to estimate action values $Q(s, a)$ for Q\* and Q-Beam search.

## Usage

```bash
uv run distance_train qfunction [OPTIONS]
uv run distance_train qfunction -p rubikscube -pre qlearning -s 10000
uv run distance_train qfunction --help
```

`distance_train qfunction` uses **tyro**: both `--kebab-case` and `--snake_case` work, numeric flags accept `10M` / `2^20` suffixes, and omitted options inherit defaults (`puzzle="rubikscube"`, `hard=True` internally, `--optimizer muon`, and the chosen `--preset`). Use `--help` for the full option catalog.

## Key Semantics

- **Presets (`-pre, --preset`)**: `qlearning` (default), `quality`, `largebatch`, `diffusion_distance`, `diffusion_distance_warmup_qlearning`.
- **Target generation (`--label`)**:
  - `td` (default, Q-learning): Bootstrap targets min-capped by diffusion distances only on rows where the sampled action matches the trajectory action.
  - `diffusion`: Trajectory Bellman propagation directly without bootstrap targets.
  - `warmup_td`: Uses diffusion targets for the first `--warmup-ratio` (default `0.2`) fraction of steps, then switches to `td`.
- **Multi-device scaling (`-md, --multi-device`, default `True`)**: `-ui, --update-interval` and `-fui, --force-update-interval` are single-device-equivalent outer intervals. Across `n_devices > 1`, hard-update cadence and outer `steps` are divided by `n_devices`, and soft updates use the compounded Polyak rate `1 - (1 - 1 / update_interval) ** n_devices`.
- **Training evaluation**: Controlled via `-ec, --eval-count` and `--eval-search-metric` (`qstar`, `bi_qstar`, `qbeam`).
