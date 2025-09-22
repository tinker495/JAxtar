# Hydra Workflow

Hydra now powers configuration for JAxtar experiments. Both distance-based training pipelines (`distance_train davi` and `distance_train qlearning`) are available through the new launcher while the original Click CLI continues to work.

## Entry Point

Use `hydra_runner.py` as the new launcher:

```bash
# Inspect the composed configuration without running training
python hydra_runner.py --cfg job --resolve

# Run DAVI training with the default preset (defaults to task=davi)
python hydra_runner.py

# Run Q-learning with the default preset
python hydra_runner.py task=qlearning
```

Hydra stores outputs under `outputs/<date>/<time>` and the legacy logger continues to write inside that working directory (e.g., `runs/rubikscube-dist-train_<timestamp>`).

## Configuration Structure

The configuration tree lives in `conf/` and follows Hydra's standard layout:

- `conf/config.yaml`: top-level defaults (`task: davi`).
- `conf/task/davi.yaml`: bundles the pieces needed for DAVI training.
- `conf/task/qlearning.yaml`: bundles the pieces needed for Q-learning.
- `conf/puzzle/*.yaml`: puzzle selection and overrides (size, hardness, shuffle length, etc.). All bundles from `config/puzzle_registry.py` now have matching YAML entries so you can switch puzzles via `task.puzzle=<name>`.
- `conf/heuristic/neural.yaml`: neural heuristic overrides (parameter path, per-model tweaks).
- `conf/qfunction/neural.yaml`: neural Q-function overrides (parameter path, per-model tweaks).
- `conf/q_options/*.yaml`: options that control Q-learning specific features such as policy sampling.
- `conf/train/dist/*.yaml`: distance-training hyper-parameters (mirrors the old presets).
- `conf/eval/*.yaml`: evaluation sweep options.

You can override any field from the command line using standard Hydra syntax. Example:

```bash
# Use the "quality" preset and disable evaluation
python hydra_runner.py \
  task.train=dist/quality \
  task.eval=disabled \
  task.puzzle.shuffle_length=40 \
  task.heuristic.param_path=/tmp/rubik_params.pkl

# Run Q-learning with policy disabled and a custom parameter snapshot
python hydra_runner.py \
  task=qlearning \
  task.q_options.with_policy=false \
  task.qfunction.param_path=/tmp/rubik_q_params.pkl
```

## Legacy Interoperability

`hydra_runner.py` bridges the YAML configs to the existing training stack by reusing the refactored `run_davi_training` and `run_qlearning_training` helpers. The original Click commands remain functional (`python main.py distance_train davi ...` and `python main.py distance_train qlearning ...`), so you can adopt Hydra incrementally.

## Next Steps for Full Migration

- Port the remaining training flows (world-model commands) to Hydra groups and entry points.
- Expose search/evaluation-only workflows via Hydra so sweeps can be orchestrated from YAML.
- Deduplicate option-parsing logic by extracting shared builders from `cli/options.py` into reusable helpers (Hydra already uses `hydra_support/builders.py`).
- Provide additional config groups for logger backends, dataset generation, and puzzle registry entries.
- Once coverage is complete, consider simplifying or deprecating the Click CLI in favour of pure Hydra launches.
