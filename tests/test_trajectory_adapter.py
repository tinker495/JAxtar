"""Architecture + behaviour guard for the JAxtar trajectory-dataset adapter.

Locks the contracts documented in CONTEXT.md "Puzzle Trajectory Module":

- PuxLe's `puxle.core.trajectory` is the single source of truth for the three
  creators (`create_target_shuffled_path`, `create_hindsight_target_shuffled_path`,
  `create_hindsight_target_triangular_shuffled_path`).
- `JAxtar/train_util/sampling.py` MUST NOT redeclare those creators.
- The JAxtar-owned adapter
  `JAxtar/train_util/trajectory_dataset_adapter.py` exposes the three names
  with the same signatures but returns the flat `dict[str, chex.Array]` shape
  the JIT'd training step consumes.
- The dict keys produced by `trajectory_to_dataset_dict` match the legacy
  schema so neural heuristic / Q-function dataset builders do not change at
  the call site.
"""

from __future__ import annotations

from pathlib import Path


_SAMPLING_PATH = Path(__file__).resolve().parents[1] / "train_util" / "sampling.py"
_EXPECTED_DICT_KEYS = frozenset(
    {
        "solve_configs",
        "states",
        "move_costs",
        "move_costs_tm1",
        "actions",
        "action_costs",
        "parent_indices",
        "trajectory_indices",
        "step_indices",
    }
)


def test_jaxtar_does_not_redeclare_trajectory_creators():
    source = _SAMPLING_PATH.read_text()
    for fn in (
        "def create_target_shuffled_path",
        "def create_hindsight_target_shuffled_path",
        "def create_hindsight_target_triangular_shuffled_path",
    ):
        assert fn not in source, (
            f"`train_util/sampling.py` must not redeclare `{fn[4:]}`; the "
            "canonical implementation lives in `puxle.core.trajectory`. JAxtar "
            "callers import the dict-shape wrapper from "
            "`train_util.trajectory_dataset_adapter` instead."
        )


def test_adapter_re_exports_three_creators():
    from train_util import trajectory_dataset_adapter as adapter

    for name in (
        "trajectory_to_dataset_dict",
        "create_target_shuffled_path",
        "create_hindsight_target_shuffled_path",
        "create_hindsight_target_triangular_shuffled_path",
    ):
        assert callable(getattr(adapter, name, None)), (
            f"`trajectory_dataset_adapter.{name}` must be callable so existing "
            "neural-training imports keep working."
        )


def test_trajectory_to_dataset_dict_preserves_legacy_keys():
    from puxle.core.trajectory import PuzzleTrajectory
    from train_util.trajectory_dataset_adapter import trajectory_to_dataset_dict

    sentinel = object()
    traj = PuzzleTrajectory(
        solve_configs=sentinel,
        states=sentinel,
        move_costs=sentinel,
        move_costs_tm1=sentinel,
        actions=sentinel,
        action_costs=sentinel,
        parent_indices=sentinel,
        trajectory_indices=sentinel,
        step_indices=sentinel,
    )
    out = trajectory_to_dataset_dict(traj)
    assert isinstance(out, dict)
    assert set(out.keys()) == _EXPECTED_DICT_KEYS, (
        "Adapter dict keys must match the legacy schema so heuristic / Q-function "
        "dataset builders keep working without call-site changes."
    )
    assert all(
        v is sentinel for v in out.values()
    ), "Adapter must pass record fields through without mutation."


def test_neural_builders_consume_adapter():
    """Heuristic and Q-function dataset builders must import the dict-shape
    creators from the adapter, not from `train_util.sampling`.
    """
    import heuristic.neuralheuristic.target_dataset_builder as hbuilder
    import qfunction.neuralq.target_dataset_builder as qbuilder
    from train_util.trajectory_dataset_adapter import (
        create_hindsight_target_shuffled_path,
        create_hindsight_target_triangular_shuffled_path,
        create_target_shuffled_path,
    )

    for builder in (hbuilder, qbuilder):
        assert builder.create_target_shuffled_path is create_target_shuffled_path
        assert (
            builder.create_hindsight_target_shuffled_path is create_hindsight_target_shuffled_path
        )
        assert (
            builder.create_hindsight_target_triangular_shuffled_path
            is create_hindsight_target_triangular_shuffled_path
        )
