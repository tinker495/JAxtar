"""Behaviour tests for the JAxtar trajectory-dataset adapter."""

from __future__ import annotations

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
