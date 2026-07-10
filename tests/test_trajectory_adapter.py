"""Behaviour tests for the JAxtar trajectory-dataset adapter."""

from __future__ import annotations

import jax.numpy as jnp

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


def test_world_model_consumers_only_adapt_puxle_trajectories():
    from puxle.core.trajectory import PuzzleTrajectory
    from puxle.puzzles.slidepuzzle import SlidePuzzle
    from world_model_puzzle import world_model_ds

    state_cls = SlidePuzzle(size=3).State
    transition_traj = PuzzleTrajectory(
        solve_configs=None,
        states=state_cls(board=jnp.arange(40, dtype=jnp.uint8).reshape(4, 2, 5)),
        move_costs=None,
        move_costs_tm1=None,
        actions=jnp.arange(6).reshape(3, 2),
        action_costs=None,
    )
    eval_traj = PuzzleTrajectory(
        solve_configs=None,
        states=state_cls(board=jnp.arange(20, dtype=jnp.uint8).reshape(4, 1, 5)),
        move_costs=None,
        move_costs_tm1=None,
        actions=jnp.arange(3).reshape(3, 1),
        action_costs=None,
    )
    key = jnp.array([0, 1], dtype=jnp.uint32)
    calls = []

    class PuzzleStub:
        def batched_get_random_trajectory(self, k_max, shuffle_parallel, call_key):
            calls.append((k_max, shuffle_parallel, call_key))
            return transition_traj if shuffle_parallel == 2 else eval_traj

    puzzle = PuzzleStub()
    states, actions, next_states = world_model_ds.create_shuffled_path(puzzle, 3, 2, 5, key)
    eval_states, eval_actions = world_model_ds.create_eval_trajectory(puzzle, 3, key)

    assert jnp.array_equal(states.board, transition_traj.states.board[:-1].reshape(6, 5)[:5])
    assert jnp.array_equal(actions, transition_traj.actions.reshape(6)[:5])
    assert jnp.array_equal(next_states.board, transition_traj.states.board[1:].reshape(6, 5)[:5])
    assert jnp.array_equal(eval_states.board, eval_traj.states.board.reshape(4, 5))
    assert jnp.array_equal(eval_actions, eval_traj.actions.reshape(3))
    assert [(k_max, parallel) for k_max, parallel, _ in calls] == [(3, 2), (3, 1)]
    assert all(jnp.array_equal(call_key, key) for _, _, call_key in calls)
