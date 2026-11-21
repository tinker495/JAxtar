"""Adapts PuXle ``PuzzleTrajectory`` records to JAxtar training shapes; wrappers preserve PuXle's positional parameter order for ``partial(...)`` callers."""

from __future__ import annotations

from typing import TypedDict

import chex
from puxle.core.trajectory import (
    PuzzleTrajectory,
    create_hindsight_target_shuffled_path as _create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path as _create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path as _create_target_shuffled_path,
)
from xtructure import Xtructurable


class TrajectoryDataset(TypedDict):
    solve_configs: Xtructurable
    states: Xtructurable
    move_costs: chex.Array
    move_costs_tm1: chex.Array
    actions: chex.Array
    action_costs: chex.Array
    parent_indices: chex.Array | None
    trajectory_indices: chex.Array | None
    step_indices: chex.Array | None


def trajectory_to_dataset_dict(traj: PuzzleTrajectory) -> TrajectoryDataset:
    if not isinstance(traj.solve_configs, Xtructurable) or not isinstance(
        traj.states, Xtructurable
    ):
        raise TypeError("Trajectory solve_configs and states must be Xtructurable values")
    return {
        "solve_configs": traj.solve_configs,
        "states": traj.states,
        "move_costs": traj.move_costs,
        "move_costs_tm1": traj.move_costs_tm1,
        "actions": traj.actions,
        "action_costs": traj.action_costs,
        "parent_indices": traj.parent_indices,
        "trajectory_indices": traj.trajectory_indices,
        "step_indices": traj.step_indices,
    }


def trajectory_to_transition_dataset(
    traj: PuzzleTrajectory, limit: int
) -> tuple[Xtructurable, chex.Array, Xtructurable]:
    return (
        traj.states[:-1].flatten()[:limit],
        traj.actions.reshape(-1)[:limit],
        traj.states[1:].flatten()[:limit],
    )


def trajectory_to_eval_trajectory(
    traj: PuzzleTrajectory,
) -> tuple[Xtructurable, chex.Array]:
    """Remove the singleton parallel axis from an evaluation trajectory."""
    return traj.states.flatten(), traj.actions.reshape(-1)


def create_target_shuffled_path(
    puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> TrajectoryDataset:
    return trajectory_to_dataset_dict(
        _create_target_shuffled_path(
            puzzle,
            k_max,
            shuffle_parallel,
            include_solved_states,
            key,
            non_backtracking_steps=non_backtracking_steps,
        )
    )


def create_hindsight_target_shuffled_path(
    puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> TrajectoryDataset:
    return trajectory_to_dataset_dict(
        _create_hindsight_target_shuffled_path(
            puzzle,
            k_max,
            shuffle_parallel,
            include_solved_states,
            key,
            non_backtracking_steps=non_backtracking_steps,
        )
    )


def create_hindsight_target_triangular_shuffled_path(
    puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> TrajectoryDataset:
    return trajectory_to_dataset_dict(
        _create_hindsight_target_triangular_shuffled_path(
            puzzle,
            k_max,
            shuffle_parallel,
            include_solved_states,
            key,
            non_backtracking_steps=non_backtracking_steps,
        )
    )


__all__ = [
    "trajectory_to_dataset_dict",
    "trajectory_to_transition_dataset",
    "trajectory_to_eval_trajectory",
    "create_target_shuffled_path",
    "create_hindsight_target_shuffled_path",
    "create_hindsight_target_triangular_shuffled_path",
]
