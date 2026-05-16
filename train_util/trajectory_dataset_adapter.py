"""JAxtar-owned adapter at the **Puzzle Trajectory Module** seam.

PuXle's ``puxle.core.trajectory`` is the single source of truth for
target-shuffled-path generation; it returns ``PuzzleTrajectory`` records.
JAxtar's neural heuristic and Q-function dataset builders need the flat
``dict[str, chex.Array]`` shape that their JIT'd training step consumes, so
this Module wraps each creator with ``trajectory_to_dataset_dict`` while
keeping PuXle agnostic of the dict shape.

The three wrapped names re-export the PuXle creators with the same parameter
order so call sites (``partial(create_target_shuffled_path, puzzle, ...)``)
remain unchanged when migrating off the legacy ``train_util.sampling``
duplicates.
"""

from __future__ import annotations

import chex
from puxle.core.trajectory import (
    PuzzleTrajectory,
    create_hindsight_target_shuffled_path as _create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path as _create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path as _create_target_shuffled_path,
)


def trajectory_to_dataset_dict(traj: PuzzleTrajectory) -> dict[str, chex.Array]:
    """Translate a ``PuzzleTrajectory`` into the flat dict shape consumed by
    JAxtar's JIT'd training step. Field names match the legacy dict keys.
    """
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


def create_target_shuffled_path(
    puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> dict[str, chex.Array]:
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
) -> dict[str, chex.Array]:
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
) -> dict[str, chex.Array]:
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
    "create_target_shuffled_path",
    "create_hindsight_target_shuffled_path",
    "create_hindsight_target_triangular_shuffled_path",
]
