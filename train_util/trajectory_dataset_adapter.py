"""JAxtar-owned adapter at the **Puzzle Trajectory Module** seam.

PuXle's ``puxle.core.trajectory`` is the single source of truth for trajectory
generation. JAxtar only adapts ``PuzzleTrajectory`` records to training shapes.

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


def trajectory_to_transition_dataset(
    traj: PuzzleTrajectory, limit: int
) -> tuple[chex.ArrayTree, chex.Array, chex.ArrayTree]:
    """Flatten time/batch axes and truncate aligned transition triples."""
    return (
        traj.states[:-1].flatten()[:limit],
        traj.actions.reshape(-1)[:limit],
        traj.states[1:].flatten()[:limit],
    )


def trajectory_to_eval_trajectory(
    traj: PuzzleTrajectory,
) -> tuple[chex.ArrayTree, chex.Array]:
    """Remove the singleton parallel axis from an evaluation trajectory."""
    return traj.states.flatten(), traj.actions.reshape(-1)


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
    "trajectory_to_transition_dataset",
    "trajectory_to_eval_trajectory",
    "create_target_shuffled_path",
    "create_hindsight_target_shuffled_path",
    "create_hindsight_target_triangular_shuffled_path",
]
