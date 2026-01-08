import jax.numpy as jnp
import pytest
from xtructure import FieldDescriptor, xtructure_dataclass

from heuristic.neuralheuristic.target_dataset_builder import _compute_diffusion_distance
from qfunction.neuralq.target_dataset_builder import _compute_diffusion_q
from train_util.sampling import flatten_scanned_paths


@xtructure_dataclass
class SolveConfigsAndStates:
    solveconfigs: FieldDescriptor.scalar(dtype=jnp.int32)
    states: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass
class SolveConfigsAndStatesAndActions:
    solveconfigs: FieldDescriptor.scalar(dtype=jnp.int32)
    states: FieldDescriptor.scalar(dtype=jnp.int32)
    actions: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,))


def test_diffusion_distance_overwrites_duplicate_child_with_shorter_path():
    """Regression test: if a duplicated child state improves, all duplicates must be overwritten.

    Construct two copies of the same child state B. Only one copy (B1) has a valid parent path to
    the goal; the other (B2) is the one referenced by A. Correct diffusion must re-collapse
    duplicates each iteration so A can benefit from B1's improvement.
    """
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([1, 2, 2, 3], dtype=jnp.int32)  # A, B1, B2 (duplicate), C(goal-ish)

    move_costs = jnp.array([100.0, 100.0, 100.0, 0.0], dtype=jnp.float32)
    action_costs = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    parent_indices = jnp.array([2, 3, -1, -1], dtype=jnp.int32)  # A->B2, B1->C
    is_solved = jnp.array([False, False, False, True], dtype=bool)  # Only state C is at goal

    out = _compute_diffusion_distance(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=4,
    )

    assert out.shape == (4,)
    assert out[1] == pytest.approx(1.0)  # B1 becomes 1 via C
    assert out[2] == pytest.approx(1.0)  # B2 overwritten from B1 (duplicate collapse)
    assert out[0] == pytest.approx(2.0)  # A = 1 + min(B) (must see B1 improvement)


def test_diffusion_q_overwrites_duplicate_child_state_action_with_shorter_path():
    """Same idea as heuristic diffusion, but for (state,action) diffusion Q."""
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([1, 2, 2, 3], dtype=jnp.int32)  # A, B1, B2, C
    trajectory_actions = jnp.array([[0], [0], [0], [0]], dtype=jnp.uint8)

    move_costs = jnp.array([100.0, 100.0, 100.0, 0.0], dtype=jnp.float32)
    action_costs = jnp.array([[1.0], [1.0], [1.0], [0.0]], dtype=jnp.float32)
    parent_indices = jnp.array([2, 3, -1, -1], dtype=jnp.int32)  # A->B2, B1->C
    is_solved = jnp.array([False, False, False, True], dtype=bool)  # Only state C is at goal

    out = _compute_diffusion_q(
        solve_configs=solve_configs,
        states=states,
        trajectory_actions=trajectory_actions,
        is_solved=is_solved,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStatesAndActions=SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=4,
    )

    assert out.shape == (4, 1)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[2, 0] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(2.0)


def test_flatten_scanned_paths_offsets_parent_indices_to_prevent_cross_chunk_shortcuts():
    """Regression: when concatenating multiple path chunks, parent_indices must be offset per chunk.

    Without the offset, a later chunk can incorrectly point into an earlier chunk, creating an
    artificial shortcut and producing labels smaller than the true trajectory cost.
    """
    steps = 2
    chunk_size = 3
    dataset_size = steps * chunk_size

    paths = {
        "solve_configs": jnp.array([[0, 0, 0], [1, 1, 1]], dtype=jnp.int32),
        "states": jnp.array([[10, 11, 12], [20, 21, 22]], dtype=jnp.int32),
        "move_costs": jnp.array([[2.0, 1.0, 0.0], [20.0, 10.0, 0.0]], dtype=jnp.float32),
        "action_costs": jnp.array([[1.0, 1.0, 0.0], [10.0, 10.0, 0.0]], dtype=jnp.float32),
        "parent_indices": jnp.array([[1, 2, -1], [1, 2, -1]], dtype=jnp.int32),
        "is_solved": jnp.array([[False, False, True], [False, False, True]], dtype=bool),
    }

    # Old behavior: flatten without offsetting per-scan-step indices.
    naive = {k: v.flatten()[:dataset_size] for k, v in paths.items()}
    out_naive = _compute_diffusion_distance(
        solve_configs=naive["solve_configs"],
        states=naive["states"],
        is_solved=naive["is_solved"],
        move_costs=naive["move_costs"],
        action_costs=naive["action_costs"],
        parent_indices=naive["parent_indices"],
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=8,
    )
    assert out_naive[3] == pytest.approx(11.0)  # Incorrect shortcut into chunk-0.

    # Fixed behavior: offset parent_indices for each chunk before flattening.
    fixed = flatten_scanned_paths(paths, dataset_size)
    assert fixed["parent_indices"].tolist() == [1, 2, -1, 4, 5, -1]
    out_fixed = _compute_diffusion_distance(
        solve_configs=fixed["solve_configs"],
        states=fixed["states"],
        is_solved=fixed["is_solved"],
        move_costs=fixed["move_costs"],
        action_costs=fixed["action_costs"],
        parent_indices=fixed["parent_indices"],
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=8,
    )
    assert out_fixed[3] == pytest.approx(20.0)  # No cross-chunk contamination.


def test_flatten_scanned_paths_offset_avoids_underestimating_diffusion_q():
    steps = 2
    chunk_size = 3
    dataset_size = steps * chunk_size

    paths = {
        "solve_configs": jnp.array([[0, 0, 0], [1, 1, 1]], dtype=jnp.int32),
        "states": jnp.array([[10, 11, 12], [20, 21, 22]], dtype=jnp.int32),
        "move_costs": jnp.array([[2.0, 1.0, 0.0], [20.0, 10.0, 0.0]], dtype=jnp.float32),
        "actions": jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.uint8),
        "action_costs": jnp.array([[1.0, 1.0, 0.0], [10.0, 10.0, 0.0]], dtype=jnp.float32),
        "parent_indices": jnp.array([[1, 2, -1], [1, 2, -1]], dtype=jnp.int32),
        "is_solved": jnp.array([[False, False, True], [False, False, True]], dtype=bool),
    }

    fixed = flatten_scanned_paths(paths, dataset_size)
    out = _compute_diffusion_q(
        solve_configs=fixed["solve_configs"],
        states=fixed["states"],
        is_solved=fixed["is_solved"],
        trajectory_actions=fixed["actions"].reshape((-1, 1)),
        move_costs=fixed["move_costs"],
        action_costs=fixed["action_costs"].reshape((-1, 1)),
        parent_indices=fixed["parent_indices"],
        SolveConfigsAndStatesAndActions=SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=8,
    )

    assert out.shape == (dataset_size, 1)
    assert out[3, 0] == pytest.approx(20.0)
