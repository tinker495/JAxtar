import jax.numpy as jnp
import pytest
from xtructure import FieldDescriptor, xtructure_dataclass

from heuristic.neuralheuristic.davi import _compute_diffusion_distance
from qfunction.neuralq.qlearning import _compute_diffusion_q


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

    out = _compute_diffusion_distance(
        solve_configs=solve_configs,
        states=states,
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

    out = _compute_diffusion_q(
        solve_configs=solve_configs,
        states=states,
        trajectory_actions=trajectory_actions,
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
