import jax.numpy as jnp
import pytest
from xtructure import FieldDescriptor, xtructure_dataclass

from heuristic.neuralheuristic.target_dataset_builder import _compute_diffusion_distance
from qfunction.neuralq.target_dataset_builder import _compute_diffusion_q


@xtructure_dataclass
class SolveConfigsAndStates:
    solveconfigs: FieldDescriptor.scalar(dtype=jnp.int32)
    states: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass
class SolveConfigsAndStatesAndActions:
    solveconfigs: FieldDescriptor.scalar(dtype=jnp.int32)
    states: FieldDescriptor.scalar(dtype=jnp.int32)
    actions: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,))


def test_diffusion_distance_inverse_parent_uses_parent_indexed_action_costs():
    """Inverse-trajectory sampling uses parent indices behind the child; action_costs are parent-aligned.

    We build a chain goal->s1->s2->s3 with *parent indices behind*:
      parent(0) = -1
      parent(1) = 0
      parent(2) = 1
      parent(3) = 2
    Costs are stored on the parent row:
      cost(0->1)=5 stored at idx0, cost(1->2)=1 stored at idx1, cost(2->3)=1 stored at idx2
    """
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([10, 11, 12, 13], dtype=jnp.int32)
    parent_indices = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)
    is_solved = jnp.array([True, False, False, False], dtype=jnp.bool_)

    # Parent-aligned edge costs (note: last entry is unused)
    action_costs = jnp.array([5.0, 1.0, 1.0, 999.0], dtype=jnp.float32)
    move_costs = jnp.array([0.0, 100.0, 100.0, 100.0], dtype=jnp.float32)

    out = _compute_diffusion_distance(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=10,
    )
    assert out.tolist() == pytest.approx([0.0, 5.0, 6.0, 7.0])


def test_diffusion_distance_hindsight_parent_uses_child_aligned_action_costs():
    """Hindsight sampling uses parent indices ahead of the child; action_costs are child-aligned.

    Chain s0->s1->s2->goal where parent points forward (ahead), and action_costs live on the child row.
      parent(0)=1, parent(1)=2, parent(2)=3, parent(3)=-1
      cost(0->1)=2 stored at idx0, cost(1->2)=1 stored at idx1, cost(2->3)=4 stored at idx2
    """
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([20, 21, 22, 23], dtype=jnp.int32)
    parent_indices = jnp.array([1, 2, 3, -1], dtype=jnp.int32)
    is_solved = jnp.array([False, False, False, True], dtype=jnp.bool_)
    action_costs = jnp.array([2.0, 1.0, 4.0, 999.0], dtype=jnp.float32)
    move_costs = jnp.array([100.0, 100.0, 100.0, 0.0], dtype=jnp.float32)

    out = _compute_diffusion_distance(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=10,
    )
    assert out.tolist() == pytest.approx([7.0, 5.0, 4.0, 0.0])


def test_diffusion_q_inverse_parent_uses_parent_indexed_action_costs():
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([30, 31, 32, 33], dtype=jnp.int32)
    actions = jnp.array([[0], [0], [0], [0]], dtype=jnp.uint8)
    parent_indices = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)
    is_solved = jnp.array([True, False, False, False], dtype=jnp.bool_)
    action_costs = jnp.array([[5.0], [1.0], [1.0], [999.0]], dtype=jnp.float32)
    move_costs = jnp.array([0.0, 100.0, 100.0, 100.0], dtype=jnp.float32)

    out = _compute_diffusion_q(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        trajectory_actions=actions,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStatesAndActions=SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=10,
    )
    assert out.reshape(-1).tolist() == pytest.approx([0.0, 5.0, 6.0, 7.0])


def test_diffusion_distance_inverse_parent_supports_child_aligned_action_costs_when_goal_excluded():
    """Inverse sampling can exclude the solved state; then action_costs may be child-aligned even if parent is behind.

    Dataset rows represent states {A,B,C} with A duplicated. Parent pointers go "back" (behind), but the edge
    costs are stored on the child row:
      parent(A*) = -1
      parent(B) = A2
      parent(C) = B
    """
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([1, 1, 2, 3], dtype=jnp.int32)  # A1, A2(dup), B, C
    parent_indices = jnp.array([-1, -1, 1, 2], dtype=jnp.int32)
    is_solved = jnp.array([False, False, False, False], dtype=jnp.bool_)

    # Child-aligned edge costs to the parent.
    action_costs = jnp.array([1.0, 50.0, 100.0, 1.0], dtype=jnp.float32)
    # Path costs consistent with child alignment: B=100+A2(50)=150, C=1+B(150)=151.
    move_costs = jnp.array([1.0, 50.0, 150.0, 151.0], dtype=jnp.float32)

    out = _compute_diffusion_distance(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=10,
    )
    assert out.tolist() == pytest.approx([1.0, 1.0, 101.0, 102.0])


def test_diffusion_q_inverse_parent_supports_child_aligned_action_costs_when_goal_excluded():
    solve_configs = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    states = jnp.array([1, 1, 2, 3], dtype=jnp.int32)
    actions = jnp.array([[0], [0], [0], [0]], dtype=jnp.uint8)
    parent_indices = jnp.array([-1, -1, 1, 2], dtype=jnp.int32)
    is_solved = jnp.array([False, False, False, False], dtype=jnp.bool_)
    action_costs = jnp.array([[1.0], [50.0], [100.0], [1.0]], dtype=jnp.float32)
    move_costs = jnp.array([1.0, 50.0, 150.0, 151.0], dtype=jnp.float32)

    out = _compute_diffusion_q(
        solve_configs=solve_configs,
        states=states,
        is_solved=is_solved,
        trajectory_actions=actions,
        move_costs=move_costs,
        action_costs=action_costs,
        parent_indices=parent_indices,
        SolveConfigsAndStatesAndActions=SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates=SolveConfigsAndStates,
        k_max=10,
    )
    assert out.reshape(-1).tolist() == pytest.approx([1.0, 1.0, 101.0, 102.0])
