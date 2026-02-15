import jax.numpy as jnp
from xtructure import FieldDescriptor, xtructure_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.id_stars.id_frontier import ACTION_PAD, IDFrontier


@xtructure_dataclass
class _FlatBatch:
    state: FieldDescriptor.tensor(dtype=jnp.int32, shape=(2,))
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    depth: FieldDescriptor.scalar(dtype=jnp.int32)
    trail: FieldDescriptor.tensor(dtype=jnp.int32, shape=(1,))
    action_history: FieldDescriptor.tensor(dtype=ACTION_DTYPE, shape=(4,))
    action: FieldDescriptor.scalar(dtype=jnp.int32)
    parent_index: FieldDescriptor.scalar(dtype=jnp.int32)
    root_index: FieldDescriptor.scalar(dtype=jnp.int32)


def _build_frontier(batch_size: int, max_path_len: int) -> IDFrontier:
    states = jnp.arange(batch_size * 2, dtype=jnp.int32).reshape((batch_size, 2))
    costs = jnp.arange(batch_size, dtype=KEY_DTYPE)
    depths = jnp.arange(batch_size, dtype=jnp.int32)
    valid_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
    f_scores = jnp.arange(batch_size, dtype=KEY_DTYPE)
    trail = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    action_history = jnp.full((batch_size, max_path_len), ACTION_PAD, dtype=ACTION_DTYPE)
    solution_state = jnp.zeros((1, 2), dtype=jnp.int32)
    solution_cost = jnp.array(jnp.inf, dtype=KEY_DTYPE)
    solution_actions_arr = jnp.full((max_path_len,), ACTION_PAD, dtype=ACTION_DTYPE)
    return IDFrontier(
        states=states,
        costs=costs,
        depths=depths,
        valid_mask=valid_mask,
        f_scores=f_scores,
        trail=trail,
        action_history=action_history,
        solved=jnp.array(False),
        solution_state=solution_state,
        solution_cost=solution_cost,
        solution_actions_arr=solution_actions_arr,
    )


def _build_flat_batch(flat_size: int, max_path_len: int) -> _FlatBatch:
    return _FlatBatch(
        state=jnp.arange(flat_size * 2, dtype=jnp.int32).reshape((flat_size, 2)),
        cost=jnp.arange(flat_size, dtype=KEY_DTYPE),
        depth=jnp.arange(flat_size, dtype=jnp.int32),
        trail=jnp.zeros((flat_size, 1), dtype=jnp.int32),
        action_history=jnp.full((flat_size, max_path_len), ACTION_PAD, dtype=ACTION_DTYPE),
        action=jnp.zeros((flat_size,), dtype=jnp.int32),
        parent_index=jnp.zeros((flat_size,), dtype=jnp.int32),
        root_index=jnp.zeros((flat_size,), dtype=jnp.int32),
    )


def test_select_top_k_empty_candidates_returns_invalid_frontier():
    batch_size = 3
    max_path_len = 4
    flat_size = 6
    frontier = _build_frontier(batch_size, max_path_len)
    flat_batch = _build_flat_batch(flat_size, max_path_len)
    flat_valid = jnp.zeros((flat_size,), dtype=jnp.bool_)
    f_safe = jnp.array([9, 8, 7, 6, 5, 4], dtype=KEY_DTYPE)

    out = frontier.select_top_k(
        flat_batch=flat_batch,
        flat_valid=flat_valid,
        f_safe=f_safe,
        batch_size=batch_size,
        new_solved=jnp.array(True),
        new_sol_state=jnp.ones((1, 2), dtype=jnp.int32),
        new_sol_cost=jnp.array(1.0, dtype=KEY_DTYPE),
        new_sol_actions=jnp.full((max_path_len,), 1, dtype=ACTION_DTYPE),
    )

    assert out.valid_mask.tolist() == [False, False, False]
    assert jnp.all(jnp.isinf(out.f_scores))
    assert bool(out.solved) is True
    assert float(out.solution_cost) == 1.0
    assert out.states.tolist() == frontier.states.tolist()


def test_select_top_k_dense_candidates_matches_smallest_f_scores():
    batch_size = 3
    max_path_len = 4
    flat_size = 6
    frontier = _build_frontier(batch_size, max_path_len)
    flat_batch = _build_flat_batch(flat_size, max_path_len)
    flat_valid = jnp.ones((flat_size,), dtype=jnp.bool_)
    f_safe = jnp.array([5.0, 1.0, 4.0, 0.5, 3.0, 2.0], dtype=KEY_DTYPE)

    out = frontier.select_top_k(
        flat_batch=flat_batch,
        flat_valid=flat_valid,
        f_safe=f_safe,
        batch_size=batch_size,
        new_solved=jnp.array(False),
        new_sol_state=frontier.solution_state,
        new_sol_cost=frontier.solution_cost,
        new_sol_actions=frontier.solution_actions_arr,
    )

    expected_idx = jnp.argsort(f_safe)[:batch_size]
    assert out.valid_mask.tolist() == [True, True, True]
    assert out.states.tolist() == flat_batch.state[expected_idx].tolist()
    assert out.costs.tolist() == flat_batch.cost[expected_idx].tolist()
    assert out.depths.tolist() == flat_batch.depth[expected_idx].tolist()
