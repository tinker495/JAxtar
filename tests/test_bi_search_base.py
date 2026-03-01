import jax
import jax.numpy as jnp
from puxle import SlidePuzzle
from xtructure import HashIdx

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.bi_stars.bi_search_base import (
    MeetingPoint,
    build_bi_search_result,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
)


def test_import():
    assert build_bi_search_result is not None


def test_build_bi_search_result_initializes_forward_and_backward_consistently():
    puzzle = SlidePuzzle(size=2)
    batch_size = 8
    max_nodes = 32

    result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=batch_size,
        max_nodes=max_nodes,
        action_size=puzzle.action_size,
    )

    assert result.batch_size == batch_size
    assert result.forward.capacity == max_nodes
    assert result.backward.capacity == max_nodes
    assert int(result.forward.generated_size) == 0
    assert int(result.backward.generated_size) == 0

    assert bool(result.meeting.found) is False
    assert jnp.isinf(result.meeting.total_cost)


def test_initialize_bi_loop_common_sets_meeting_when_start_equals_goal():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    start = goal

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    assert bool(jax.device_get(bi_result.meeting.found)) is True
    assert float(jax.device_get(bi_result.meeting.total_cost)) == 0.0
    assert float(jax.device_get(bi_result.meeting.fwd_cost)) == 0.0
    assert float(jax.device_get(bi_result.meeting.bwd_cost)) == 0.0


def test_materialize_meeting_point_hashidxs_materializes_edge_only_meeting():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    start = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    _, fwd_current, _, _, _, _ = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)
    start_hashidx = fwd_current.hashidx[0]

    start_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], start)
    filled_b = jnp.array([True])
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, start_b, filled_b)
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    step_cost = ncosts[action, 0].astype(KEY_DTYPE)
    meeting_state = neighbours[action, 0]

    bi_result.backward.hashtable, _, meeting_bwd_hashidx = bi_result.backward.hashtable.insert(
        meeting_state
    )
    bi_result.backward.cost = bi_result.backward.cost.at[meeting_bwd_hashidx.index].set(
        jnp.array(0, dtype=KEY_DTYPE)
    )

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)
    bi_result.meeting = MeetingPoint(
        fwd_hashidx=dummy_hashidx,
        bwd_hashidx=meeting_bwd_hashidx,
        fwd_cost=step_cost,
        bwd_cost=jnp.array(0, dtype=KEY_DTYPE),
        total_cost=step_cost,
        found=jnp.array(True),
        fwd_has_hashidx=jnp.array(False),
        bwd_has_hashidx=jnp.array(True),
        fwd_parent_hashidx=start_hashidx,
        fwd_parent_action=jnp.array(action, dtype=ACTION_DTYPE),
        bwd_parent_hashidx=dummy_hashidx,
        bwd_parent_action=dummy_action,
    )

    bi_result = materialize_meeting_point_hashidxs(bi_result, puzzle, solve_config)

    assert bool(jax.device_get(bi_result.meeting.fwd_has_hashidx)) is True
    assert bool(jax.device_get(bi_result.meeting.bwd_has_hashidx)) is True

    fwd_hidx, found = bi_result.forward.hashtable.lookup(meeting_state)
    assert bool(jax.device_get(found)) is True
    assert int(jax.device_get(fwd_hidx.index)) == int(
        jax.device_get(bi_result.meeting.fwd_hashidx.index)
    )


def test_materialize_meeting_point_hashidxs_materializes_backward_edge_only_meeting():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    start = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    (
        _,
        fwd_current,
        _,
        _,
        bwd_current,
        bwd_states,
    ) = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    goal = jax.tree_util.tree_map(lambda x: x[0], bwd_states)
    goal_hashidx = bwd_current.hashidx[0]

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    filled_b = jnp.array([True])
    inv_neighbours, inv_costs = puzzle.batched_get_inverse_neighbours(
        inverse_solveconfig,
        goal_b,
        filled_b,
    )
    valid = jnp.isfinite(inv_costs[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    step_cost = inv_costs[action, 0].astype(KEY_DTYPE)
    meeting_state = inv_neighbours[action, 0]

    bi_result.forward.hashtable, _, meeting_fwd_hashidx = bi_result.forward.hashtable.insert(
        meeting_state
    )
    bi_result.forward.cost = bi_result.forward.cost.at[meeting_fwd_hashidx.index].set(
        jnp.array(0, dtype=KEY_DTYPE)
    )

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)
    bi_result.meeting = MeetingPoint(
        fwd_hashidx=meeting_fwd_hashidx,
        bwd_hashidx=dummy_hashidx,
        fwd_cost=jnp.array(0, dtype=KEY_DTYPE),
        bwd_cost=step_cost,
        total_cost=step_cost,
        found=jnp.array(True),
        fwd_has_hashidx=jnp.array(True),
        bwd_has_hashidx=jnp.array(False),
        fwd_parent_hashidx=dummy_hashidx,
        fwd_parent_action=dummy_action,
        bwd_parent_hashidx=goal_hashidx,
        bwd_parent_action=jnp.array(action, dtype=ACTION_DTYPE),
    )

    bi_result = materialize_meeting_point_hashidxs(
        bi_result, puzzle, solve_config, inverse_solveconfig=inverse_solveconfig
    )

    assert bool(jax.device_get(bi_result.meeting.fwd_has_hashidx)) is True
    assert bool(jax.device_get(bi_result.meeting.bwd_has_hashidx)) is True

    bwd_hidx, found = bi_result.backward.hashtable.lookup(meeting_state)
    assert bool(jax.device_get(found)) is True
    assert int(jax.device_get(bwd_hidx.index)) == int(
        jax.device_get(bi_result.meeting.bwd_hashidx.index)
    )
