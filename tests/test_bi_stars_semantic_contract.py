from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from puxle import SlidePuzzle

from heuristic.empty_heuristic import EmptyHeuristic
from JAxtar.bi_stars.bi_astar import _bi_astar_loop_builder
from JAxtar.bi_stars.bi_astar_d import _bi_astar_d_loop_builder, bi_astar_d_builder
from JAxtar.bi_stars.bi_qstar import _bi_qstar_loop_builder, bi_qstar_builder
from JAxtar.bi_stars.bi_search_base import (
    BiLoopState,
    BiLoopStateWithStates,
    build_bi_search_result,
    initialize_bi_loop_common,
)
from JAxtar.stars.astar import _astar_loop_builder
from JAxtar.stars.search_base import Current, SearchResult
from qfunction.q_base import QFunction


def _make_valid_slide_solve_config(puzzle: SlidePuzzle):
    size = puzzle.size
    board = jnp.concatenate(
        [
            jnp.arange(1, size**2, dtype=jnp.uint8),
            jnp.array([0], dtype=jnp.uint8),
        ],
        axis=0,
    )
    goal = puzzle.State.default().set_unpacked(board=board)
    solve_config = puzzle.SolveConfig(TargetState=goal)
    return solve_config, goal


class _ZeroQFunction(QFunction):
    def q_value(self, q_parameters, current):
        return jnp.zeros((self.puzzle.action_size,), dtype=jnp.float32)


def test_bi_astar_d_builder_does_not_build_bi_search_result_outside_compile():
    puzzle = SlidePuzzle(size=3)
    heuristic = EmptyHeuristic(puzzle)

    with patch("JAxtar.bi_stars.bi_astar_d.build_bi_search_result", side_effect=RuntimeError):
        with patch(
            "JAxtar.bi_stars.bi_astar_d.compile_search_builder",
            side_effect=lambda fn, _puzzle, _show_compile_time, _warmup_inputs: fn,
        ):
            search_fn = bi_astar_d_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    assert callable(search_fn)


def test_bi_qstar_builder_does_not_build_bi_search_result_outside_compile():
    puzzle = SlidePuzzle(size=3)
    q_fn = _ZeroQFunction(puzzle)

    with patch("JAxtar.bi_stars.bi_qstar.build_bi_search_result", side_effect=RuntimeError):
        with patch(
            "JAxtar.bi_stars.bi_qstar.compile_search_builder",
            side_effect=lambda fn, _puzzle, _show_compile_time, _warmup_inputs: fn,
        ):
            search_fn = bi_qstar_builder(
                puzzle, q_fn, batch_size=8, max_nodes=64, backward_mode="dijkstra"
            )

    assert callable(search_fn)


def test_initialize_bi_loop_common_padding_costs_are_inf_when_unfilled():
    puzzle = SlidePuzzle(size=3)
    solve_config = puzzle.SolveConfig.default()
    start = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    (
        fwd_filled,
        fwd_current,
        _,
        bwd_filled,
        bwd_current,
        _,
    ) = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    assert bool(jax.device_get(fwd_filled[0])) is True
    assert bool(jax.device_get(bwd_filled[0])) is True
    assert jnp.isinf(fwd_current.cost[~fwd_filled]).all()
    assert jnp.isinf(bwd_current.cost[~bwd_filled]).all()


def test_bi_astar_forward_key_formula_matches_astar_on_empty_heuristic():
    puzzle = SlidePuzzle(size=3)
    heuristic = EmptyHeuristic(puzzle)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start = neighbours[action, 0]

    batch_size = 8
    max_nodes = 128
    cost_weight = 1.0

    astar_init, _, astar_body = _astar_loop_builder(
        puzzle,
        heuristic,
        batch_size=batch_size,
        max_nodes=max_nodes,
        pop_ratio=jnp.inf,
        cost_weight=cost_weight,
    )
    bi_init, _, bi_body = _bi_astar_loop_builder(
        puzzle,
        heuristic,
        batch_size=batch_size,
        max_nodes=max_nodes,
        pop_ratio=jnp.inf,
        cost_weight=cost_weight,
        use_backward_heuristic=False,
        terminate_on_first_solution=False,
    )

    astar_state = astar_init(solve_config, start)
    bi_state0 = bi_init(solve_config, start)
    bi_state = BiLoopState(
        bi_result=bi_state0.bi_result,
        solve_config=bi_state0.solve_config,
        inverse_solveconfig=bi_state0.inverse_solveconfig,
        params_forward=bi_state0.params_forward,
        params_backward=bi_state0.params_backward,
        current_forward=bi_state0.current_forward,
        current_backward=bi_state0.current_backward,
        filled_forward=bi_state0.filled_forward,
        filled_backward=jnp.zeros_like(bi_state0.filled_backward),
    )

    def _pop_full_noop(self, **kwargs):
        batch_size = self.batch_size
        return self, Current.default((batch_size,)), jnp.zeros((batch_size,), dtype=jnp.bool_)

    with patch.object(SearchResult, "pop_full", _pop_full_noop):
        astar_state_out = astar_body(astar_state)
        bi_state_out = bi_body(bi_state)

    pq_astar = astar_state_out.search_result.priority_queue
    pq_bi = bi_state_out.bi_result.forward.priority_queue

    assert bool(jax.device_get(jnp.isfinite(pq_astar.key_store).any())) is True
    assert jnp.allclose(pq_astar.key_store, pq_bi.key_store)
    assert jnp.allclose(pq_astar.val_store.cost, pq_bi.val_store.cost)


def test_bi_astar_d_forward_key_uses_lookahead_costs_when_look_ahead_pruning_true():
    puzzle = SlidePuzzle(size=3)
    heuristic = EmptyHeuristic(puzzle)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start = neighbours[action, 0]

    init_loop_state, _, loop_body = _bi_astar_d_loop_builder(
        puzzle,
        heuristic,
        batch_size=8,
        max_nodes=128,
        pop_ratio=jnp.inf,
        cost_weight=1.0,
        look_ahead_pruning=True,
        use_backward_heuristic=False,
        terminate_on_first_solution=False,
    )
    loop_state0 = init_loop_state(solve_config, start)
    loop_state = BiLoopStateWithStates(
        bi_result=loop_state0.bi_result,
        solve_config=loop_state0.solve_config,
        inverse_solveconfig=loop_state0.inverse_solveconfig,
        params_forward=loop_state0.params_forward,
        params_backward=loop_state0.params_backward,
        current_forward=loop_state0.current_forward,
        current_backward=loop_state0.current_backward,
        states_forward=loop_state0.states_forward,
        states_backward=loop_state0.states_backward,
        filled_forward=loop_state0.filled_forward,
        filled_backward=jnp.zeros_like(loop_state0.filled_backward),
    )

    def _pop_full_with_actions_noop(
        self,
        *,
        puzzle,
        solve_config,
        use_heuristic: bool = False,
        is_backward: bool = False,
        **kwargs,
    ):
        batch_size = self.batch_size
        current = Current.default((batch_size,))
        states = puzzle.State.default((batch_size,))
        filled = jnp.zeros((batch_size,), dtype=jnp.bool_)
        return self, current, states, filled

    with patch.object(SearchResult, "pop_full_with_actions", _pop_full_with_actions_noop):
        loop_state_out = loop_body(loop_state)

    key_store = loop_state_out.bi_result.forward.priority_queue.key_store
    finite_keys = key_store[jnp.isfinite(key_store)]
    assert finite_keys.size > 0
    assert jnp.allclose(finite_keys, jnp.array(1.0))


def test_bi_qstar_key_formula_matches_qstar_forward_direction():
    puzzle = SlidePuzzle(size=3)
    q_fn = _ZeroQFunction(puzzle)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start = neighbours[action, 0]

    init_loop_state, _, loop_body = _bi_qstar_loop_builder(
        puzzle,
        q_fn,
        batch_size=8,
        max_nodes=128,
        pop_ratio=jnp.inf,
        cost_weight=1.0,
        look_ahead_pruning=True,
        pessimistic_update=True,
        use_backward_q=False,
        backward_mode="dijkstra",
        terminate_on_first_solution=False,
    )
    loop_state0 = init_loop_state(solve_config, start)
    loop_state = BiLoopStateWithStates(
        bi_result=loop_state0.bi_result,
        solve_config=loop_state0.solve_config,
        inverse_solveconfig=loop_state0.inverse_solveconfig,
        params_forward=loop_state0.params_forward,
        params_backward=loop_state0.params_backward,
        current_forward=loop_state0.current_forward,
        current_backward=loop_state0.current_backward,
        states_forward=loop_state0.states_forward,
        states_backward=loop_state0.states_backward,
        filled_forward=loop_state0.filled_forward,
        filled_backward=jnp.zeros_like(loop_state0.filled_backward),
    )

    def _pop_full_with_actions_noop(
        self,
        *,
        puzzle,
        solve_config,
        use_heuristic: bool = False,
        is_backward: bool = False,
        **kwargs,
    ):
        batch_size = self.batch_size
        current = Current.default((batch_size,))
        states = puzzle.State.default((batch_size,))
        filled = jnp.zeros((batch_size,), dtype=jnp.bool_)
        return self, current, states, filled

    with patch.object(SearchResult, "pop_full_with_actions", _pop_full_with_actions_noop):
        loop_state_out = loop_body(loop_state)

    key_store = loop_state_out.bi_result.forward.priority_queue.key_store
    finite_keys = key_store[jnp.isfinite(key_store)]
    assert finite_keys.size > 0
    assert jnp.allclose(finite_keys, jnp.array(0.0))


def test_bi_qstar_safety_guard_matches_plan_constraints():
    class _DummyPuzzle:
        action_size = 4
        State = object

    class _DummyQFunction:
        is_fixed = False

    with pytest.raises(ValueError) as exc:
        bi_qstar_builder(
            _DummyPuzzle(),
            _DummyQFunction(),
            batch_size=8,
            max_nodes=16,
            terminate_on_first_solution=False,
            backward_mode="value_v",
        )

    assert "unsafe_allow_nonadmissible=True" in str(exc.value)
