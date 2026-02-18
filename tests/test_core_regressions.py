from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xtructure.numpy as xnp
from puxle import SlidePuzzle

import JAxtar.core.expansion as expansion_mod
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.annotate import KEY_DTYPE
from JAxtar.bi_stars.bi_astar import bi_astar_builder
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.core.expansion import DeferredExpansion
from JAxtar.core.result import Current, SearchResult
from JAxtar.core.scoring import AStarScoring


def _get_goal_state(puzzle, solve_config):
    if hasattr(puzzle, "get_goal_state"):
        return puzzle.get_goal_state(solve_config)
    if hasattr(solve_config, "TargetState"):
        return solve_config.TargetState
    raise AttributeError("Unable to obtain goal state for trivial start==goal regression test.")


@pytest.mark.parametrize("builder", [bi_astar_builder, bi_astar_d_builder])
def test_bidirectional_start_equals_goal_returns_zero_cost(builder):
    """Regression: bidirectional builders must terminate immediately on start==goal."""
    puzzle = SlidePuzzle(size=2)
    heuristic = SlidePuzzleHeuristic(puzzle)
    solve_config, _ = puzzle.get_inits(jax.random.PRNGKey(0))
    goal_state = _get_goal_state(puzzle, solve_config)

    search_fn = builder(
        puzzle=puzzle,
        heuristic=heuristic,
        batch_size=64,
        max_nodes=50_000,
        cost_weight=0.6,
        show_compile_time=False,
    )
    result = search_fn(solve_config, goal_state)

    assert bool(result.meeting.found)
    total_cost = float(jnp.ravel(result.meeting.total_cost)[0])
    assert total_cost == 0.0


class _LayoutCaptureRaised(Exception):
    pass


class _StubSearchResult:
    action_size = 3
    batch_size = 2

    def get_state(self, current):
        del current
        return jnp.array([11.0, 22.0], dtype=KEY_DTYPE)


def _capture_dist_for_no_lookahead(monkeypatch, call_fn):
    captured = {}

    def _capture_and_abort(priority, vals, optimal_mask, action_size, batch_size):
        del priority, optimal_mask, action_size, batch_size
        captured["dist"] = np.asarray(jax.device_get(vals.dist))
        raise _LayoutCaptureRaised

    monkeypatch.setattr(expansion_mod, "sort_and_pack_action_candidates", _capture_and_abort)

    with pytest.raises(_LayoutCaptureRaised):
        call_fn()

    return captured["dist"]


def test_deferred_expand_no_lookahead_uses_action_major_heuristic_layout(monkeypatch):
    """Regression: deferred no-lookahead must tile parent heuristic in action-major order."""
    expansion = DeferredExpansion(
        scoring_policy=AStarScoring(),
        heuristic_fn=lambda params, states, mask: jnp.where(mask, states, jnp.inf).astype(
            KEY_DTYPE
        ),
        look_ahead_pruning=False,
    )

    current = SimpleNamespace(
        hashidx=jnp.array([0, 1], dtype=jnp.uint32),
        cost=jnp.array([1.0, 2.0], dtype=KEY_DTYPE),
    )
    filled = jnp.array([True, True], dtype=jnp.bool_)
    expected = np.array([11.0, 22.0, 11.0, 22.0, 11.0, 22.0], dtype=np.float32)

    dist = _capture_dist_for_no_lookahead(
        monkeypatch,
        lambda: expansion.expand(
            search_result=_StubSearchResult(),
            puzzle=None,
            solve_config=None,
            heuristic_params=None,
            current=current,
            filled=filled,
        ),
    )
    np.testing.assert_allclose(dist, expected)


def test_deferred_expand_bi_no_lookahead_uses_action_major_heuristic_layout(
    monkeypatch,
):
    """Regression: bidirectional deferred no-lookahead path must preserve action-major mapping."""
    expansion = DeferredExpansion(
        scoring_policy=AStarScoring(),
        heuristic_fn=lambda params, states, mask: jnp.where(mask, states, jnp.inf).astype(
            KEY_DTYPE
        ),
        look_ahead_pruning=False,
    )

    current = SimpleNamespace(
        hashidx=jnp.array([0, 1], dtype=jnp.uint32),
        cost=jnp.array([1.0, 2.0], dtype=KEY_DTYPE),
    )
    filled = jnp.array([True, True], dtype=jnp.bool_)
    expected = np.array([11.0, 22.0, 11.0, 22.0, 11.0, 22.0], dtype=np.float32)

    dist = _capture_dist_for_no_lookahead(
        monkeypatch,
        lambda: expansion.expand_bi(
            search_result=_StubSearchResult(),
            opposite_search_result=SimpleNamespace(),
            meeting_point=SimpleNamespace(),
            puzzle=None,
            solve_config=None,
            heuristic_params=None,
            current=current,
            filled=filled,
            is_forward=True,
        ),
    )
    np.testing.assert_allclose(dist, expected)


def test_pop_full_masks_stale_current_entries():
    """Regression: eager pop_full should not process stale PQ entries."""
    puzzle = SlidePuzzle(size=2)
    _, start_state = puzzle.get_inits(jax.random.PRNGKey(0))

    search_result = SearchResult.build(
        statecls=puzzle.State,
        batch_size=4,
        max_nodes=256,
        action_size=puzzle.action_size,
        pq_val_type=Current,
    )
    search_result.hashtable, _, hash_idx = search_result.hashtable.insert(start_state)
    search_result.cost = search_result.cost.at[hash_idx.index].set(jnp.array(1.0, dtype=KEY_DTYPE))

    stale_hashidxs = xnp.pad(hash_idx, (0, search_result.batch_size - 1))
    stale_costs = jnp.full((search_result.batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(5.0)
    stale_current = Current(hashidx=stale_hashidxs, cost=stale_costs)
    stale_keys = jnp.full((search_result.batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(5.0)
    stale_masks = jnp.array([True, False, False, False], dtype=jnp.bool_)
    search_result = search_result.insert_batch(stale_keys, stale_current, stale_masks)

    _, _, process_mask = search_result.pop_full()
    assert not bool(jnp.any(process_mask))
