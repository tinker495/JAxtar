import inspect

import jax
import jax.numpy as jnp
import pytest
import xtructure.numpy as xnp
from puxle import SlidePuzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from config import SEARCH_ALGORITHM_CATALOG
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.search_base import Parent, Parant_with_Costs, SearchResult


def _one_move_from_goal(puzzle: SlidePuzzle):
    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    goal_batch = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_batch, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    return solve_config, goal, neighbours[action, 0]


def _build_deferred_search_result(
    puzzle: SlidePuzzle,
) -> tuple[SearchResult, SlidePuzzle.SolveConfig]:
    solve_config, goal, start = _one_move_from_goal(puzzle)
    search_result = SearchResult.build(
        puzzle.State,
        puzzle.action_size,
        64,
        puzzle.action_size,
        parant_with_costs=True,
        emit_workload_signature=True,
    )

    search_result.hashtable, _, start_hash = search_result.hashtable.insert(start)
    search_result.cost = search_result.cost.at[start_hash.index].set(0)
    search_result.hashtable, _, goal_hash = search_result.hashtable.insert(goal)
    search_result.cost = search_result.cost.at[goal_hash.index].set(1)

    start_batch = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], start)
    _, ncosts = puzzle.batched_get_neighbours(solve_config, start_batch, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    keys = jnp.where(valid, jnp.zeros((puzzle.action_size,), dtype=KEY_DTYPE), jnp.inf)
    vals = Parant_with_Costs(
        parent=Parent(
            hashidx=xnp.tile(start_hash, (puzzle.action_size,)),
            action=jnp.arange(puzzle.action_size, dtype=ACTION_DTYPE),
        ),
        cost=jnp.zeros((puzzle.action_size,), dtype=KEY_DTYPE),
        dist=jnp.zeros((puzzle.action_size,), dtype=KEY_DTYPE),
    )
    search_result.priority_queue = search_result.priority_queue.insert(keys, vals)
    return search_result, solve_config


def test_pop_full_with_actions_tracks_deferred_lookup_counters_without_heuristic():
    puzzle = SlidePuzzle(size=2)
    search_result, solve_config = _build_deferred_search_result(puzzle)

    search_result, _, _, _ = search_result.pop_full_with_actions(
        puzzle=puzzle,
        solve_config=solve_config,
        use_heuristic=False,
    )

    assert int(jax.device_get(search_result.xtr_ht_lookup)) > 0
    assert int(jax.device_get(search_result.xtr_ht_found)) > 0


def test_pop_full_with_actions_tracks_deferred_lookup_counters_with_heuristic():
    puzzle = SlidePuzzle(size=2)
    search_result, solve_config = _build_deferred_search_result(puzzle)

    search_result, _, _, _ = search_result.pop_full_with_actions(
        puzzle=puzzle,
        solve_config=solve_config,
        use_heuristic=True,
    )

    assert int(jax.device_get(search_result.xtr_ht_lookup)) > 0
    assert int(jax.device_get(search_result.xtr_ht_found)) > 0


def test_astar_d_builder_accepts_emit_workload_signature():
    signature = inspect.signature(astar_d_builder)
    assert "spec" in signature.parameters


@pytest.mark.parametrize(
    "entry",
    [entry for entry in SEARCH_ALGORITHM_CATALOG if not entry.supports_workload_signature],
    ids=lambda entry: entry.python_id,
)
def test_non_emitting_catalog_builders_reject_workload_signature(entry):
    with pytest.raises(ValueError, match="emit_workload_signature"):
        entry.builder_fn(
            object(),
            object(),
            1,
            1,
            SearchBuildSpec(emit_workload_signature=True),
        )
