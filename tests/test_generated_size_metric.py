"""Generated-state metric must mean the same thing for eager and deferred search."""

import jax
from puxle import SlidePuzzle

from JAxtar.stars.search_base import SearchResult


def _build(puzzle: SlidePuzzle, *, deferred: bool) -> SearchResult:
    return SearchResult.build(
        puzzle.State,
        puzzle.action_size,
        64,
        puzzle.action_size,
        parant_with_costs=deferred,
        is_reversible=puzzle.is_reversible,
    )


def _insert_one(search_result: SearchResult, puzzle: SlidePuzzle) -> SearchResult:
    state = puzzle.solve_config_to_state_transform(
        puzzle.SolveConfig.default(), key=jax.random.PRNGKey(0)
    )
    search_result.hashtable, _, _ = search_result.hashtable.insert(state)
    return search_result


def test_eager_generated_size_is_hash_usage():
    puzzle = SlidePuzzle(size=2)
    search_result = _insert_one(_build(puzzle, deferred=False), puzzle)

    assert int(search_result.generated_size) == 1


def test_deferred_generated_size_scales_hash_usage_by_branching_factor():
    puzzle = SlidePuzzle(size=2)
    assert puzzle.is_reversible
    search_result = _insert_one(_build(puzzle, deferred=True), puzzle)

    assert int(search_result.hashtable.size) == 1
    assert int(search_result.generated_size) == puzzle.action_size - 1


def test_irreversible_deferred_generated_size_keeps_full_action_size():
    puzzle = SlidePuzzle(size=2)
    search_result = SearchResult.build(
        puzzle.State,
        puzzle.action_size,
        64,
        puzzle.action_size,
        parant_with_costs=True,
        is_reversible=False,
    )
    search_result = _insert_one(search_result, puzzle)

    assert int(search_result.generated_size) == puzzle.action_size
