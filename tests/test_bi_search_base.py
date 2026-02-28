import jax.numpy as jnp
from puxle import SlidePuzzle

from JAxtar.bi_stars.bi_search_base import build_bi_search_result


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
