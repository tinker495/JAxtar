import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from heuristic.empty_heuristic import EmptyHeuristic
from JAxtar.annotate import KEY_DTYPE
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.astar_d import astar_d_builder


def test_astar_d_warmup_preserves_loop_carry_cost_dtype():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.get_solve_config()
    start = puzzle.State.from_unpacked(board=jnp.array([1, 2, 0, 3], dtype=jnp.uint8))

    search_fn = astar_d_builder(
        puzzle,
        EmptyHeuristic(puzzle),
        batch_size=8,
        max_nodes=64,
        spec=SearchBuildSpec(warmup_inputs=(solve_config, start)),
    )

    result = search_fn(solve_config, start)

    assert bool(jax.device_get(result.solved))
    assert result.cost.dtype == KEY_DTYPE
    assert result.dist.dtype == KEY_DTYPE
