import jax
from puxle import SlidePuzzle

from heuristic.empty_heuristic import EmptyHeuristic
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.astar_d import astar_d_builder


def test_astar_d_warmup_preserves_loop_carry_cost_dtype():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    start = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    search_fn = astar_d_builder(
        puzzle,
        EmptyHeuristic(puzzle),
        batch_size=8,
        max_nodes=64,
        spec=SearchBuildSpec(warmup_inputs=(solve_config, start)),
    )

    assert callable(search_fn)
