import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.beamsearch.q_beam import qbeam_builder
from qfunction.q_base import QFunction


class _UnitQFunction(QFunction):
    def q_value(self, q_parameters, current):
        return jnp.ones((self.puzzle.action_size,), dtype=jnp.float32)


def test_qbeam_warmup_preserves_loop_carry_dist_dtype():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    start = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    search_fn = qbeam_builder(
        puzzle,
        _UnitQFunction(puzzle),
        batch_size=8,
        max_nodes=64,
        spec=SearchBuildSpec(warmup_inputs=(solve_config, start)),
    )

    assert callable(search_fn)
