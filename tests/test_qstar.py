import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from JAxtar.annotate import KEY_DTYPE
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFunction


class _UnitQFunction(QFunction):
    def q_value(self, q_parameters, current):
        return jnp.ones((self.puzzle.action_size,), dtype=jnp.float32)


def test_qstar_preserves_search_state_dtype():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.get_solve_config()
    start = puzzle.State.from_unpacked(board=jnp.array([1, 2, 0, 3], dtype=jnp.uint8))

    search_fn = qstar_builder(
        puzzle,
        _UnitQFunction(puzzle),
        batch_size=8,
        max_nodes=64,
        spec=SearchBuildSpec(warmup_inputs=(solve_config, start)),
    )
    result = search_fn(solve_config, start)

    assert bool(jax.device_get(result.solved))
    assert result.cost.dtype == KEY_DTYPE
    assert result.dist.dtype == KEY_DTYPE
