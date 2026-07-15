import io
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import pytest
from puxle import PDDL, TSP
from xtructure import StructuredType

from helpers.search_utils import vmapping_init_target, vmapping_search


class _InitTargetPuzzle:
    def __init__(self):
        self.calls = 0
        self.keys = []

    def get_inits(self, key):
        self.keys.append(key)
        idx = self.calls
        self.calls += 1
        return jnp.array([idx], dtype=jnp.int32), jnp.array([idx + 10], dtype=jnp.int32)


class _VmapPuzzle:
    class State:
        @staticmethod
        def default(shape):
            return jnp.zeros(shape, dtype=jnp.int32)

    class SolveConfig:
        @staticmethod
        def default(shape):
            return jnp.zeros(shape, dtype=jnp.int32)


def _add_fn(solve_config, state):
    return solve_config + state


def test_vmapping_init_target_uses_seed_order_and_repeats_first_row():
    puzzle = _InitTargetPuzzle()
    solve_configs, states = vmapping_init_target(puzzle, vmap_size=4, start_state_seeds=[5, 20])

    assert puzzle.calls == 2
    assert all(
        jnp.array_equal(actual, jax.random.PRNGKey(seed))
        for actual, seed in zip(puzzle.keys, [5, 20])
    )
    assert solve_configs.shape == (4, 1)
    assert states.shape == (4, 1)

    # Explicit seeds fill the first rows; remaining rows reuse the first row.
    assert jnp.array_equal(solve_configs[:, 0], jnp.array([0, 1, 0, 0], dtype=jnp.int32))
    assert jnp.array_equal(states[:, 0], jnp.array([10, 11, 10, 10], dtype=jnp.int32))


@pytest.fixture(scope="module")
def pddl_puzzle():
    return PDDL.from_preset("blocksworld", problem_basename="bw-S-01")


@pytest.mark.parametrize("seeds", ([0], [0, 1]))
def test_vmapping_init_target_batches_real_pddl_context(pddl_puzzle, seeds):
    solve_configs, states = vmapping_init_target(pddl_puzzle, vmap_size=2, start_state_seeds=seeds)

    context = solve_configs.InstanceContext
    assert context.pre_mask.shape == (2, pddl_puzzle.num_actions, pddl_puzzle.num_atoms)
    assert solve_configs.GoalSpec.GoalMask.shape == (2, pddl_puzzle.num_atoms)

    next_states, costs = jax.jit(jax.vmap(pddl_puzzle.get_actions, in_axes=(0, 0, 0)))(
        solve_configs, states, jnp.zeros(2, dtype=jnp.int32)
    )
    assert next_states.batch_shape == (2,)
    assert costs.shape == (2,)


@pytest.mark.parametrize("seeds", ([0], [0, 1]))
def test_vmapping_init_target_batches_real_tsp_context(seeds):
    puzzle = TSP(size=4)
    solve_configs, states = vmapping_init_target(puzzle, vmap_size=2, start_state_seeds=seeds)

    context = solve_configs.InstanceContext
    assert context.distance_matrix.shape == (2, puzzle.size, puzzle.size)
    assert solve_configs.GoalSpec.structured_type == StructuredType.SINGLE

    next_states, costs = jax.jit(jax.vmap(puzzle.get_actions, in_axes=(0, 0, 0)))(
        solve_configs, states, jnp.ones(2, dtype=jnp.int32)
    )
    assert next_states.batch_shape == (2,)
    assert costs.shape == (2,)


def test_vmapping_search_caches_by_function_and_vmap_size():
    puzzle = _VmapPuzzle()

    vmapped_first = vmapping_search(puzzle, _add_fn, vmap_size=2)
    vmapped_cached = vmapping_search(puzzle, _add_fn, vmap_size=2)
    assert vmapped_first is vmapped_cached

    solve_configs = jnp.array([1, 2], dtype=jnp.int32)
    states = jnp.array([3, 4], dtype=jnp.int32)
    out = vmapped_first(solve_configs, states)
    assert out.shape == (2,)
    assert jnp.array_equal(out, jnp.array([4, 6], dtype=jnp.int32))


def test_vmapping_search_prints_compile_time_when_requested():
    puzzle = _VmapPuzzle()

    buf = io.StringIO()
    with redirect_stdout(buf):

        def _fresh_fn(solve_config, state):
            return solve_config + state

        vmapping_search(puzzle, _fresh_fn, vmap_size=1, show_compile_time=True)

    out = buf.getvalue()
    assert "initializing vmapped jit" in out
    assert "Compile Time" in out
    assert "JIT compiled" in out
