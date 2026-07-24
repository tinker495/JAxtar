"""Heuristics must return float: the search pads batches with jnp.inf, which
cannot be cast to an integer dtype (OverflowError at trace time)."""

import jax
import jax.numpy as jnp
import pytest

from config.puzzle_registry import puzzle_bundles

# Bundles whose puzzle/heuristic needs config args or optional deps are covered
# by the plain-callable ones below; skip them instead of guessing constructors.
_BUNDLES = sorted(puzzle_bundles.items())


@pytest.mark.parametrize("name,bundle", _BUNDLES, ids=[n for n, _ in _BUNDLES])
def test_heuristic_returns_float(name, bundle):
    try:
        puzzle = bundle.puzzle()
        heuristic = bundle.heuristic(puzzle)
    except (TypeError, ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"{name} not directly constructible: {e}")

    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(0))
    out = jax.eval_shape(
        heuristic.distance, heuristic.prepare_heuristic_parameters(solve_config), state
    )
    assert jnp.issubdtype(out.dtype, jnp.floating), f"{name}: {out.dtype}"
