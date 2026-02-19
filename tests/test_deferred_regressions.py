from types import SimpleNamespace

import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

import cli.search_runner as search_runner
import JAxtar.bi_stars.bi_astar as bi_astar_mod
import JAxtar.bi_stars.bi_astar_d as bi_astar_d_mod
import JAxtar.stars.astar as astar_mod
import JAxtar.stars.astar_d as astar_d_mod
from config.pydantic_models import SearchOptions, VisualizeOptions
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.annotate import MIN_BATCH_SIZE
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.slidepuzzle_q import SlidePuzzleQ


def test_deferred_search_variants_solve_n_puzzle_3_seed_0():
    """Regression: astar-d/qstar/bi-astar-d should solve n-puzzle-3 for seed 0."""
    puzzle = SlidePuzzle(size=3)
    heuristic = SlidePuzzleHeuristic(puzzle)
    q_function = SlidePuzzleQ(puzzle)
    solve_config, start_state = puzzle.get_inits(jax.random.PRNGKey(0))

    common_options = {
        "batch_size": 256,
        "max_nodes": 200_000,
        "cost_weight": 0.6,
        "show_compile_time": False,
    }

    astar_d = astar_d_builder(puzzle=puzzle, heuristic=heuristic, **common_options)
    astar_d_result = astar_d(solve_config, start_state)
    astar_d_cost = float(jnp.ravel(astar_d_result.get_cost(astar_d_result.solved_idx))[0])
    assert bool(astar_d_result.solved)
    assert astar_d_cost > 0.0

    qstar = qstar_builder(puzzle=puzzle, q_fn=q_function, **common_options)
    qstar_result = qstar(solve_config, start_state)
    qstar_cost = float(jnp.ravel(qstar_result.get_cost(qstar_result.solved_idx))[0])
    assert bool(qstar_result.solved)
    assert qstar_cost > 0.0

    bi_astar_d = bi_astar_d_builder(puzzle=puzzle, heuristic=heuristic, **common_options)
    bi_result = bi_astar_d(solve_config, start_state)
    bi_cost = float(jnp.ravel(bi_result.meeting.total_cost)[0])
    assert bool(bi_result.meeting.found)
    assert bi_cost > 0.0


class _DummyPuzzle:
    has_target = True

    def get_inits(self, key):
        del key
        return object(), jnp.array([0], dtype=jnp.int32)


class _DummySearchResult:
    solved = jnp.array(True)
    generated_size = jnp.array(12, dtype=jnp.int32)
    solved_idx = jnp.array([3, 5], dtype=jnp.int32)

    def get_cost(self, idx):
        del idx
        return jnp.array([7.5, 9.5], dtype=jnp.float32)


def test_search_samples_handles_non_scalar_solved_cost():
    """Regression: CLI result rendering should handle non-scalar solved-cost arrays."""
    captured = {"solved_cost": None}

    original_build_result_table = search_runner.build_result_table
    original_build_seed_setup_panel = search_runner.build_seed_setup_panel

    def _capture_result_table(**kwargs):
        captured["solved_cost"] = kwargs["solved_cost"]
        return ""

    def _capture_seed_setup_panel(**kwargs):
        del kwargs
        return ""

    try:
        search_runner.build_result_table = _capture_result_table
        search_runner.build_seed_setup_panel = _capture_seed_setup_panel

        search_runner.search_samples(
            search_fn=lambda solve_config, state: _DummySearchResult(),
            puzzle=_DummyPuzzle(),
            puzzle_name="dummy",
            dist_fn=lambda dist_params, state: jnp.array(0.0, dtype=jnp.float32),
            dist_fn_format=lambda puzzle, dist_values: "0.0",
            seeds=[0],
            search_options=SearchOptions(),
            visualize_options=VisualizeOptions(),
        )
    finally:
        search_runner.build_result_table = original_build_result_table
        search_runner.build_seed_setup_panel = original_build_seed_setup_panel

    assert captured["solved_cost"] == 7.5


class _BuilderDummySolveConfig:
    pass


class _BuilderDummyState:
    pass


class _BuilderDummyPuzzle:
    action_size = 4
    SolveConfig = _BuilderDummySolveConfig
    State = _BuilderDummyState


class _BuilderDummyHeuristic:
    def batched_distance(self, params, states):
        del params, states
        raise AssertionError("direct heuristic.batched_distance should not be used here")

    def prepare_heuristic_parameters(self, solve_config, **kwargs):
        del solve_config, kwargs
        return None


def test_bi_astar_d_builder_uses_masked_batch_switcher(monkeypatch):
    """Regression: bi_astar_d should use variable batch switcher instead of direct full-batch heuristic."""
    captured = {"switcher_calls": 0}

    def _fake_switcher_builder(*args, **kwargs):
        del args, kwargs

        def _switcher(params, states, mask):
            del params, states
            captured["switcher_calls"] += 1
            return jnp.where(mask, jnp.array(7.0, dtype=jnp.float32), jnp.inf)

        return _switcher

    def _fake_loop_builder(*args, **kwargs):
        fwd_expansion = args[1]
        bwd_expansion = args[2]
        captured["fwd_heuristic_fn"] = fwd_expansion.heuristic_fn
        captured["bwd_heuristic_fn"] = bwd_expansion.heuristic_fn

        def _init_loop_state(*inner_args, **inner_kwargs):
            del inner_args, inner_kwargs
            return SimpleNamespace(bi_result=SimpleNamespace())

        return _init_loop_state, (lambda _: False), (lambda loop_state: loop_state)

    monkeypatch.setattr(bi_astar_d_mod, "variable_batch_switcher_builder", _fake_switcher_builder)
    monkeypatch.setattr(bi_astar_d_mod, "unified_bi_search_loop_builder", _fake_loop_builder)
    monkeypatch.setattr(bi_astar_d_mod, "jit_with_warmup", lambda fn, **kwargs: fn)

    search_fn = bi_astar_d_mod.bi_astar_d_builder(
        puzzle=_BuilderDummyPuzzle(),
        heuristic=_BuilderDummyHeuristic(),
        batch_size=8,
        max_nodes=64,
    )
    assert callable(search_fn)

    states = jnp.arange(8, dtype=jnp.int32)
    mask = jnp.array([True, False, True, False, True, False, True, False])

    fwd_vals = captured["fwd_heuristic_fn"](None, states, mask)
    bwd_vals = captured["bwd_heuristic_fn"](None, states, mask)

    assert captured["switcher_calls"] == 2
    assert bool(jnp.all(jnp.isfinite(fwd_vals[mask])))
    assert bool(jnp.all(jnp.isinf(fwd_vals[~mask])))
    assert bool(jnp.all(jnp.isfinite(bwd_vals[mask])))
    assert bool(jnp.all(jnp.isinf(bwd_vals[~mask])))


def test_astar_builder_uses_masked_batch_switcher(monkeypatch):
    """Regression: astar should route heuristic evaluation through mask-aware switcher."""
    captured = {"switcher_calls": 0}

    def _fake_switcher_builder(*args, **kwargs):
        del args, kwargs

        def _switcher(params, states, mask):
            del params, states
            captured["switcher_calls"] += 1
            return jnp.where(mask, jnp.array(5.0, dtype=jnp.float32), jnp.inf)

        return _switcher

    def _fake_loop_builder(*args, **kwargs):
        expansion = args[1]
        captured["heuristic_fn"] = expansion.heuristic_fn

        def _init_loop_state(*inner_args, **inner_kwargs):
            del inner_args, inner_kwargs
            return SimpleNamespace(search_result=SimpleNamespace(), current=None, filled=None)

        return _init_loop_state, (lambda _: False), (lambda loop_state: loop_state)

    monkeypatch.setattr(astar_mod, "variable_batch_switcher_builder", _fake_switcher_builder)
    monkeypatch.setattr(astar_mod, "unified_search_loop_builder", _fake_loop_builder)
    monkeypatch.setattr(astar_mod, "jit_with_warmup", lambda fn, **kwargs: fn)

    search_fn = astar_mod.astar_builder(
        puzzle=_BuilderDummyPuzzle(),
        heuristic=_BuilderDummyHeuristic(),
        batch_size=8,
        max_nodes=64,
    )
    assert callable(search_fn)

    states = jnp.arange(8, dtype=jnp.int32)
    mask = jnp.array([True, False, True, False, True, False, True, False])
    values = captured["heuristic_fn"](None, states, mask)

    assert captured["switcher_calls"] == 1
    assert bool(jnp.all(jnp.isfinite(values[mask])))
    assert bool(jnp.all(jnp.isinf(values[~mask])))


def test_astar_d_builder_uses_masked_batch_switcher_and_action_scaled_min_pop(monkeypatch):
    """Regression: astar_d should use masked switcher and restored action-scaled min_pop."""
    captured = {"switcher_calls": 0, "min_pop": None}

    def _fake_switcher_builder(*args, **kwargs):
        del args, kwargs

        def _switcher(params, states, mask):
            del params, states
            captured["switcher_calls"] += 1
            return jnp.where(mask, jnp.array(6.0, dtype=jnp.float32), jnp.inf)

        return _switcher

    def _fake_loop_builder(*args, **kwargs):
        expansion = args[1]
        captured["heuristic_fn"] = expansion.heuristic_fn
        captured["min_pop"] = kwargs["min_pop"]

        def _init_loop_state(*inner_args, **inner_kwargs):
            del inner_args, inner_kwargs
            return SimpleNamespace(search_result=SimpleNamespace(), current=None, filled=None)

        return _init_loop_state, (lambda _: False), (lambda loop_state: loop_state)

    monkeypatch.setattr(astar_d_mod, "variable_batch_switcher_builder", _fake_switcher_builder)
    monkeypatch.setattr(astar_d_mod, "unified_search_loop_builder", _fake_loop_builder)
    monkeypatch.setattr(astar_d_mod, "jit_with_warmup", lambda fn, **kwargs: fn)

    search_fn = astar_d_mod.astar_d_builder(
        puzzle=_BuilderDummyPuzzle(),
        heuristic=_BuilderDummyHeuristic(),
        batch_size=8,
        max_nodes=64,
    )
    assert callable(search_fn)

    states = jnp.arange(8, dtype=jnp.int32)
    mask = jnp.array([True, False, True, False, True, False, True, False])
    values = captured["heuristic_fn"](None, states, mask)

    expected_min_pop = max(1, MIN_BATCH_SIZE // max(1, _BuilderDummyPuzzle.action_size // 2))
    assert captured["min_pop"] == expected_min_pop
    assert captured["switcher_calls"] == 1
    assert bool(jnp.all(jnp.isfinite(values[mask])))
    assert bool(jnp.all(jnp.isinf(values[~mask])))


def test_bi_astar_builder_uses_masked_batch_switcher(monkeypatch):
    """Regression: bi_astar should route both directions through mask-aware switcher."""
    captured = {"switcher_calls": 0}

    def _fake_switcher_builder(*args, **kwargs):
        del args, kwargs

        def _switcher(params, states, mask):
            del params, states
            captured["switcher_calls"] += 1
            return jnp.where(mask, jnp.array(9.0, dtype=jnp.float32), jnp.inf)

        return _switcher

    def _fake_loop_builder(*args, **kwargs):
        fwd_expansion = args[1]
        bwd_expansion = args[2]
        captured["fwd_heuristic_fn"] = fwd_expansion.heuristic_fn
        captured["bwd_heuristic_fn"] = bwd_expansion.heuristic_fn

        def _init_loop_state(*inner_args, **inner_kwargs):
            del inner_args, inner_kwargs
            return SimpleNamespace(bi_result=SimpleNamespace())

        return _init_loop_state, (lambda _: False), (lambda loop_state: loop_state)

    monkeypatch.setattr(bi_astar_mod, "variable_batch_switcher_builder", _fake_switcher_builder)
    monkeypatch.setattr(bi_astar_mod, "unified_bi_search_loop_builder", _fake_loop_builder)
    monkeypatch.setattr(bi_astar_mod, "jit_with_warmup", lambda fn, **kwargs: fn)

    search_fn = bi_astar_mod.bi_astar_builder(
        puzzle=_BuilderDummyPuzzle(),
        heuristic=_BuilderDummyHeuristic(),
        batch_size=8,
        max_nodes=64,
    )
    assert callable(search_fn)

    states = jnp.arange(8, dtype=jnp.int32)
    mask = jnp.array([True, False, True, False, True, False, True, False])
    fwd_vals = captured["fwd_heuristic_fn"](None, states, mask)
    bwd_vals = captured["bwd_heuristic_fn"](None, states, mask)

    assert captured["switcher_calls"] == 2
    assert bool(jnp.all(jnp.isfinite(fwd_vals[mask])))
    assert bool(jnp.all(jnp.isinf(fwd_vals[~mask])))
    assert bool(jnp.all(jnp.isfinite(bwd_vals[mask])))
    assert bool(jnp.all(jnp.isinf(bwd_vals[~mask])))
