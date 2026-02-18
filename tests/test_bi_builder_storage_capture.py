from types import SimpleNamespace

import jax.numpy as jnp

import JAxtar.bi_stars.bi_astar as bi_astar_mod
import JAxtar.bi_stars.bi_astar_d as bi_astar_d_mod
import JAxtar.bi_stars.bi_qstar as bi_qstar_mod


class _DummySolveConfig:
    @staticmethod
    def default():
        return _DummySolveConfig()


class _DummyState:
    @staticmethod
    def default():
        return _DummyState()


class _DummyPuzzle:
    action_size = 4
    SolveConfig = _DummySolveConfig
    State = _DummyState

    def hindsight_transform(self, solve_config, start):
        return solve_config


class _DummyHeuristic:
    is_fixed = False

    def batched_distance(self, params, states):
        return states

    def prepare_heuristic_parameters(self, solve_config, **kwargs):
        return None


class _DummyQFunction:
    is_fixed = False

    def batched_q_value(self, params, states):
        return states

    def prepare_q_parameters(self, solve_config, **kwargs):
        return None


def _jit_no_warmup(fn, **kwargs):
    return fn


def test_bi_astar_builder_does_not_prebuild_search_storage(monkeypatch):
    calls = []

    def _spy_build(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bi_astar_mod, "jit_with_warmup", _jit_no_warmup)
    monkeypatch.setattr(bi_astar_mod, "build_bi_search_result", _spy_build)

    search_fn = bi_astar_mod.bi_astar_builder(
        _DummyPuzzle(),
        _DummyHeuristic(),
        batch_size=8,
        max_nodes=64,
    )

    assert callable(search_fn)
    assert calls == []


def test_bi_qstar_builder_does_not_prebuild_search_storage(monkeypatch):
    calls = []

    def _spy_build(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bi_qstar_mod, "jit_with_warmup", _jit_no_warmup)
    monkeypatch.setattr(bi_qstar_mod, "build_bi_search_result", _spy_build)

    search_fn = bi_qstar_mod.bi_qstar_builder(
        _DummyPuzzle(),
        _DummyQFunction(),
        batch_size=8,
        max_nodes=64,
    )

    assert callable(search_fn)
    assert calls == []


def test_bi_astar_d_builder_uses_masked_heuristic_batch_switcher(monkeypatch):
    captured = {"heuristic_fns": []}

    class _HeuristicMustNotCallBatchedDistance:
        is_fixed = False

        def batched_distance(self, params, states):
            del params, states
            raise AssertionError("bi_astar_d should use the masked switcher path")

        def prepare_heuristic_parameters(self, solve_config, **kwargs):
            del solve_config, kwargs
            return None

    def _spy_switcher_builder(eval_fn, **kwargs):
        del eval_fn, kwargs

        def _switcher(params, states, filled):
            del params, states
            return jnp.where(filled, 7.0, jnp.inf)

        return _switcher

    def _spy_deferred_expansion(**kwargs):
        captured["heuristic_fns"].append(kwargs["heuristic_fn"])
        return SimpleNamespace(**kwargs)

    def _stub_loop_builder(*args, **kwargs):
        del args, kwargs
        return (
            lambda *loop_args, **loop_kwargs: SimpleNamespace(bi_result=SimpleNamespace()),
            lambda loop_state: False,
            lambda loop_state: loop_state,
        )

    monkeypatch.setattr(bi_astar_d_mod, "jit_with_warmup", _jit_no_warmup)
    monkeypatch.setattr(bi_astar_d_mod, "variable_batch_switcher_builder", _spy_switcher_builder)
    monkeypatch.setattr(bi_astar_d_mod, "DeferredExpansion", _spy_deferred_expansion)
    monkeypatch.setattr(bi_astar_d_mod, "unified_bi_search_loop_builder", _stub_loop_builder)

    search_fn = bi_astar_d_mod.bi_astar_d_builder(
        _DummyPuzzle(),
        _HeuristicMustNotCallBatchedDistance(),
        batch_size=4,
        max_nodes=32,
    )

    assert callable(search_fn)
    assert len(captured["heuristic_fns"]) == 2

    filled = jnp.array([True, False, True, False], dtype=jnp.bool_)
    expected = [7.0, jnp.inf, 7.0, jnp.inf]
    for heuristic_fn in captured["heuristic_fns"]:
        out = heuristic_fn(None, jnp.arange(4, dtype=jnp.float32), filled)
        assert out.tolist() == expected


class _DummySolveConfigWithTarget:
    def __init__(self, target):
        self.TargetState = target

    def replace(self, **kwargs):
        return _DummySolveConfigWithTarget(kwargs.get("TargetState", self.TargetState))


def _patch_bi_qstar_runtime(monkeypatch, captured):
    def _stub_loop_builder(*args, **kwargs):
        def _init_loop_state(
            bi_result,
            solve_config,
            inverse_solveconfig,
            start,
            q_params_forward,
            q_params_backward,
        ):
            captured["inverse_solveconfig"] = inverse_solveconfig
            return SimpleNamespace(bi_result=bi_result)

        return _init_loop_state, (lambda _: False), (lambda loop_state: loop_state)

    monkeypatch.setattr(bi_qstar_mod, "jit_with_warmup", _jit_no_warmup)
    monkeypatch.setattr(
        bi_qstar_mod.jax.lax, "while_loop", lambda cond, body, loop_state: loop_state
    )
    monkeypatch.setattr(bi_qstar_mod, "_bi_qstar_loop_builder", _stub_loop_builder)
    monkeypatch.setattr(
        bi_qstar_mod,
        "build_bi_search_result",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        bi_qstar_mod,
        "materialize_meeting_point_hashidxs",
        lambda bi_result, puzzle, solve_config: bi_result,
    )
    monkeypatch.setattr(bi_qstar_mod, "finalize_bidirectional_result", lambda bi_result: bi_result)


def test_bi_qstar_builder_uses_get_inverse_solve_config_when_no_hindsight(monkeypatch):
    class _PuzzleWithInverseOnly:
        action_size = 4
        SolveConfig = _DummySolveConfig
        State = _DummyState

        def get_inverse_solve_config(self, solve_config, start):
            return solve_config.replace(TargetState=("inverse", start))

    captured = {}
    _patch_bi_qstar_runtime(monkeypatch, captured)

    search_fn = bi_qstar_mod.bi_qstar_builder(
        _PuzzleWithInverseOnly(),
        _DummyQFunction(),
        batch_size=8,
        max_nodes=64,
    )

    start = _DummyState.default()
    solve_config = _DummySolveConfigWithTarget(target="goal")
    search_fn(solve_config, start)

    assert captured["inverse_solveconfig"].TargetState == ("inverse", start)


def test_bi_qstar_builder_falls_back_to_targetstate_replace(monkeypatch):
    class _PuzzleNoInverse:
        action_size = 4
        SolveConfig = _DummySolveConfig
        State = _DummyState

    captured = {}
    _patch_bi_qstar_runtime(monkeypatch, captured)

    search_fn = bi_qstar_mod.bi_qstar_builder(
        _PuzzleNoInverse(),
        _DummyQFunction(),
        batch_size=8,
        max_nodes=64,
    )

    start = _DummyState.default()
    solve_config = _DummySolveConfigWithTarget(target="goal")
    search_fn(solve_config, start)

    assert captured["inverse_solveconfig"].TargetState is start
