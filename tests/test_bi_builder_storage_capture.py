import JAxtar.bi_stars.bi_astar as bi_astar_mod
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
