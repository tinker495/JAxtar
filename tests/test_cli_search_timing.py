import jax.numpy as jnp
from rich.text import Text

from cli import search_outcome, search_runner
from config.pydantic_models import SearchOptions, VisualizeOptions


def test_search_time_excludes_result_normalisation(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(search_outcome.time, "time", lambda: clock["now"])

    class _State:
        def str(self, *, solve_config):
            return "state"

    class _Puzzle:
        has_goal_data = True

        def get_inits(self, key):
            return "goal", _State()

    class _SearchResult:
        solved = True
        generated_size = 1
        solved_idx = 0

        def get_cost(self, solved_idx):
            clock["now"] += 5.0
            return jnp.array(1.0)

    def search_fn(solve_config, state):
        clock["now"] += 2.0
        return _SearchResult()

    search_times, _, _ = search_runner.search_samples(
        search_fn=search_fn,
        puzzle=_Puzzle(),
        puzzle_name="test",
        dist_fn=lambda solve_config, state: jnp.array(0.0),
        dist_fn_format=lambda puzzle, value: Text("0"),
        seeds=[0],
        search_options=SearchOptions(batch_size=1, max_node_size=1),
        visualize_options=VisualizeOptions(),
    )

    assert search_times == [2.0]


def test_measure_search_waits_for_bidirectional_result(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(search_outcome.time, "time", lambda: clock["now"])

    class _ReadyValue:
        def block_until_ready(self):
            clock["now"] += 3.0

    class _Meeting:
        found = _ReadyValue()

    class _SearchResult:
        meeting = _Meeting()
        forward = object()
        backward = object()

    def search_fn():
        clock["now"] += 2.0
        return _SearchResult()

    result, search_time = search_outcome.measure_search(search_fn)

    assert isinstance(result, _SearchResult)
    assert search_time == 5.0
