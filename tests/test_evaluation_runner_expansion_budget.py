"""Regression test: expansion traces are only retained for plotted samples.

Retaining an expansion trace (O(expanded nodes) host memory) for every
evaluated sample made long benchmark runs OOM; collection must stop after
`max_expansion_plots` traces.
"""

import numpy as np

from cli.evaluation_runner import EvaluationRunner
from config.pydantic_models import EvalOptions, PuzzleOptions


class _FakeSearchResult:
    solved = False
    generated_size = 7

    def to_expansion_trace(self):
        from JAxtar.expansion_trace import ExpansionTrace

        return ExpansionTrace(
            pop_generation=np.array([0]),
            cost=np.array([1.0]),
            dist=np.array([2.0]),
            original_indices=np.array([0]),
        )


class _FakePuzzle:
    def get_inits(self, key):
        return object(), object()


def test_expansion_traces_stop_after_budget(tmp_path):
    eval_options = EvalOptions(
        num_eval=5,
        max_expansion_plots=2,
        plot_unsolved=True,
        use_early_stopping=False,
    )
    runner = EvaluationRunner(
        puzzle=_FakePuzzle(),
        puzzle_name="fake",
        search_model=object(),
        search_model_name="heuristic",
        search_builder_fn=None,
        eval_options=eval_options,
        puzzle_opts=PuzzleOptions(puzzle="fake"),
        output_dir=tmp_path,
    )

    results = runner._run_evaluation(
        search_fn=lambda solve_config, state: _FakeSearchResult(),
        puzzle=runner.puzzle,
        eval_inputs=list(range(5)),
    )

    attached = [r["expansion_analysis"] is not None for r in results]
    assert attached == [True, True, False, False, False]


def test_expansion_traces_disabled_with_zero_budget(tmp_path):
    eval_options = EvalOptions(
        num_eval=3,
        max_expansion_plots=0,
        plot_unsolved=True,
        use_early_stopping=False,
    )
    runner = EvaluationRunner(
        puzzle=_FakePuzzle(),
        puzzle_name="fake",
        search_model=object(),
        search_model_name="heuristic",
        search_builder_fn=None,
        eval_options=eval_options,
        puzzle_opts=PuzzleOptions(puzzle="fake"),
        output_dir=tmp_path,
    )

    results = runner._run_evaluation(
        search_fn=lambda solve_config, state: _FakeSearchResult(),
        puzzle=runner.puzzle,
        eval_inputs=list(range(3)),
    )

    assert all(r["expansion_analysis"] is None for r in results)
