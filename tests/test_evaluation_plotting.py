"""Behaviour tests for evaluation plotting helpers."""

from __future__ import annotations

import inspect

import pandas as pd

from cli.evaluation_runner import (
    EvaluationRunner,
    _plot_benchmark_comparison,
    _plot_solved_distributions,
)


def test_runner_no_longer_accepts_plot_adapter_kwarg():
    sig = inspect.signature(EvaluationRunner.__init__)
    assert "plot_adapter" not in sig.parameters


def test_plot_helpers_skip_empty_solved_df(tmp_path):
    """Empty solved_df must short-circuit plot emission."""

    class _StubArtifactManager:
        def __init__(self):
            self.calls = []

        def save_and_log_plot(self, name, fig, sub_dir=None):
            self.calls.append((name, sub_dir))

    artifact_manager = _StubArtifactManager()
    solved_df = pd.DataFrame()

    _plot_solved_distributions(solved_df=solved_df, artifact_manager=artifact_manager)
    _plot_benchmark_comparison(
        solved_df=solved_df,
        has_benchmark=True,
        artifact_manager=artifact_manager,
    )

    assert artifact_manager.calls == []
