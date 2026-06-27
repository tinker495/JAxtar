"""Behaviour tests for evaluation plotting."""

from __future__ import annotations

import pytest


def test_runner_accepts_duck_typed_plot_adapter_kwarg():
    """`EvaluationRunner.__init__` keeps a plot_adapter keyword so tests and
    headless callers can inject any object with the four plotting methods.
    """
    import inspect

    from cli.evaluation_runner import EvaluationRunner

    sig = inspect.signature(EvaluationRunner.__init__)
    assert "plot_adapter" in sig.parameters
    assert sig.parameters["plot_adapter"].default is None


def test_adapter_module_only_exports_default_adapter():
    import cli.evaluation_plot_adapter as adapter_module

    assert adapter_module.__all__ == ["MatplotlibPlotAdapter"]
    assert not hasattr(adapter_module, "EvaluationPlotAdapter")
    assert not hasattr(adapter_module, "NullPlotAdapter")


@pytest.fixture
def _stub_artifact_manager(tmp_path):
    """Capture (name, sub_dir) pairs without touching the real ArtifactManager."""

    class _Stub:
        def __init__(self):
            self.calls = []

        def save_and_log_plot(self, name, fig, sub_dir=None):
            self.calls.append((name, sub_dir))

    return _Stub()


def test_matplotlib_adapter_skips_when_solved_df_empty(_stub_artifact_manager):
    """Empty solved_df must short-circuit plot emission — the Matplotlib
    adapter should not import matplotlib helpers in that case.
    """
    import pandas as pd

    from cli.evaluation_plot_adapter import MatplotlibPlotAdapter

    adapter = MatplotlibPlotAdapter()
    adapter.plot_solved_distributions(
        solved_df=pd.DataFrame(), artifact_manager=_stub_artifact_manager
    )
    adapter.plot_benchmark_comparison(
        solved_df=pd.DataFrame(),
        has_benchmark=True,
        artifact_manager=_stub_artifact_manager,
    )
    assert _stub_artifact_manager.calls == []
