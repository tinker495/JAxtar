"""Architecture + behaviour guard for evaluation plotting.

Locks the small seam documented in CONTEXT.md "Evaluation Plot Adapter":

- ``EvaluationRunner`` MUST NOT import the seven ``plot_*`` helpers from
  ``helpers.plots`` directly.
- The default ``MatplotlibPlotAdapter`` keeps plotting imports behind methods.
- ``EvaluationRunner`` still accepts a duck-typed ``plot_adapter`` override for
  tests/headless callers, without maintaining a separate Null adapter class.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_EVAL_RUNNER_PATH = Path(__file__).resolve().parents[1] / "cli" / "evaluation_runner.py"

_PLOT_NAMES = (
    "plot_path_cost_distribution",
    "plot_search_time_by_path_cost",
    "plot_nodes_generated_by_path_cost",
    "plot_benchmark_path_comparison",
    "plot_heuristic_accuracy",
    "plot_expansion_distribution",
    "plot_search_tree_semantic",
)


def test_evaluation_runner_does_not_import_helpers_plots_directly():
    source = _EVAL_RUNNER_PATH.read_text()
    assert "from helpers.plots" not in source, (
        "`EvaluationRunner` must consume plots through its plot adapter. "
        "Move new plot emissions into `MatplotlibPlotAdapter` instead of "
        "importing `helpers.plots` here."
    )
    for plot_name in _PLOT_NAMES:
        assert plot_name + "(" not in source, (
            f"Direct call to `{plot_name}` found in evaluation_runner.py — "
            "route it through `self.plot_adapter` instead."
        )


def test_runner_imports_default_adapter_seam():
    source = _EVAL_RUNNER_PATH.read_text()
    assert "from .evaluation_plot_adapter import MatplotlibPlotAdapter" in source


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
