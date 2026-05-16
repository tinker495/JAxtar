"""Architecture + behaviour guard for the Evaluation Plot Adapter.

Locks the contracts documented in CONTEXT.md "Evaluation Plot Adapter":

- `EvaluationRunner` MUST NOT import the seven `plot_*` helpers from
  `helpers.plots` directly. Every plot emission flows through the adapter.
- The `EvaluationPlotAdapter` Protocol is runtime-checkable; both
  `MatplotlibPlotAdapter` and `NullPlotAdapter` satisfy it.
- `NullPlotAdapter` no-ops every method (CI / headless paths can opt out of
  plotting without a matplotlib dependency).
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
        "`EvaluationRunner` must consume plots through `EvaluationPlotAdapter`. "
        "Move new plot emissions into a method on the adapter (and on "
        "`MatplotlibPlotAdapter` + `NullPlotAdapter`) instead of importing "
        "`helpers.plots` here."
    )
    for plot_name in _PLOT_NAMES:
        assert plot_name + "(" not in source, (
            f"Direct call to `{plot_name}` found in evaluation_runner.py — "
            "route it through `self.plot_adapter` instead."
        )


def test_runner_imports_adapter_seam():
    source = _EVAL_RUNNER_PATH.read_text()
    assert "from .evaluation_plot_adapter import" in source, (
        "`EvaluationRunner` must import `EvaluationPlotAdapter` and "
        "`MatplotlibPlotAdapter` from the adapter module."
    )


def test_both_adapters_satisfy_protocol():
    from cli.evaluation_plot_adapter import (
        EvaluationPlotAdapter,
        MatplotlibPlotAdapter,
        NullPlotAdapter,
    )

    assert isinstance(
        MatplotlibPlotAdapter(), EvaluationPlotAdapter
    ), "MatplotlibPlotAdapter must satisfy the EvaluationPlotAdapter Protocol."
    assert isinstance(
        NullPlotAdapter(), EvaluationPlotAdapter
    ), "NullPlotAdapter must satisfy the EvaluationPlotAdapter Protocol."


def test_null_adapter_methods_are_no_op():
    """NullPlotAdapter must accept the canonical kwargs without touching them."""
    from cli.evaluation_plot_adapter import NullPlotAdapter

    n = NullPlotAdapter()
    assert n.plot_solved_distributions(solved_df=None, artifact_manager=None) is None
    assert (
        n.plot_benchmark_comparison(solved_df=None, has_benchmark=False, artifact_manager=None)
        is None
    )
    assert (
        n.plot_heuristic_panel(results=[], metrics=None, file_suffix="", artifact_manager=None)
        is None
    )
    assert (
        n.plot_per_seed_expansion(
            results=[],
            max_plots=0,
            scatter_max_points=0,
            max_node_size=0,
            artifact_manager=None,
        )
        is None
    )


def test_runner_accepts_plot_adapter_kwarg():
    """`EvaluationRunner.__init__` must declare a `plot_adapter` keyword so CI
    paths can inject `NullPlotAdapter()`.
    """
    import inspect

    from cli.evaluation_runner import EvaluationRunner

    sig = inspect.signature(EvaluationRunner.__init__)
    assert "plot_adapter" in sig.parameters, (
        "`EvaluationRunner.__init__` must accept a `plot_adapter` parameter "
        "for CI / NullPlotAdapter injection."
    )
    param = sig.parameters["plot_adapter"]
    assert param.default is None, (
        "`plot_adapter` should default to None so the constructor falls back "
        "to MatplotlibPlotAdapter when no adapter is injected."
    )


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
    assert (
        _stub_artifact_manager.calls == []
    ), "MatplotlibPlotAdapter must not emit plots for an empty solved_df."
