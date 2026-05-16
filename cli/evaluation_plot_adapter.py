"""Evaluation Plot Adapter.

Owns every plot emission previously inlined in ``EvaluationRunner.run()``.
The Module exposes ``EvaluationPlotAdapter`` as a Protocol with four
high-level methods and ships two first-party implementations:

- ``MatplotlibPlotAdapter`` — production default, consumes ``helpers.plots``.
- ``NullPlotAdapter`` — test/CI default; every method is a no-op so the eval
  loop can run without importing matplotlib or ``helpers.plots``.

``EvaluationRunner`` takes the adapter as a constructor parameter and never
imports ``helpers.plots`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from helpers.artifact_manager import ArtifactManager


@runtime_checkable
class EvaluationPlotAdapter(Protocol):
    """Plot-emission Interface consumed by ``EvaluationRunner``."""

    def plot_solved_distributions(
        self,
        *,
        solved_df: "pd.DataFrame",
        artifact_manager: "ArtifactManager",
    ) -> None:
        """Emit path cost / search time / nodes-generated distribution plots."""
        ...

    def plot_benchmark_comparison(
        self,
        *,
        solved_df: "pd.DataFrame",
        has_benchmark: bool,
        artifact_manager: "ArtifactManager",
    ) -> None:
        """Emit the benchmark vs solved-path comparison plot when benchmark
        data is present."""
        ...

    def plot_heuristic_panel(
        self,
        *,
        results: Sequence[Mapping],
        metrics: Optional[Mapping],
        file_suffix: str,
        artifact_manager: "ArtifactManager",
    ) -> None:
        """Emit the heuristic-accuracy panel."""
        ...

    def plot_per_seed_expansion(
        self,
        *,
        results: Sequence[Mapping],
        max_plots: int,
        scatter_max_points: int,
        max_node_size: int,
        artifact_manager: "ArtifactManager",
    ) -> None:
        """For each result with expansion analysis, emit expansion-distribution
        and search-tree-semantic plots inside ``expansion_plots/``."""
        ...


class NullPlotAdapter:
    """No-op adapter for test/CI environments."""

    def plot_solved_distributions(
        self,
        *,
        solved_df,
        artifact_manager,
    ) -> None:
        return None

    def plot_benchmark_comparison(
        self,
        *,
        solved_df,
        has_benchmark,
        artifact_manager,
    ) -> None:
        return None

    def plot_heuristic_panel(
        self,
        *,
        results,
        metrics,
        file_suffix,
        artifact_manager,
    ) -> None:
        return None

    def plot_per_seed_expansion(
        self,
        *,
        results,
        max_plots,
        scatter_max_points,
        max_node_size,
        artifact_manager,
    ) -> None:
        return None


class MatplotlibPlotAdapter:
    """Production default: consumes ``helpers.plots`` for every emission."""

    def plot_solved_distributions(
        self,
        *,
        solved_df,
        artifact_manager,
    ) -> None:
        if solved_df.empty:
            return
        from helpers.plots import (
            plot_nodes_generated_by_path_cost,
            plot_path_cost_distribution,
            plot_search_time_by_path_cost,
        )

        fig = plot_path_cost_distribution(solved_df)
        artifact_manager.save_and_log_plot("path_cost_distribution", fig)

        fig = plot_search_time_by_path_cost(solved_df)
        artifact_manager.save_and_log_plot("search_time_by_path_cost", fig)

        fig = plot_nodes_generated_by_path_cost(solved_df)
        artifact_manager.save_and_log_plot("nodes_generated_by_path_cost", fig)

    def plot_benchmark_comparison(
        self,
        *,
        solved_df,
        has_benchmark,
        artifact_manager,
    ) -> None:
        if solved_df.empty or not has_benchmark:
            return
        from helpers.plots import plot_benchmark_path_comparison

        fig = plot_benchmark_path_comparison(solved_df)
        artifact_manager.save_and_log_plot("benchmark_path_comparison", fig)

    def plot_heuristic_panel(
        self,
        *,
        results,
        metrics,
        file_suffix,
        artifact_manager,
    ) -> None:
        from helpers.plots import plot_heuristic_accuracy

        fig = plot_heuristic_accuracy(results, metrics=metrics)
        artifact_manager.save_and_log_plot(f"heuristic_accuracy{file_suffix}", fig)

    def plot_per_seed_expansion(
        self,
        *,
        results,
        max_plots,
        scatter_max_points,
        max_node_size,
        artifact_manager,
    ) -> None:
        from helpers.plots import (
            plot_expansion_distribution,
            plot_search_tree_semantic,
        )

        for r in results[:max_plots]:
            if not r.get("expansion_analysis"):
                continue
            fig = plot_expansion_distribution([r], scatter_max_points=scatter_max_points)
            artifact_manager.save_and_log_plot(
                f"expansion_dist_seed_{r['seed']}",
                fig,
                sub_dir="expansion_plots",
            )
            try:
                fig_tree = plot_search_tree_semantic(r, max_points=max_node_size)
                artifact_manager.save_and_log_plot(
                    f"search_tree_semantic_seed_{r['seed']}",
                    fig_tree,
                    sub_dir="expansion_plots",
                )
            except (ValueError, RuntimeError, AttributeError, OSError) as e:
                # Semantic tree plot is best-effort; mirror prior behaviour.
                print(f"Warning: Failed to generate semantic search tree plot: {e}")


__all__ = [
    "EvaluationPlotAdapter",
    "MatplotlibPlotAdapter",
    "NullPlotAdapter",
]
