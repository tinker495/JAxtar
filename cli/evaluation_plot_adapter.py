"""Evaluation plotting helpers kept out of ``EvaluationRunner``.

``EvaluationRunner`` calls methods on a duck-typed plot adapter. The default
adapter below imports ``helpers.plots`` only inside plotting methods, so the
main evaluation loop can be imported without loading matplotlib plot helpers.
"""

from __future__ import annotations


class MatplotlibPlotAdapter:
    """Default adapter: consumes ``helpers.plots`` for every emission."""

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


__all__ = ["MatplotlibPlotAdapter"]
