from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "plot_expansion_distribution",
    "plot_heuristic_accuracy",
    "plot_nodes_generated_by_path_cost",
    "plot_path_cost_distribution",
    "plot_search_time_by_path_cost",
    "plot_benchmark_path_comparison",
    "plot_comparison_analysis",
    "plot_search_tree_semantic",
    "_plot_scatter_with_ellipses",
]

_EXPORTS = {
    "plot_expansion_distribution": "helpers.plots.analysis",
    "plot_heuristic_accuracy": "helpers.plots.analysis",
    "plot_nodes_generated_by_path_cost": "helpers.plots.basic",
    "plot_path_cost_distribution": "helpers.plots.basic",
    "plot_search_time_by_path_cost": "helpers.plots.basic",
    "plot_benchmark_path_comparison": "helpers.plots.benchmark",
    "plot_comparison_analysis": "helpers.plots.comparison",
    "plot_search_tree_semantic": "helpers.plots.tree.plotting",
    "_plot_scatter_with_ellipses": "helpers.plots.utils",
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
