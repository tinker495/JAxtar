import matplotlib.pyplot as plt
import pandas as pd

from helpers.plots.analysis import plot_expansion_distribution, plot_heuristic_accuracy
from helpers.plots.benchmark import plot_benchmark_path_comparison
from helpers.plots.comparison import plot_comparison_analysis


def _has_text(fig: plt.Figure, needle: str) -> bool:
    return any(needle in text.get_text() for axis in fig.axes for text in axis.texts)


def test_plot_expansion_distribution_returns_placeholder_when_no_1d_data():
    fig = plot_expansion_distribution(
        [
            {"seed": 0, "expansion_analysis": {}},
        ]
    )

    assert _has_text(fig, "No expansion data available.")
    plt.close(fig)


def test_plot_expansion_distribution_with_valid_series_adds_three_axes():
    fig = plot_expansion_distribution(
        [
            {
                "seed": 1,
                "expansion_analysis": {
                    "pop_generation": [0, 1, 2],
                    "cost": [1, 2, 3],
                    "dist": [4, 3, 2],
                },
            }
        ],
        scatter_max_points=2,
    )

    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() == "Cost (g) Distribution"
    assert fig.axes[1].get_title() == "Heuristic (h) Distribution"
    assert fig.axes[2].get_title() == "Key (f=g+h) Distribution"
    plt.close(fig)


def test_plot_heuristic_accuracy_with_metrics_uses_title_annotations():
    fig = plot_heuristic_accuracy(
        [
            {
                "path_analysis": {"actual": [1, 2], "estimated": [1.1, 2.2]},
                "used_optimal_path_for_analysis": True,
            }
        ],
        metrics={"has_optimal_path_used": True, "r_squared": 0.75, "ccc": 0.5},
    )

    assert "R^2" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_heuristic_accuracy_without_data_shows_placeholder():
    fig = plot_heuristic_accuracy([])

    assert _has_text(fig, "No data for heuristic accuracy plot.")
    plt.close(fig)


def test_plot_comparison_analysis_returns_expected_plot_keys():
    solved_df = pd.DataFrame(
        {
            "run_label": ["a", "a", "b", "b"],
            "nodes_generated": [10, 12, 14, 16],
            "search_time_s": [0.1, 0.2, 0.3, 0.4],
            "path_cost": [5, 6, 7, 8],
        }
    )

    plots = plot_comparison_analysis(solved_df, sorted_labels=["a", "b"])

    assert set(plots.keys()) == {
        "path_cost_comparison",
        "search_time_comparison",
        "nodes_generated_comparison",
        "nodes_vs_time_scatter",
        "nodes_vs_path_cost_scatter",
    }
    assert all(isinstance(fig, plt.Figure) for fig in plots.values())
    for fig in plots.values():
        plt.close(fig)


def test_plot_comparison_analysis_returns_empty_for_empty_dataframe():
    assert plot_comparison_analysis(pd.DataFrame(), sorted_labels=["a"]) == {}


def test_plot_benchmark_path_comparison_with_data_and_missing_matches():
    solved_df = pd.DataFrame(
        {
            "path_cost": [8, 9],
            "benchmark_optimal_path_cost": [8, 10],
            "path_action_count": [6, 7],
            "benchmark_optimal_action_count": [6, 7],
            "matches_optimal_path": [True, False],
        }
    )

    fig = plot_benchmark_path_comparison(solved_df)

    assert fig.axes[0].get_title() == "Solution Cost vs. Optimal Cost"
    assert fig.axes[1].get_title() == "Solution Length vs. Optimal Length"
    plt.close(fig)


def test_plot_benchmark_path_comparison_with_missing_columns_shows_placeholders():
    fig = plot_benchmark_path_comparison(
        pd.DataFrame(
            {
                "seed": [1, 2],
                "path_cost": [None, None],
                "benchmark_optimal_path_cost": [None, None],
                "path_action_count": [None, None],
                "benchmark_optimal_action_count": [None, None],
            }
        )
    )

    assert _has_text(fig, "No cost comparison data available.")
    assert _has_text(fig, "No path length comparison data available.")
    plt.close(fig)
