import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark_path_comparison(solved_df: pd.DataFrame) -> plt.Figure:
    """Compares solution costs and lengths against benchmark-optimal references."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cost comparison subplot
    ax_cost = axes[0]
    cost_df = solved_df.dropna(subset=["path_cost", "benchmark_optimal_path_cost"])
    if not cost_df.empty:
        matches = cost_df.get("matches_optimal_path")
        colors = None
        if matches is not None and not matches.isna().all():
            colors = matches.map({True: "#2ca02c", False: "#d62728"})
        ax_cost.scatter(
            cost_df["benchmark_optimal_path_cost"],
            cost_df["path_cost"],
            c=colors,
            edgecolor="black" if colors is None else None,
            alpha=0.7,
        )
        min_val = min(cost_df["benchmark_optimal_path_cost"].min(), cost_df["path_cost"].min())
        max_val = max(cost_df["benchmark_optimal_path_cost"].max(), cost_df["path_cost"].max())
        ax_cost.plot([min_val, max_val], [min_val, max_val], "--", color="#555555", linewidth=1)
        ax_cost.set_xlim(
            min_val - 0.1 * abs(min_val),
            max_val + 0.1 * abs(max_val) if max_val else max_val + 1,
        )
        ax_cost.set_ylim(
            min_val - 0.1 * abs(min_val),
            max_val + 0.1 * abs(max_val) if max_val else max_val + 1,
        )
    else:
        ax_cost.text(
            0.5,
            0.5,
            "No cost comparison data available.",
            transform=ax_cost.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    ax_cost.set_title("Solution Cost vs. Optimal Cost")
    ax_cost.set_xlabel("Optimal Path Cost")
    ax_cost.set_ylabel("Solution Path Cost")
    ax_cost.grid(True, linestyle="--", alpha=0.5)

    # Path length comparison subplot
    ax_len = axes[1]
    length_df = solved_df.dropna(subset=["path_action_count", "benchmark_optimal_action_count"])
    if not length_df.empty:
        matches = length_df.get("matches_optimal_path")
        colors = None
        if matches is not None and not matches.isna().all():
            colors = matches.map({True: "#2ca02c", False: "#d62728"})
        ax_len.scatter(
            length_df["benchmark_optimal_action_count"],
            length_df["path_action_count"],
            c=colors,
            edgecolor="black" if colors is None else None,
            alpha=0.7,
        )
        min_val = min(
            length_df["benchmark_optimal_action_count"].min(),
            length_df["path_action_count"].min(),
        )
        max_val = max(
            length_df["benchmark_optimal_action_count"].max(),
            length_df["path_action_count"].max(),
        )
        ax_len.plot([min_val, max_val], [min_val, max_val], "--", color="#555555", linewidth=1)
        ax_len.set_xlim(min_val - 0.5, max_val + 0.5)
        ax_len.set_ylim(min_val - 0.5, max_val + 0.5)
    else:
        ax_len.text(
            0.5,
            0.5,
            "No path length comparison data available.",
            transform=ax_len.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    ax_len.set_title("Solution Length vs. Optimal Length")
    ax_len.set_xlabel("Optimal Action Count")
    ax_len.set_ylabel("Solution Action Count")
    ax_len.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Benchmark Solution Quality", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
