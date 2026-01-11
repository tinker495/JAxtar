import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_path_cost_distribution(solved_df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of path costs for solved puzzles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    costs = solved_df["path_cost"].dropna().to_numpy()
    if costs.size == 0:
        ax.text(
            0.5,
            0.5,
            "No path cost data available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        return fig

    if np.allclose(costs, np.round(costs), atol=1e-6):
        binwidth = 1.0
        bins = np.arange(costs.min() - 0.5, costs.max() + 1.5, binwidth)
    else:
        raw_bins = np.histogram_bin_edges(costs, bins="fd")
        target_bins = int(np.clip(len(raw_bins), 10, 40))
        bins = np.linspace(costs.min(), costs.max(), target_bins + 1)

    sns.histplot(
        costs,
        bins=bins,
        stat="count",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    mean_val = float(np.mean(costs))
    median_val = float(np.median(costs))
    ax.axvline(mean_val, color="#DD8452", linestyle="--", linewidth=1.2, label="Mean")
    ax.axvline(median_val, color="#55A868", linestyle=":", linewidth=1.4, label="Median")
    ax.set_title("Path Cost Distribution")
    ax.set_xlabel("Path Cost")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_search_time_by_path_cost(solved_df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of search time by path cost."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=solved_df, x="path_cost", y="search_time_s", ax=ax)
    sns.pointplot(
        data=solved_df,
        x="path_cost",
        y="search_time_s",
        estimator=np.median,
        color="red",
        linestyles="--",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Search Time Distribution by Path Cost")
    ax.set_xlabel("Path Cost")
    ax.set_ylabel("Search Time (s)")
    fig.tight_layout()
    return fig


def plot_nodes_generated_by_path_cost(solved_df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of generated nodes by path cost."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=solved_df, x="path_cost", y="nodes_generated", ax=ax)
    sns.pointplot(
        data=solved_df,
        x="path_cost",
        y="nodes_generated",
        estimator=np.median,
        color="red",
        linestyles="--",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Generated Nodes Distribution by Path Cost")
    ax.set_xlabel("Path Cost")
    ax.set_ylabel("Nodes Generated")
    fig.tight_layout()
    return fig
