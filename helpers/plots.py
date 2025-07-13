import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse


def plot_path_cost_distribution(solved_df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of path costs for solved puzzles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=solved_df, x="path_cost", kde=True, ax=ax)
    ax.set_title("Distribution of Path Cost")
    ax.set_xlabel("Path Cost")
    ax.set_ylabel("Frequency")
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


def plot_heuristic_accuracy(results: list[dict]) -> plt.Figure:
    """Plots the heuristic/q-function accuracy."""
    all_actual_dists = []
    all_estimated_dists = []
    solved_results = [r for r in results if r["solved"]]
    for r in solved_results:
        if r.get("path_analysis"):
            analysis_data = r["path_analysis"]
            if analysis_data.get("actual") and analysis_data.get("estimated"):
                all_actual_dists.extend(analysis_data["actual"])
                all_estimated_dists.extend(analysis_data["estimated"])

    fig, ax = plt.subplots(figsize=(12, 12))

    if all_actual_dists:
        plot_df = pd.DataFrame(
            {
                "Actual Cost to Goal": all_actual_dists,
                "Estimated Distance": all_estimated_dists,
            }
        )

        sns.boxplot(data=plot_df, x="Actual Cost to Goal", y="Estimated Distance", ax=ax)
        sns.pointplot(
            data=plot_df,
            x="Actual Cost to Goal",
            y="Estimated Distance",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )

        max_val = 0
        if all_actual_dists and all_estimated_dists:
            max_val = max(np.max(all_actual_dists), np.max(all_estimated_dists))

        limit = int(max_val) + 1 if max_val > 0 else 10

        ax.plot(
            [0, limit], [0, limit], "g--", alpha=0.75, zorder=0, label="y=x (Perfect Heuristic)"
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)

        xticks = range(int(limit) + 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        xticklabels = ax.get_xticklabels()
        if len(xticklabels) > 10:
            step = int(np.ceil(len(xticklabels) / 10))
            for i, label in enumerate(xticklabels):
                if i % step != 0:
                    label.set_visible(False)

    ax.set_title("Heuristic/Q-function Accuracy Analysis")
    ax.set_xlabel("Actual Cost to Goal")
    ax.set_ylabel("Estimated Distance (Heuristic/Q-Value)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def _plot_scatter_analysis(
    solved_df: pd.DataFrame,
    hue_col: str,
    sorted_labels: list[str],
    scatter_max_points: int,
    legend_title: str,
) -> plt.Figure:
    """Internal function to generate scatter plot with confidence ellipses."""
    fig_scatter, ax_scatter = plt.subplots(figsize=(12, 8))

    plot_df = solved_df
    if len(solved_df) > scatter_max_points:
        plot_df = solved_df.sample(n=scatter_max_points, random_state=42)

    sns.scatterplot(
        data=plot_df,
        x="nodes_generated",
        y="search_time_s",
        hue=hue_col,
        hue_order=sorted_labels,
        palette="tab10",
        alpha=0.7,
        edgecolor=None,
        ax=ax_scatter,
    )

    def plot_confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
        if x.size <= 1 or y.size <= 1:
            return
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        ellipse = Ellipse(
            (0, 0),
            width=2 * ell_radius_x,
            height=2 * ell_radius_y,
            facecolor=facecolor,
            **kwargs,
        )
        transf = (
            transforms.Affine2D()
            .rotate_deg(45 if pearson != 0 else 0)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    palette = sns.color_palette("tab10")
    grouped = solved_df.groupby(hue_col)

    for i, label in enumerate(sorted_labels):
        if label not in grouped.groups:
            continue
        group = grouped.get_group(label)
        x = group["nodes_generated"].values
        y = group["search_time_s"].values
        color = palette[i % len(palette)]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ax_scatter.scatter(
            mean_x,
            mean_y,
            color=color,
            s=120,
            marker="X",
            edgecolor="black",
            zorder=10,
        )
        plot_confidence_ellipse(
            x, y, ax_scatter, n_std=1.0, edgecolor=color, linewidth=2, alpha=0.5
        )

    ax_scatter.annotate(
        "Fewer nodes (better)",
        xy=(0.03, 0.02),
        xycoords="axes fraction",
        xytext=(0.35, 0.02),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        color="black",
    )
    ax_scatter.annotate(
        "Faster (better)",
        xy=(0.03, 0.02),
        xycoords="axes fraction",
        xytext=(0.03, 0.35),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        rotation=90,
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        color="black",
    )

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_title(f"Search Time vs. Generated Nodes by {legend_title}")
    ax_scatter.set_xlabel("Nodes Generated (log scale)")
    ax_scatter.set_ylabel("Search Time (s, log scale)")
    ax_scatter.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_scatter.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
    fig_scatter.tight_layout()
    return fig_scatter


def plot_pop_ratio_analysis(
    solved_df: pd.DataFrame, scatter_max_points: int = 2000
) -> dict[str, plt.Figure]:
    """Generates a dictionary of plots for pop ratio analysis."""
    if solved_df.empty:
        return {}

    plots = {}
    hue_col = "pop_ratio"
    if "pop_ratio" in solved_df.columns:
        solved_df["pop_ratio_str"] = (
            solved_df["pop_ratio"].replace([np.inf, -np.inf], "inf").astype(str)
        )
        hue_col = "pop_ratio_str"

    hue_labels = solved_df[hue_col].unique()
    try:
        hue_labels = sorted(hue_labels, key=float)
    except (ValueError, TypeError):
        hue_labels = sorted(hue_labels)

    # Path cost by pop ratio
    fig_pc, ax_pc = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x=hue_col, y="path_cost", order=hue_labels, ax=ax_pc)
    ax_pc.set_title("Path Cost by Pop Ratio")
    ax_pc.set_xlabel("Pop Ratio")
    ax_pc.set_ylabel("Path Cost")
    fig_pc.tight_layout()
    plots["path_cost_by_popratio"] = fig_pc

    # Search time by pop ratio
    fig_st, ax_st = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x=hue_col, y="search_time_s", order=hue_labels, ax=ax_st)
    ax_st.set_yscale("log")
    ax_st.set_title("Search Time by Pop Ratio")
    ax_st.set_xlabel("Pop Ratio")
    ax_st.set_ylabel("Search Time (s, log scale)")
    fig_st.tight_layout()
    plots["search_time_by_popratio"] = fig_st

    # Nodes generated by pop ratio
    fig_ng, ax_ng = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x=hue_col, y="nodes_generated", order=hue_labels, ax=ax_ng)
    ax_ng.set_title("Generated Nodes by Pop Ratio")
    ax_ng.set_xlabel("Pop Ratio")
    ax_ng.set_ylabel("Nodes Generated")
    fig_ng.tight_layout()
    plots["nodes_generated_by_popratio"] = fig_ng

    # Scatter plot
    plots["nodes_vs_time_popratio_scatter"] = _plot_scatter_analysis(
        solved_df=solved_df,
        hue_col=hue_col,
        sorted_labels=hue_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Pop Ratio",
    )

    return plots


def plot_comparison_analysis(
    solved_df: pd.DataFrame, sorted_labels: list[str], scatter_max_points: int = 2000
) -> dict[str, plt.Figure]:
    """Generates a dictionary of plots for comparing multiple evaluation runs."""
    if solved_df.empty:
        return {}

    plots = {}

    # Path cost comparison
    fig_pc, ax_pc = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x="run_label", y="path_cost", order=sorted_labels, ax=ax_pc)
    ax_pc.set_title("Path Cost Comparison")
    ax_pc.set_xlabel("Run (Config Differences)")
    ax_pc.set_ylabel("Path Cost")
    plt.setp(ax_pc.get_xticklabels(), rotation=45, ha="right")
    fig_pc.tight_layout()
    plots["path_cost_comparison"] = fig_pc

    # Search time comparison
    fig_st, ax_st = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x="run_label", y="search_time_s", order=sorted_labels, ax=ax_st)
    ax_st.set_yscale("log")
    ax_st.set_title("Search Time Comparison")
    ax_st.set_xlabel("Run (Config Differences)")
    ax_st.set_ylabel("Search Time (s, log scale)")
    plt.setp(ax_st.get_xticklabels(), rotation=45, ha="right")
    fig_st.tight_layout()
    plots["search_time_comparison"] = fig_st

    # Nodes generated comparison
    fig_ng, ax_ng = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=solved_df, x="run_label", y="nodes_generated", order=sorted_labels, ax=ax_ng)
    ax_ng.set_title("Generated Nodes Comparison")
    ax_ng.set_xlabel("Run (Config Differences)")
    ax_ng.set_ylabel("Nodes Generated")
    plt.setp(ax_ng.get_xticklabels(), rotation=45, ha="right")
    fig_ng.tight_layout()
    plots["nodes_generated_comparison"] = fig_ng

    # Scatter plot
    plots["nodes_vs_time_scatter"] = _plot_scatter_analysis(
        solved_df=solved_df,
        hue_col="run_label",
        sorted_labels=sorted_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Run (Config Differences)",
    )

    return plots
