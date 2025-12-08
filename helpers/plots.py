from typing import Dict, List, Optional

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
            min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val) if max_val else max_val + 1
        )
        ax_cost.set_ylim(
            min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val) if max_val else max_val + 1
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


def plot_expansion_distribution(results: list[dict], scatter_max_points: int = 5000) -> plt.Figure:
    """Plots the distribution of node costs, heuristics, and keys over expansion steps."""
    expansion_data = []
    for r in results:
        if r.get("expansion_analysis"):
            df = pd.DataFrame(r["expansion_analysis"])
            df["seed"] = r["seed"]
            expansion_data.append(df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    title = "Node Value Distribution over Expansion Steps"
    if len(results) == 1 and "seed" in results[0]:
        title += f" (Seed: {results[0]['seed']})"
    fig.suptitle(title, fontsize=16)

    if not expansion_data:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No expansion data available.",
                ha="center",
                va="center",
                fontsize=12,
            )
        return fig

    combined_df = pd.concat(expansion_data, ignore_index=True)
    combined_df["key"] = combined_df["cost"] + combined_df["dist"]

    # Use a smaller sample for scatter plot if data is too large, to avoid overplotting
    sample_df = combined_df
    if len(combined_df) > scatter_max_points:
        sample_df = combined_df.sample(n=scatter_max_points, random_state=42)

    # Plot Cost (g) vs. Expansion Step
    sns.lineplot(
        data=combined_df,
        x="pop_generation",
        y="cost",
        ax=axes[0],
        color="blue",
        label="Mean Cost",
        errorbar="sd",
    )
    sns.scatterplot(
        data=sample_df,
        x="pop_generation",
        y="cost",
        ax=axes[0],
        alpha=0.1,
        color="blue",
        edgecolor=None,
        label="Expanded Nodes",
    )
    axes[0].set_title("Cost (g) Distribution")
    axes[0].set_ylabel("Cost")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Heuristic (h) vs. Expansion Step
    sns.lineplot(
        data=combined_df,
        x="pop_generation",
        y="dist",
        ax=axes[1],
        color="green",
        label="Mean Heuristic",
        errorbar="sd",
    )
    sns.scatterplot(
        data=sample_df,
        x="pop_generation",
        y="dist",
        ax=axes[1],
        alpha=0.1,
        color="green",
        edgecolor=None,
        label="Expanded Nodes",
    )
    axes[1].set_title("Heuristic (h) Distribution")
    axes[1].set_ylabel("Heuristic")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot Key (f) vs. Expansion Step
    sns.lineplot(
        data=combined_df,
        x="pop_generation",
        y="key",
        ax=axes[2],
        color="red",
        label="Mean Key (f=g+h)",
        errorbar="sd",
    )
    sns.scatterplot(
        data=sample_df,
        x="pop_generation",
        y="key",
        ax=axes[2],
        alpha=0.1,
        color="red",
        edgecolor=None,
        label="Expanded Nodes",
    )
    axes[2].set_title("Key (f=g+h) Distribution")
    axes[2].set_xlabel("Expansion Step (Pop Generation)")
    axes[2].set_ylabel("Key Value")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_heuristic_accuracy(
    results: list[dict], metrics: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """Plots the heuristic/q-function accuracy."""
    all_actual_dists = []
    all_estimated_dists = []

    # Check if we're plotting data derived from optimal paths
    has_optimal_path_used = metrics.get("has_optimal_path_used", False) if metrics else False

    # Include results with path analysis.
    # If metrics says we have optimal path used, we generally assume most/all valid points come from it,
    # or at least we want to label it as such.
    results_with_analysis = [r for r in results if r.get("path_analysis")]

    for r in results_with_analysis:
        if r.get("path_analysis"):
            analysis_data = r["path_analysis"]
            if analysis_data.get("actual") and analysis_data.get("estimated"):
                all_actual_dists.extend(analysis_data["actual"])
                all_estimated_dists.extend(analysis_data["estimated"])

    fig, ax = plt.subplots(figsize=(12, 12))

    # Adjust title based on data source
    source_label = "Optimal Path" if has_optimal_path_used else "Search Path"
    title = f"Heuristic/Q-function Accuracy Analysis ({source_label})"

    if metrics:
        r_squared = metrics.get("r_squared")
        ccc = metrics.get("ccc")
        if r_squared is not None and ccc is not None:
            title += f"\n($R^2={r_squared:.3f}$, $\\rho_c={ccc:.3f}$)"

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
    else:
        ax.text(
            0.5,
            0.5,
            "No data for heuristic accuracy plot.",
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_title(title)
    ax.set_xlabel("Actual Cost to Goal")
    ax.set_ylabel("Estimated Distance (Heuristic/Q-Value)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def _plot_scatter_with_ellipses(
    solved_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    sorted_labels: list[str],
    scatter_max_points: int,
    legend_title: str,
    title: str,
    x_log: bool,
    y_log: bool,
    add_annotations: bool = False,
    varying_params: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Internal function to generate a generic scatter plot with confidence ellipses.
    If varying_params are provided, it connects the centers of runs that share common parameters.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_df = solved_df.copy()
    if len(plot_df) > scatter_max_points:
        plot_df = plot_df.sample(n=scatter_max_points, random_state=42)

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=sorted_labels,
        palette="tab10",
        alpha=0.7,
        edgecolor=None,
        ax=ax,
    )

    def plot_confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
        if x.size <= 1 or y.size <= 1:
            return
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)) or cov[0, 0] == 0 or cov[1, 1] == 0:
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

    # Pre-calculate run means for centers and lines
    run_means = {}
    for label, group_data in grouped:
        if not group_data.empty:
            run_means[label] = (
                np.mean(group_data[x_col]),
                np.mean(group_data[y_col]),
            )

    # Plot ellipses and centers
    for i, label in enumerate(sorted_labels):
        if label not in grouped.groups:
            continue
        group_data = grouped.get_group(label)
        color = palette[i % len(palette)]
        if label in run_means:
            mean_x, mean_y = run_means[label]
            ax.scatter(
                mean_x,
                mean_y,
                color=color,
                s=120,
                marker="X",
                edgecolor="black",
                zorder=10,
            )
            plot_confidence_ellipse(
                group_data[x_col].values,
                group_data[y_col].values,
                ax,
                n_std=1.0,
                edgecolor=color,
                linewidth=2,
                alpha=0.5,
            )

    # Connect centers if there are parameters to group by
    if varying_params and len(varying_params) > 1:
        if not run_means:
            return fig

        param_mapping = solved_df[["run_label"] + varying_params].drop_duplicates("run_label")
        mean_coords_df = (
            pd.DataFrame.from_dict(run_means, orient="index", columns=[x_col, y_col])
            .reset_index()
            .rename(columns={"index": "run_label"})
        )
        mean_points_df = pd.merge(mean_coords_df, param_mapping, on="run_label")

        sorted_params = sorted(
            varying_params, key=lambda p: mean_points_df[p].nunique(), reverse=True
        )
        line_var = sorted_params[0]
        grouping_vars = sorted_params[1:]

        if grouping_vars:
            line_groups = mean_points_df.groupby(grouping_vars)
            for _, group_df in line_groups:
                group_df_copy = group_df.copy()
                try:
                    sort_key_series = group_df_copy[line_var].replace(
                        [np.inf, -np.inf], float("inf")
                    )
                    pd.to_numeric(sort_key_series)
                    group_df_copy["sort_key"] = sort_key_series.astype(float)
                except (ValueError, TypeError):
                    group_df_copy["sort_key"] = group_df_copy[line_var].astype(str)

                sorted_group = group_df_copy.sort_values("sort_key")
                ax.plot(
                    sorted_group[x_col],
                    sorted_group[y_col],
                    linestyle=":",
                    alpha=0.7,
                    marker="o",
                    markersize=4,
                )

    if add_annotations:
        ax.annotate(
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
        ax.annotate(
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

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    x_label = x_col.replace("_", " ").title() + (" (log scale)" if x_log else "")
    y_label = y_col.replace("_", " ").title() + (" (log scale)" if y_log else "")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_pop_ratio_analysis(
    solved_df: pd.DataFrame, scatter_max_points: int = 2000
) -> dict[str, plt.Figure]:
    """Generates a dictionary of plots for pop ratio analysis."""
    if solved_df.empty:
        return {}

    plots = {}
    hue_col = "pop_ratio"
    if "pop_ratio" in solved_df.columns:
        solved_df["pop_ratio"] = (
            solved_df["pop_ratio"].replace([np.inf, -np.inf], "inf").astype(str)
        )
        hue_col = "pop_ratio"

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
    plots["nodes_vs_time_popratio_scatter"] = _plot_scatter_with_ellipses(
        solved_df=solved_df,
        x_col="nodes_generated",
        y_col="search_time_s",
        hue_col=hue_col,
        sorted_labels=hue_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Pop Ratio",
        title="Search Time vs. Generated Nodes by Pop Ratio",
        x_log=True,
        y_log=True,
        add_annotations=True,
    )

    # NEW PLOT: Scatter plot for nodes vs path cost
    plots["nodes_vs_path_cost_popratio_scatter"] = _plot_scatter_with_ellipses(
        solved_df=solved_df,
        x_col="nodes_generated",
        y_col="path_cost",
        hue_col=hue_col,
        sorted_labels=hue_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Pop Ratio",
        title="Generated Nodes vs. Path Cost by Pop Ratio",
        x_log=True,
        y_log=False,
        add_annotations=False,
    )

    return plots


def plot_comparison_analysis(
    solved_df: pd.DataFrame,
    sorted_labels: list[str],
    scatter_max_points: int = 2000,
    varying_params: Optional[List[str]] = None,
) -> dict[str, plt.Figure]:
    """
    Generates a dictionary of plots for comparing multiple evaluation runs.
    Dynamically creates 1D, 2D, or faceted plots based on the number of varying parameters.
    """
    if solved_df.empty:
        return {}

    plots = {}
    plot_df = solved_df.copy()

    if varying_params is None:
        varying_params = []

    def create_faceted_boxplot(y_metric, y_label, base_title):
        """Helper to create simple, 2D, or faceted plots."""
        # 0 or 1 varying parameter
        if len(varying_params) <= 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            x_var = "run_label" if not varying_params else varying_params[0]
            order = sorted_labels if x_var == "run_label" else None

            if x_var != "run_label":
                plot_df[x_var] = plot_df[x_var].replace([np.inf, -np.inf], "inf").astype(str)
                try:
                    order = sorted(
                        plot_df[x_var].unique(),
                        key=lambda x: float(x) if x != "inf" else float("inf"),
                    )
                except (ValueError, TypeError):
                    order = sorted(plot_df[x_var].unique())

            sns.boxplot(data=plot_df, x=x_var, y=y_metric, order=order, ax=ax)
            y_label_final = y_label
            if y_metric in ["search_time_s", "nodes_generated"]:
                ax.set_yscale("log")
                y_label_final += " (log scale)"
            ax.set_title(f"{base_title} by {x_var}")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_label_final)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
            fig.tight_layout()
            return fig

        # 2 or more varying parameters
        # Sort params by cardinality (descending) to decide axes.
        sorted_params = sorted(varying_params, key=lambda p: plot_df[p].nunique(), reverse=True)
        p_x, p_hue = sorted_params[0], sorted_params[1]
        p_col = None
        if len(varying_params) == 3:
            p_col = sorted_params[2]
        elif len(varying_params) > 3:
            p_col_list = sorted_params[2:]
            p_col = " & ".join(p.split(".")[-1] for p in p_col_list)
            plot_df[p_col] = plot_df[p_col_list].apply(
                lambda row: ",\n".join(f"{col.split('.')[-1]}={row[col]}" for col in p_col_list),
                axis=1,
            )

        # Prepare for sorting categorical axes
        def get_sorted_unique(param_name):
            if param_name not in plot_df.columns:
                return None
            # Convert to string for consistent sorting
            plot_df[param_name] = plot_df[param_name].replace([np.inf, -np.inf], "inf").astype(str)
            try:
                return sorted(
                    plot_df[param_name].unique(),
                    key=lambda x: float(x) if x != "inf" else float("inf"),
                )
            except (ValueError, TypeError):
                return sorted(plot_df[param_name].unique())

        x_order, hue_order = get_sorted_unique(p_x), get_sorted_unique(p_hue)
        y_label_final = (
            f"{y_label} (log scale)"
            if y_metric in ["search_time_s", "nodes_generated"]
            else y_label
        )

        if p_col:
            g = sns.catplot(
                data=plot_df,
                x=p_x,
                y=y_metric,
                hue=p_hue,
                col=p_col,
                kind="box",
                order=x_order,
                hue_order=hue_order,
                height=5,
                aspect=1.3,
                legend_out=True,
                col_wrap=min(3, plot_df[p_col].nunique()),
            )
            if y_metric in ["search_time_s", "nodes_generated"]:
                g.set(yscale="log")
            g.fig.suptitle(f"{base_title} by {p_x}, {p_hue}, and {p_col}", y=1.03)
            g.set_axis_labels(x_var=p_x, y_var=y_label_final)
            g.set_titles(col_template="{col_name}")
            fig = g.fig
        else:  # 2 params
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(
                data=plot_df,
                x=p_x,
                y=y_metric,
                hue=p_hue,
                order=x_order,
                hue_order=hue_order,
                ax=ax,
            )
            if y_metric in ["search_time_s", "nodes_generated"]:
                ax.set_yscale("log")
            ax.set_title(f"{base_title} by {p_x} and {p_hue}")
            ax.set_xlabel(p_x)
            ax.set_ylabel(y_label_final)

        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # Generate plots for each metric
    plots["path_cost_comparison"] = create_faceted_boxplot(
        "path_cost", "Path Cost", "Path Cost Comparison"
    )
    plots["search_time_comparison"] = create_faceted_boxplot(
        "search_time_s", "Search Time (s)", "Search Time Comparison"
    )
    plots["nodes_generated_comparison"] = create_faceted_boxplot(
        "nodes_generated", "Nodes Generated", "Generated Nodes Comparison"
    )

    # Scatter plot remains the same, showing all combinations distinctly
    plots["nodes_vs_time_scatter"] = _plot_scatter_with_ellipses(
        solved_df=plot_df,
        x_col="nodes_generated",
        y_col="search_time_s",
        hue_col="run_label",
        sorted_labels=sorted_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Run (Config Differences)",
        title="Search Time vs. Generated Nodes by Run",
        x_log=True,
        y_log=True,
        add_annotations=True,
        varying_params=varying_params,
    )

    # NEW PLOT: Scatter plot for nodes vs path cost
    plots["nodes_vs_path_cost_scatter"] = _plot_scatter_with_ellipses(
        solved_df=plot_df,
        x_col="nodes_generated",
        y_col="path_cost",
        hue_col="run_label",
        sorted_labels=sorted_labels,
        scatter_max_points=scatter_max_points,
        legend_title="Run (Config Differences)",
        title="Generated Nodes vs. Path Cost by Run",
        x_log=True,
        y_log=False,
        add_annotations=False,
        varying_params=varying_params,
    )

    return plots
