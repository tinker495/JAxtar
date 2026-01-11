from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import _plot_scatter_with_ellipses


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
