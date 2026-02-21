from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse

from .constants import (
    COMPARISON_ELLIPSE_ALPHA,
    COMPARISON_ELLIPSE_LINEWIDTH,
    COMPARISON_FIGSIZE,
    COMPARISON_SCATTER_MAX_POINTS,
    GRID_ALPHA,
    GRID_LINESTYLE,
    GRID_LINEWIDTH,
    PALETTE_TAB10,
    SCATTER_ALPHA_HIGH,
)


def _plot_scatter_with_ellipses(
    solved_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    sorted_labels: list[str],
    scatter_max_points: int = COMPARISON_SCATTER_MAX_POINTS,
    legend_title: str = "",
    title: str = "",
    x_log: bool = False,
    y_log: bool = False,
    add_annotations: bool = False,
    varying_params: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Internal function to generate a generic scatter plot with confidence ellipses.
    If varying_params are provided, it connects the centers of runs that share common parameters.
    """
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)

    plot_df = solved_df.copy()
    if len(plot_df) > scatter_max_points:
        plot_df = plot_df.sample(n=scatter_max_points, random_state=42)

    num_labels = max(1, len(sorted_labels))
    scatter_palette = (
        PALETTE_TAB10
        if len(PALETTE_TAB10) >= num_labels
        else sns.color_palette("tab10", num_labels)
    )
    if len(PALETTE_TAB10) >= num_labels:
        scatter_palette = PALETTE_TAB10[:num_labels]

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=sorted_labels,
        palette=scatter_palette,
        alpha=SCATTER_ALPHA_HIGH,
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

    palette = PALETTE_TAB10
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
                linewidth=COMPARISON_ELLIPSE_LINEWIDTH,
                alpha=COMPARISON_ELLIPSE_ALPHA,
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
    ax.grid(
        True, which="both", linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA
    )
    fig.tight_layout()
    return fig
