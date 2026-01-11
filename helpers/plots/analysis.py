from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_expansion_distribution(results: list[dict], scatter_max_points: int = 5000) -> plt.Figure:
    """Plots the distribution of node costs, heuristics, and keys over expansion steps."""
    expansion_data = []
    for r in results:
        if r.get("expansion_analysis"):
            analysis = r["expansion_analysis"]
            # Filter only 1D arrays for DataFrame construction (states are 2D)
            flat_data = {
                k: v
                for k, v in analysis.items()
                if isinstance(v, (np.ndarray, list)) and np.ndim(v) == 1
            }
            df = pd.DataFrame(flat_data)
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
    # - Optimal Path: "Actual" is optimal remaining cost-to-go (benchmark reference).
    # - Search Path: "Actual" is remaining cost-to-go along the *found* solution path, which can be suboptimal.
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

        diag_label = (
            "y=x (Perfect Heuristic)"
            if has_optimal_path_used
            else "y=x (Perfect if found path is optimal)"
        )
        ax.plot([0, limit], [0, limit], "g--", alpha=0.75, zorder=0, label=diag_label)
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
