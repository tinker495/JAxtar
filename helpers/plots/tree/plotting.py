"""Main plotting orchestration for search tree visualization."""

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..constants import (
    GRID_ALPHA,
    GRID_LINESTYLE,
    MARKER_SIZE_LARGE,
    SCATTER_ALPHA_HIGH,
    TREE_FIGSIZE,
    TREE_GOAL_COLOR,
    TREE_PATH_COLOR,
    TREE_PATH_LINEWIDTH,
    TREE_START_COLOR,
)
from .data_preparation import prepare_tree_data
from .path_tracing import trace_solution_path

logger = logging.getLogger(__name__)


def plot_search_tree_semantic(
    result_item: Dict[str, Any],
    max_points: int = 100000,
    top_n_leaf_nodes: Optional[int] = None,
) -> plt.Figure:
    """
    Plot search tree in semantic coordinates (g vs h).

    Args:
        result_item: Dictionary containing search results with keys:
            - expansion_analysis: Dict with costs, dists, original_indices, parent_indices
            - path_analysis: Optional dict with actual/estimated values
            - solved_index: Optional hash index of solved node
            - path_cost: Optional total path cost
        max_points: Maximum number of points to plot (downsampling threshold)
        top_n_leaf_nodes: If specified, only show paths to top N leaf nodes
            by h-value (closest to goal). None shows all paths.

    Returns:
        Matplotlib figure object
    """
    tree_data = prepare_tree_data(result_item, max_points, top_n_leaf_nodes)

    if tree_data is None:
        fig, ax = plt.subplots(figsize=TREE_FIGSIZE)
        ax.text(
            0.5, 0.5, "No tree data available", ha="center", va="center", transform=ax.transAxes
        )
        return fig

    costs = tree_data.costs
    dists = tree_data.dists
    original_indices = tree_data.original_indices
    parent_indices = tree_data.parent_indices
    kept_node_mask = tree_data.kept_node_mask

    # Filter data if leaf node mask is specified
    if kept_node_mask is not None:
        costs = costs[kept_node_mask]
        dists = dists[kept_node_mask]
        original_indices = original_indices[kept_node_mask]
        parent_indices = parent_indices[kept_node_mask]

    fig, ax = plt.subplots(figsize=TREE_FIGSIZE)

    sc = ax.scatter(
        costs,
        dists,
        c=costs + dists,
        cmap="viridis",
        s=tree_data.scatter_size,
        alpha=tree_data.scatter_alpha,
        edgecolor="none",
        label="Expanded Nodes",
    )
    plt.colorbar(sc, ax=ax, label="f-value (g + h)")

    _plot_solution_path(ax, result_item, costs, dists, original_indices, parent_indices)

    ax.set_title("Search Tree Topology: Cost (g) vs Heuristic (h)")
    ax.set_xlabel("Cost (g) - Distance from Start")
    ax.set_ylabel("Heuristic (h) - Estimated Distance to Goal")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

    return fig


def _plot_solution_path(
    ax,
    result_item: Dict[str, Any],
    costs: np.ndarray,
    dists: np.ndarray,
    original_indices: np.ndarray,
    parent_indices: np.ndarray,
) -> bool:
    """
    Attempt to plot the solution path using multiple fallback strategies.

    Returns:
        True if path was successfully plotted, False otherwise
    """
    analysis = result_item.get("expansion_analysis", {})

    path_indices = None
    path_plotted = False

    try:
        solved_index = analysis.get("solved_index")
        if solved_index is not None:
            path_indices = trace_solution_path(original_indices, parent_indices, solved_index)
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to trace path from solved index: {e}")

    if not path_plotted and (path_indices is None or len(path_indices) == 0):
        path_plotted = _plot_path_from_analysis(ax, result_item)

    if not path_plotted and (path_indices is None or len(path_indices) == 0):
        try:
            goal_node_idx = np.argmin(dists)
            if dists[goal_node_idx] < 1e-6:
                path_indices = trace_solution_path(
                    original_indices, parent_indices, int(original_indices[goal_node_idx])
                )
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Path tracing failed: {e}")

    if path_indices is not None and len(path_indices) > 0:
        ax.plot(
            costs[path_indices],
            dists[path_indices],
            color=TREE_PATH_COLOR,
            linestyle="-",
            linewidth=TREE_PATH_LINEWIDTH,
            label="Solution Path",
            alpha=SCATTER_ALPHA_HIGH,
        )
        ax.scatter(costs[path_indices], dists[path_indices], c=TREE_PATH_COLOR, s=20, zorder=10)
        ax.scatter(
            costs[path_indices[-1]],
            dists[path_indices[-1]],
            c=TREE_START_COLOR,
            s=MARKER_SIZE_LARGE,
            marker="*",
            edgecolors="black",
            label="Start",
            zorder=20,
        )
        ax.scatter(
            costs[path_indices[0]],
            dists[path_indices[0]],
            c=TREE_GOAL_COLOR,
            s=MARKER_SIZE_LARGE,
            marker="*",
            edgecolors="black",
            label="Goal",
            zorder=20,
        )
        path_plotted = True

    return path_plotted


def _plot_path_from_analysis(ax, result_item: Dict[str, Any]) -> bool:
    """
    Plot path using path_analysis data if available.

    Returns:
        True if path was successfully plotted, False otherwise
    """
    try:
        path_data = result_item.get("path_analysis")
        used_optimal = result_item.get("used_optimal_path_for_analysis")
        if path_data and not bool(used_optimal):
            actual = np.asarray(path_data.get("actual") or [])
            estimated = np.asarray(path_data.get("estimated") or [])
            path_cost = result_item.get("path_cost")
            if actual.size > 0 and estimated.size > 0 and path_cost is not None:
                length = min(len(actual), len(estimated))
                g_vals = path_cost - actual[:length]
                h_vals = estimated[:length]
                mask = np.isfinite(g_vals) & np.isfinite(h_vals)
                if np.any(mask):
                    g_vals = g_vals[mask]
                    h_vals = h_vals[mask]
                    ax.plot(
                        g_vals,
                        h_vals,
                        color=TREE_PATH_COLOR,
                        linestyle="-",
                        linewidth=TREE_PATH_LINEWIDTH,
                        label="Solution Path",
                        alpha=SCATTER_ALPHA_HIGH,
                    )
                    ax.scatter(g_vals, h_vals, c=TREE_PATH_COLOR, s=20, zorder=10)
                    ax.scatter(
                        g_vals[0],
                        h_vals[0],
                        c=TREE_START_COLOR,
                        s=MARKER_SIZE_LARGE,
                        marker="*",
                        edgecolors="black",
                        label="Start",
                        zorder=20,
                    )
                    ax.scatter(
                        g_vals[-1],
                        h_vals[-1],
                        c=TREE_GOAL_COLOR,
                        s=MARKER_SIZE_LARGE,
                        marker="*",
                        edgecolors="black",
                        label="Goal",
                        zorder=20,
                    )
                    return True
    except (ValueError, RuntimeError, AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Path plotting from analysis failed: {e}")

    return False
