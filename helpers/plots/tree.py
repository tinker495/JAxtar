import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_search_tree_semantic(result_item: dict, max_points: int = 500000) -> plt.Figure:
    """
    Plots the search tree in a semantic coordinate system (g vs h).
    X-axis: Cost (g) - Distance from Start
    Y-axis: Heuristic (h) - Estimated Distance to Goal

    Draws edges between parent and child nodes to visualize the search topology.
    Highlights the optimal path if available.
    """
    analysis = result_item.get("expansion_analysis")
    if not analysis:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No expansion analysis data.", ha="center", va="center")
        return fig

    # Data extraction
    costs = analysis["cost"]  # g values
    dists = analysis["dist"]  # h values
    original_indices = analysis.get("original_indices")
    parent_indices = analysis.get("parent_indices")

    if original_indices is None or parent_indices is None:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No parent/index data for tree visualization.",
            ha="center",
            va="center",
        )
        return fig

    N = len(costs)

    # Downsample if needed, but try to keep as many as possible (max_points is large by default)
    # For tree visualization, we need consistent parent-child pairs.
    # If we just random sample nodes, we lose the edges.
    # So we prefer to plot ALL nodes if N < max_points.
    # If N > max_points, we might just plot a subset of points without lines, or just the path.
    # But user requested "max points to max", so we try to handle large N.

    # Mapping from original hashtable index to current analysis array index
    # original_indices[i] = hashtable_index of node i in arrays `costs` and `dists`
    # We need a reverse map to find parent's array index from its hashtable index.

    # Create a sparse lookup (or full array if capacity isn't too huge)
    # Since capacity can be large, a dict or hashmap is safer, but slower to build in Python.
    # However, original_indices are typically contiguous if we dumped the whole table,
    # but here we only have expanded nodes.

    # Optimized edge list construction
    # 1. Build a lookup: hashtable_idx -> array_idx
    # This might be slow for millions of nodes in Python.

    # Limit N to avoid memory explosion in LineCollection construction
    if N > max_points:
        # If too many points, we select a random subset for the scatter plot background
        # and overlay the solution path. We won't draw the full tree edges.
        indices = np.random.choice(N, max_points, replace=False)
        plot_edges = False
        subset_costs = costs[indices]
        subset_dists = dists[indices]
        scatter_alpha = 0.1
        scatter_s = 1
    else:
        indices = np.arange(N)
        plot_edges = True
        subset_costs = costs
        subset_dists = dists
        scatter_alpha = 0.3
        scatter_s = 5

    fig, ax = plt.subplots(figsize=(14, 10))

    # 1. Plot all expanded nodes (Background)
    sc = ax.scatter(
        subset_costs,
        subset_dists,
        c=subset_costs + subset_dists,  # Color by f-value
        cmap="viridis",
        s=scatter_s,
        alpha=scatter_alpha,
        edgecolor="none",
        label="Expanded Nodes",
    )
    plt.colorbar(sc, ax=ax, label="f-value (g + h)")

    # 2. Draw Edges (Tree Structure)
    # Only if N is manageable
    if plot_edges:
        # Build lookup: hash_idx -> array_idx
        # We only care about parents that are ALSO in the expanded set.
        # If a parent wasn't expanded (e.g. start node's parent is dummy), we skip.

        # Use pandas for fast join if available, or just numpy
        # df_nodes = pd.DataFrame({'idx': original_indices, 'array_pos': np.arange(N)})
        # df_edges = pd.DataFrame({'child_pos': np.arange(N), 'parent_idx': parent_indices})
        # merged = df_edges.merge(df_nodes, left_on='parent_idx', right_on='idx', how='inner')
        # This is reasonably fast for < 1M items.

        try:
            # Create a lookup array if IDs are within reasonable range
            max_id = np.max(original_indices)
            if max_id < N * 10:  # Sparse factor check
                lookup = np.full(max_id + 1, -1, dtype=np.int32)
                lookup[original_indices] = np.arange(N, dtype=np.int32)

                # Find valid parents
                valid_mask = (parent_indices >= 0) & (parent_indices <= max_id)

                # Get parent positions
                parent_pos = lookup[parent_indices]

                # Keep only edges where parent is found in our set
                valid_edges = (parent_pos != -1) & valid_mask

                # child_pos is just 0..N-1
                child_pos = np.arange(N)

                p_pos = parent_pos[valid_edges]
                c_pos = child_pos[valid_edges]

                # Construct segments: (x1, y1) -> (x2, y2)
                # (cost_p, dist_p) -> (cost_c, dist_c)
                segments = np.zeros((len(p_pos), 2, 2))
                segments[:, 0, 0] = costs[p_pos]  # x1
                segments[:, 0, 1] = dists[p_pos]  # y1
                segments[:, 1, 0] = costs[c_pos]  # x2
                segments[:, 1, 1] = dists[c_pos]  # y2

                lc = LineCollection(segments, colors="gray", alpha=0.05, linewidths=0.5)
                ax.add_collection(lc)

        except (ValueError, RuntimeError, AttributeError, IndexError) as e:
            print(f"Warning: Edge construction failed: {e}")

    # 3. Highlight Optimal Path
    if result_item.get("path_analysis"):
        path_data = result_item["path_analysis"]
        # path_data usually contains lists of costs/dists for the solution path
        if "actual" in path_data and "estimated" in path_data:
            # "actual" in path analysis is usually cost-to-go (perfect h)
            # "estimated" is h
            # We need g-values for the path.
            # Assuming path is reconstructed from Start -> Goal
            # Then g starts at 0 and increases.
            # But path_analysis might be just errors.
            pass

    # Alternative: Use the 'solved_path' if available in some form,
    # but usually we just have metrics.
    # Let's try to reconstruct from 'expansion_analysis' if we can identify the path nodes.
    # Without explicit path indices in result_item, we can't easily highlight the exact path
    # unless we trace back from goal node in 'parent_indices'.

    # Trace path from goal using explicit solved index when available.
    def _trace_path_from_hash_index(goal_hash_idx):
        node_map = {orig: i for i, orig in enumerate(original_indices)}
        curr = node_map.get(int(goal_hash_idx))
        if curr is None:
            return None
        path = []
        for _ in range(N):
            path.append(curr)
            p_idx = parent_indices[curr]
            if p_idx == -1:
                break
            next_idx = node_map.get(int(p_idx))
            if next_idx is None or next_idx == curr:
                break
            curr = next_idx
        return np.asarray(path, dtype=np.int32)

    path_indices = None
    path_plotted = False
    try:
        solved_index = analysis.get("solved_index")
        if solved_index is not None:
            path_indices = _trace_path_from_hash_index(solved_index)
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        print(f"Warning: Failed to trace path from solved index: {e}")

    # Fallback: use path analysis if we cannot trace through parent indices.
    if not path_plotted and (path_indices is None or len(path_indices) == 0):
        try:
            path_data = result_item.get("path_analysis")
            if path_data and not result_item.get("used_optimal_path_for_analysis"):
                actual = np.asarray(path_data.get("actual") or [])
                estimated = np.asarray(path_data.get("estimated") or [])
                path_cost = result_item.get("path_cost")
                if actual.size and estimated.size and path_cost is not None:
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
                            "r-",
                            linewidth=2,
                            label="Solution Path",
                            alpha=0.8,
                        )
                        ax.scatter(g_vals, h_vals, c="red", s=20, zorder=10)
                        ax.scatter(
                            g_vals[0],
                            h_vals[0],
                            c="lime",
                            s=100,
                            marker="*",
                            edgecolors="black",
                            label="Start",
                            zorder=20,
                        )
                        ax.scatter(
                            g_vals[-1],
                            h_vals[-1],
                            c="gold",
                            s=100,
                            marker="*",
                            edgecolors="black",
                            label="Goal",
                            zorder=20,
                        )
                        path_plotted = True
        except (ValueError, RuntimeError, AttributeError, IndexError, KeyError) as e:
            print(f"Warning: Path plotting from analysis failed: {e}")

    # Final fallback: use minimum distance heuristic to infer a goal.
    if not path_plotted and (path_indices is None or len(path_indices) == 0):
        try:
            goal_node_idx = np.argmin(dists)
            if dists[goal_node_idx] < 1e-6:
                path_indices = _trace_path_from_hash_index(original_indices[goal_node_idx])
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Path tracing failed: {e}")

    if path_indices is not None and len(path_indices) > 0:
        ax.plot(
            costs[path_indices],
            dists[path_indices],
            "r-",
            linewidth=2,
            label="Solution Path",
            alpha=0.8,
        )
        ax.scatter(costs[path_indices], dists[path_indices], c="red", s=20, zorder=10)
        ax.scatter(
            costs[path_indices[-1]],
            dists[path_indices[-1]],
            c="lime",
            s=100,
            marker="*",
            edgecolors="black",
            label="Start",
            zorder=20,
        )
        ax.scatter(
            costs[path_indices[0]],
            dists[path_indices[0]],
            c="gold",
            s=100,
            marker="*",
            edgecolors="black",
            label="Goal",
            zorder=20,
        )
        path_plotted = True

    ax.set_title("Search Tree Topology: Cost (g) vs Heuristic (h)")
    ax.set_xlabel("Cost (g) - Distance from Start")
    ax.set_ylabel("Heuristic (h) - Estimated Distance to Goal")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    return fig
