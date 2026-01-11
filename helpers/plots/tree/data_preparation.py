import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import SCATTER_ALPHA_LOW, SCATTER_ALPHA_MED

logger = logging.getLogger(__name__)


@dataclass
class TreeData:
    costs: np.ndarray
    dists: np.ndarray
    original_indices: np.ndarray
    parent_indices: np.ndarray
    N: int
    plot_edges: bool
    scatter_alpha: float
    scatter_size: int
    kept_node_mask: Optional[np.ndarray] = None


def _filter_top_n_leaf_paths(
    original_indices: np.ndarray,
    parent_indices: np.ndarray,
    dists: np.ndarray,
    top_n: Optional[int],
) -> Optional[np.ndarray]:
    """
    Identify top N leaf nodes and return mask of nodes to keep.

    A node is a leaf if it is not listed as a parent of any other node.
    Top N leaf nodes are selected by smallest h-value (closest to goal).

    Args:
        original_indices: Array of original hash indices for each node
        parent_indices: Array of parent hash indices for each node
        dists: Array of h-values (heuristic distance to goal)
        top_n: Number of top leaf nodes to keep, or None to keep all

    Returns:
        Boolean mask array indicating which nodes to keep, or None if top_n is None
    """
    if top_n is None or top_n <= 0:
        return None

    N = len(original_indices)

    # Create set of all parent indices (excluding -1 for root)
    parent_set = set(int(p) for p in parent_indices if p >= 0)

    # Identify leaf nodes: nodes whose original_index is NOT in parent_set
    is_leaf = np.array([int(orig) not in parent_set for orig in original_indices])

    leaf_indices = np.where(is_leaf)[0]

    if len(leaf_indices) == 0:
        return None

    # Sort leaf nodes by h-value (smallest h = closest to goal)
    leaf_h_values = dists[leaf_indices]
    sorted_order = np.argsort(leaf_h_values)

    # Take top N leaf nodes
    top_n_indices = leaf_indices[sorted_order[: min(top_n, len(leaf_indices))]]

    # Find all ancestors of top N leaf nodes by tracing back through parents
    keep_set = set()
    for leaf_idx in top_n_indices:
        curr = leaf_idx
        while curr >= 0 and curr < N:
            keep_set.add(curr)
            p_idx = parent_indices[curr]
            if p_idx == -1:
                break
            # Find parent array position from original_indices
            parent_pos = None
            for i, orig in enumerate(original_indices):
                if int(orig) == int(p_idx):
                    parent_pos = i
                    break
            if parent_pos is None or parent_pos == curr:
                break
            curr = parent_pos

    # Create mask
    kept_node_mask = np.array([i in keep_set for i in range(N)], dtype=bool)

    return kept_node_mask


def prepare_tree_data(
    result_item: dict, max_points: int, top_n_leaf_nodes: Optional[int] = None
) -> Optional[TreeData]:
    analysis = result_item.get("expansion_analysis")
    if not analysis:
        return None

    costs = analysis["cost"]
    dists = analysis["dist"]
    original_indices = analysis.get("original_indices")
    parent_indices = analysis.get("parent_indices")

    if original_indices is None or parent_indices is None:
        return None

    N = len(costs)

    kept_node_mask = _filter_top_n_leaf_paths(
        original_indices, parent_indices, dists, top_n_leaf_nodes
    )

    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        if kept_node_mask is not None:
            # Apply leaf filter to downsampled indices
            indices_to_sample = np.where(kept_node_mask)[0]
            if len(indices_to_sample) > 0:
                indices = np.random.choice(
                    indices_to_sample, min(max_points, len(indices_to_sample)), replace=False
                )
            else:
                return None

        return TreeData(
            costs=costs[indices],
            dists=dists[indices],
            original_indices=original_indices[indices],
            parent_indices=parent_indices[indices],
            N=len(indices),
            plot_edges=True,
            scatter_alpha=SCATTER_ALPHA_LOW,
            scatter_size=1,
            kept_node_mask=None,
        )
    else:
        if kept_node_mask is not None and not np.any(kept_node_mask):
            return None

        return TreeData(
            costs=costs,
            dists=dists,
            original_indices=original_indices,
            parent_indices=parent_indices,
            N=N,
            plot_edges=True,
            scatter_alpha=SCATTER_ALPHA_MED,
            scatter_size=5,
            kept_node_mask=kept_node_mask,
        )
