"""Path reconstruction logic for search tree visualization."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def trace_solution_path(
    original_indices: np.ndarray,
    parent_indices: np.ndarray,
    solved_index: Optional[int],
) -> Optional[np.ndarray]:
    """
    Trace the solution path from goal to start using parent indices.

    This function attempts to reconstruct the solution path by tracing backward
    from the solved node through parent pointers. It uses a node mapping to
    convert hash indices to array indices.

    Args:
        original_indices: Array of original hash indices for each node
        parent_indices: Array of parent hash indices for each node
        solved_index: Hash index of the solved/goal node, or None

    Returns:
        Array of indices representing the path from goal to start, or None if
        path cannot be traced
    """
    if solved_index is None:
        return None

    N = len(original_indices)
    node_map = {int(orig): i for i, orig in enumerate(original_indices)}

    # Find starting position in the downsampled arrays
    curr = node_map.get(int(solved_index))
    if curr is None:
        logger.warning(f"Solved index {solved_index} not found in node map")
        return None

    path = []
    visited = set()

    # Trace backward through parents
    for _ in range(N):
        if curr in visited:
            logger.warning("Cycle detected in path tracing")
            break
        visited.add(curr)
        path.append(curr)

        p_idx = parent_indices[curr]
        if p_idx == -1:
            # Reached root
            break

        next_idx = node_map.get(int(p_idx))
        if next_idx is None:
            # logger.warning(f"Parent index {p_idx} not found in node map")
            break

        if next_idx == curr:
            logger.warning("Self-loop detected in path tracing")
            break

        curr = next_idx

    if len(path) == 0:
        return None

    return np.asarray(path, dtype=np.int32)
