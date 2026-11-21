import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def trace_solution_path(
    original_indices: np.ndarray,
    parent_indices: np.ndarray,
    solved_index: Optional[int],
) -> Optional[np.ndarray]:
    """Return array positions along the solution path in goal-to-start order, or None."""
    if solved_index is None:
        return None

    N = len(original_indices)
    node_map = {int(orig): i for i, orig in enumerate(original_indices)}

    curr = node_map.get(int(solved_index))
    if curr is None:
        logger.warning(f"Solved index {solved_index} not found in node map")
        return None

    path = []
    visited = set()

    for _ in range(N):
        if curr in visited:
            logger.warning("Cycle detected in path tracing")
            break
        visited.add(curr)
        path.append(curr)

        p_idx = parent_indices[curr]
        if p_idx == -1:
            break

        next_idx = node_map.get(int(p_idx))
        if next_idx is None:
            break

        if next_idx == curr:
            logger.warning("Self-loop detected in path tracing")
            break

        curr = next_idx

    if len(path) == 0:
        return None

    return np.asarray(path, dtype=np.int32)
