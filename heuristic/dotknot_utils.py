"""Shared utility functions for DotKnot heuristic and Q-function."""

import jax.numpy as jnp
from puxle import DotKnot


def get_color_distances(current: DotKnot.State, color_idx: int, size: int, color_num: int) -> float:
    """
    Get distance for a single color pair in the DotKnot puzzle.

    Args:
        current: Current DotKnot state.
        color_idx: Index of the color to compute distance for.
        size: Puzzle grid size.
        color_num: Number of colors in the puzzle.

    Returns:
        Manhattan distance between the two endpoints of the given color,
        or 0 if the first endpoint is not present.
    """
    unpacked = current.board_unpacked
    point_a = unpacked == (color_idx + 1)
    point_a_available = jnp.any(point_a)
    point_a_pos = jnp.stack(jnp.unravel_index(jnp.argmax(point_a), (size, size)))
    point_b = unpacked == (color_idx + color_num + 1)
    point_b_pos = jnp.stack(jnp.unravel_index(jnp.argmax(point_b), (size, size)))
    return jnp.where(point_a_available, jnp.sum(jnp.abs(point_a_pos - point_b_pos)), 0)
