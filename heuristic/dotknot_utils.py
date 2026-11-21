import jax
import jax.numpy as jnp
from puxle import DotKnot


def get_color_distances(
    current: DotKnot.State, color_idx: int, size: int, color_num: int
) -> jax.Array:
    unpacked = current.board_unpacked
    point_a = unpacked == (color_idx + 1)
    point_a_available = jnp.any(point_a)
    point_a_pos = jnp.stack(jnp.unravel_index(jnp.argmax(point_a), (size, size)))
    point_b = unpacked == (color_idx + color_num + 1)
    point_b_pos = jnp.stack(jnp.unravel_index(jnp.argmax(point_b), (size, size)))
    return jnp.where(point_a_available, jnp.sum(jnp.abs(point_a_pos - point_b_pos)), 0)
