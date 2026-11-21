import jax
import jax.numpy as jnp
from puxle import Sokoban

from heuristic.heuristic_base import Heuristic


class SokobanHeuristic(Heuristic):
    def __init__(self, puzzle: Sokoban):
        super().__init__(puzzle)

    def distance(self, solve_config: Sokoban.SolveConfig, current: Sokoban.State) -> jax.Array:
        target_board = solve_config.GoalSpec.board_unpacked
        target_box_count = jnp.sum(target_board == Sokoban.Object.BOX.value)

        current_board = current.board_unpacked
        matching_boxes = jnp.sum(
            jnp.logical_and(
                current_board == Sokoban.Object.BOX.value, target_board == Sokoban.Object.BOX.value
            )
        )

        return ((target_box_count - matching_boxes) * 5).astype(jnp.float32)
