from typing import Any, Optional

import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.maze import Maze


class MazeHeuristic(Heuristic):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def distance(
        self, solve_config: Maze.SolveConfig, current: Maze.State, params: Optional[Any] = None
    ) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - solve_config.TargetState.pos.astype(int)))
