from typing import Any, Optional

from heuristic.heuristic_base import Heuristic
from puzzle.puzzle_base import Puzzle


class EmptyHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State, params: Optional[Any] = None
    ) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return 0.0
