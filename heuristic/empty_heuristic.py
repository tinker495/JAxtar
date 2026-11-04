from heuristic.heuristic_base import Heuristic
from puxle.core.puzzle_base import SolveConfigView
from xtructure import Xtructurable


class EmptyHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(self, solve_config: SolveConfigView, current: Xtructurable) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return 0.0
