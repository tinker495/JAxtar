from puxle import Puzzle

from heuristic.heuristic_base import Heuristic


class EmptyHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return 0.0
