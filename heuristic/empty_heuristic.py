import chex

from heuristic.heuristic_base import Heuristic


class EmptyHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(self, current: chex.Array, target: chex.Array) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return 0.0
