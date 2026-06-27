from puxle import PancakeSorting

from heuristic.pancake_heuristic import PancakeHeuristic
from qfunction.q_base import QFromHeuristic


class PancakeQ(QFromHeuristic):
    def __init__(self, puzzle: PancakeSorting):
        super().__init__(PancakeHeuristic(puzzle))
