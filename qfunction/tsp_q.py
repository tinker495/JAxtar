from puxle import TSP

from heuristic.tsp_heuristic import TSPHeuristic
from qfunction.q_base import QFromHeuristic


class TSPQ(QFromHeuristic):
    def __init__(self, puzzle: TSP):
        super().__init__(TSPHeuristic(puzzle))
