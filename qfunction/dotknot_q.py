from puxle import DotKnot

from heuristic.dotknot_heuristic import DotKnotHeuristic
from qfunction.q_base import QFromHeuristic


class DotKnotQ(QFromHeuristic):
    def __init__(self, puzzle: DotKnot):
        super().__init__(DotKnotHeuristic(puzzle))
