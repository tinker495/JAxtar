from puxle import LightsOut

from heuristic.lightsout_heuristic import LightsOutHeuristic
from qfunction.q_base import QFromHeuristic


class LightsOutQ(QFromHeuristic):
    def __init__(self, puzzle: LightsOut):
        super().__init__(LightsOutHeuristic(puzzle))
