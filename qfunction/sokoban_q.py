from puxle import Sokoban

from heuristic.sokoban_heuristic import SokobanHeuristic
from qfunction.q_base import QFromHeuristic


class SokobanQ(QFromHeuristic):
    def __init__(self, puzzle: Sokoban):
        super().__init__(SokobanHeuristic(puzzle))
