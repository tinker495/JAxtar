from puxle import RubiksCube

from heuristic.rubikscube_heuristic import RubiksCubeHeuristic
from qfunction.q_base import QFromHeuristic


class RubiksCubeQ(QFromHeuristic):
    def __init__(self, puzzle: RubiksCube):
        super().__init__(RubiksCubeHeuristic(puzzle))
