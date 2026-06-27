from puxle import SlidePuzzle

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from qfunction.q_base import QFromHeuristic


class SlidePuzzleQ(QFromHeuristic):
    def __init__(self, puzzle: SlidePuzzle):
        super().__init__(SlidePuzzleHeuristic(puzzle))
