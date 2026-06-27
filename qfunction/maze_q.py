from puxle import Maze

from heuristic.maze_heuristic import MazeHeuristic
from qfunction.q_base import QFromHeuristic


class MazeQ(QFromHeuristic):
    def __init__(self, puzzle: Maze):
        super().__init__(MazeHeuristic(puzzle))
