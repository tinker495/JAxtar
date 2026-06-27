from puxle import PDDL

from heuristic.pddl_heuristic import PDDLHeuristic
from qfunction.q_base import QFromHeuristic


class PDDLQ(QFromHeuristic):
    def __init__(self, puzzle: PDDL):
        super().__init__(PDDLHeuristic(puzzle))
