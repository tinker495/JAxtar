import chex
from puxle import PancakeSorting

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.preprocessing import pancake_pre_process


class PancakeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: PancakeSorting, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: PancakeSorting.SolveConfig, current: PancakeSorting.State
    ) -> chex.Array:
        target_stack = None if self.is_fixed else solve_config.TargetState.stack
        return pancake_pre_process(current.stack, target_stack, self.puzzle.size, self.is_fixed)
