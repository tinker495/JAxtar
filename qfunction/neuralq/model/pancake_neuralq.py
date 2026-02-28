import chex
from puxle import PancakeSorting

from neural_util.preprocessing import pancake_pre_process
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class PancakeNeuralQ(NeuralQFunctionBase):
    is_fixed: bool = True

    def __init__(self, puzzle: PancakeSorting, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: PancakeSorting.SolveConfig, current: PancakeSorting.State
    ) -> chex.Array:
        target_stack = solve_config.TargetState.stack
        return pancake_pre_process(current.stack, target_stack, self.puzzle.size, self.is_fixed)


class PancakeRandomNeuralQ(PancakeNeuralQ):
    is_fixed: bool = False
