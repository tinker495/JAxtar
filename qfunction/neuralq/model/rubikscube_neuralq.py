import chex
import jax

from puzzle.rubikscube import RubiksCube
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

NODE_SIZE = 256


class RubiksCubeNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def solve_config_pre_process(self, solve_config: RubiksCube.SolveConfig) -> chex.Array:
        state = solve_config.TargetState
        flattened_face = self.puzzle.unpack_faces(state.faces).flatten()
        one_hot = jax.nn.one_hot(flattened_face, num_classes=6).flatten()
        return (one_hot - 0.5) * 2.0

    def state_pre_process(self, state: RubiksCube.State) -> chex.Array:
        flattened_face = self.puzzle.unpack_faces(state.faces).flatten()
        one_hot = jax.nn.one_hot(flattened_face, num_classes=6).flatten()
        return (one_hot - 0.5) * 2.0
