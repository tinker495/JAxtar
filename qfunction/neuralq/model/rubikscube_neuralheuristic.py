import chex
import jax
import jax.numpy as jnp

from puzzle.rubikscube import RubiksCube
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

NODE_SIZE = 256


class RubiksCubeNeuralHeuristic(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(self, current: RubiksCube.State, target: RubiksCube.State) -> chex.Array:
        flatten_face = current.faces.flatten()
        # Create a one-hot encoding of the flattened face
        one_hot = jax.nn.one_hot(flatten_face, num_classes=6).flatten()  # 6 colors in Rubik's Cube
        return jnp.expand_dims(one_hot, axis=0)