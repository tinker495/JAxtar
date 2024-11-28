import chex
import jax
from flax import linen as nn

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.rubikscube import RubiksCube

NODE_SIZE = 256


class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(self, current: RubiksCube.State, target: RubiksCube.State) -> chex.Array:
        flatten_face = current.faces.flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        one_hot = jax.nn.one_hot(flatten_face, num_classes=6).flatten()  # 6 colors in Rubik's Cube
        return (one_hot - 0.5) * 2.0  # normalize to [-1, 1]
