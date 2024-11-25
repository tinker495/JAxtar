import chex
import jax
from flax import linen as nn

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.rubikscube import RubiksCube

NODE_SIZE = 256


# Simba Residual Block
class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0):
        x = nn.LayerNorm()(x0)
        x = nn.Dense(self.node_size)(x0)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        return x + x0


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = (x - 0.5) * 2.0  # normalize to [-1, 1]
        x = nn.Dense(1000)(x)
        x = ResBlock(1000)(x)
        x = ResBlock(1000)(x)
        x = ResBlock(1000)(x)
        x = ResBlock(1000)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x


class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, Model(), init_params=init_params)

    def pre_process(self, current: RubiksCube.State, target: RubiksCube.State) -> chex.Array:
        flatten_face = current.faces.flatten()
        # Create a one-hot encoding of the flattened face
        one_hot = jax.nn.one_hot(flatten_face, num_classes=6).flatten()  # 6 colors in Rubik's Cube
        return one_hot
