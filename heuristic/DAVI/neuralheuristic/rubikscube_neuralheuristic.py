import chex
import jax
import jax.numpy as jnp

from flax import linen as nn
from heuristic.DAVI.neuralheuristic_base import NeuralHeuristicBase
from puzzle.rubikscube import RubiksCube

NODE_SIZE = 256

class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        # [4, 4, 6] -> conv
        x = nn.Conv(512, (3, 3), strides=1, padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), strides=1)(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    base_xy : chex.Array # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(self, current: RubiksCube.State, target: RubiksCube.State) -> chex.Array:
        flatten_face = jnp.expand_dims(current.faces.flatten(), axis=0)
        return flatten_face
