import chex
import jax
import jax.numpy as jnp

from flax import linen as nn
from puzzle.slidepuzzle import SlidePuzzle

NODE_SIZE = int(1e3)

class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(NODE_SIZE)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Dummyneuralheuristic:

    def __init__(self, puzzle: SlidePuzzle):
        self.puzzle = puzzle
        self.model = Model()
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2 * self.puzzle.size**2,)))

    def distance(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        conc = jnp.concatenate([current.board, target.board], axis=0)[jnp.newaxis, ...]
        return self.model.apply(self.params, conc).squeeze()