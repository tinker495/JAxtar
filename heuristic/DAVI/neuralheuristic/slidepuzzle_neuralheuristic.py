import chex
import jax
import jax.numpy as jnp

from flax import linen as nn
from heuristic.DAVI.neuralheuristic_base import NeuralHeuristicBase
from puzzle.slidepuzzle import SlidePuzzle

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

class SlidePuzzleNeuralHeuristic(NeuralHeuristicBase):
    base_xy : chex.Array # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle):
        x = jnp.tile(jnp.arange(puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, puzzle.size, 1))
        y = jnp.tile(jnp.arange(puzzle.size)[jnp.newaxis, :, jnp.newaxis], (puzzle.size, 1, 1))
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)
        super().__init__(puzzle, model=Model())

    def pre_process(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        diff = self.to_2d(self._diff_pos(current, target)) # [n, n, 2]
        c_zero = self.to_2d(self._zero_pos(current)) # [n, n, 1]
        t_zero = self.to_2d(self._zero_pos(target)) # [n, n, 1]
        x = jnp.concatenate([diff, c_zero, t_zero], axis=-1) # [n, n, 4]
        x = jnp.expand_dims(x, axis=0)
        return x
    
    def to_2d(self, x: chex.Array) -> chex.Array:
        return x.reshape((self.puzzle.size, self.puzzle.size, x.shape[-1]))

    def _diff_pos(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the difference between the state and the target.
        """
        def to_xy(index):
            return index // self.puzzle.size, index % self.puzzle.size

        def pos(num, board):
            return to_xy(jnp.argmax(board == num))
        
        tpos = jnp.array([pos(i, target.board) for i in current.board], dtype=jnp.int8) # [16, 2]
        diff = self.base_xy - tpos # [16, 2]
        return diff
    
    def _zero_pos(self, current: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the zero position in the state.
        """
        return jnp.expand_dims(current.board == 0, axis=-1).astype(jnp.float32)