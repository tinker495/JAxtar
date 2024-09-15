import chex
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from puzzle.lightsout import LightsOut

class LightsOutHeuristic:
    puzzle: LightsOut
    cross_kernel: chex.Array

    def __init__(self, puzzle: LightsOut):
        self.puzzle = puzzle
        self.cross_kernel = jnp.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=jnp.float32)

    def distance(self, current: LightsOut.State, target: LightsOut.State) -> float:
        """
        Get distance between current state and target state.
        """
        xor_state = jnp.logical_xor(current.board, target.board)
        cross_kernel_count = self._count_cross_kernels(xor_state)
        
        return cross_kernel_count.astype(jnp.float32)

    def _count_cross_kernels(self, state):
        """Get count of cross kernels in the state."""
        state_2d = state.reshape(self.puzzle.size, self.puzzle.size)
        convolved = convolve2d(state_2d, self.cross_kernel, mode='same')
        
        cross_count = jnp.sum((convolved >= 3) & (state_2d == 1))
        return cross_count
