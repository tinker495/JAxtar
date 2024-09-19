import chex
import jax.numpy as jnp

from puzzle.lightsout import LightsOut

class LightsOutHeuristic:
    puzzle: LightsOut

    def __init__(self, puzzle: LightsOut):
        self.puzzle = puzzle

    def distance(self, current: LightsOut.State, target: LightsOut.State) -> float:
        """
        Get distance between current state and target state.
        """
        xor_state = jnp.logical_xor(self.puzzle.from_uint8(current.board), self.puzzle.from_uint8(target.board))
        sum_xor_state = jnp.sum(xor_state)
        return sum_xor_state / 5