import chex
import jax.numpy as jnp

from neural_util.modules import DTYPE
from puzzle.room import Room
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQFunctionBase


class RoomZeroshotQ(ZeroshotQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Room, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process_solve_config(self, solve_config: Room.SolveConfig) -> Room.SolveConfig:
        solve_config = solve_config.unpacking()
        pos = solve_config.TargetState.pos
        maze = solve_config.Maze
        flatten_maze = maze.reshape(-1)
        return jnp.concatenate([pos.astype(DTYPE), flatten_maze], axis=-1)

    def pre_process_state(self, state: Room.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pos = state.pos
        return pos.astype(DTYPE)
