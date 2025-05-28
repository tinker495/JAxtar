import chex

from neural_util.modules import DTYPE
from puzzle.room import Room
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQFunctionBase


class RoomZeroshotQ(ZeroshotQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Room, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process_solve_config(self, solve_config: Room.SolveConfig) -> Room.SolveConfig:
        pos = solve_config.TargetState.pos
        return pos.astype(DTYPE)

    def pre_process_state(self, state: Room.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pos = state.pos
        return pos.astype(DTYPE)
