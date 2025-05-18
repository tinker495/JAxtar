import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puzzle.util import from_uint8, to_uint8

TYPE = jnp.uint8


def action_to_char(action: int) -> str:
    """
    This function should return a string representation of the action.
    0~9 -> 0~9
    10~35 -> a~z
    36~61 -> A~Z
    """
    if action < 10:
        return colored(str(action), "light_yellow")
    elif action < 36:
        return colored(chr(action + 87), "light_yellow")
    else:
        return colored(chr(action + 29), "light_yellow")


class LightsOut(Puzzle):

    size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for LightsOut using xtructure."""
        str_parser = self.get_string_parser()
        board = jnp.zeros((self.size * self.size), dtype=bool)
        packed_board = to_uint8(board)
        size = self.size

        @state_dataclass
        class State:
            board: FieldDescriptor[TYPE, packed_board.shape]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            def packing(self) -> "LightsOut.State":
                board = to_uint8(self.board)
                return State(board=board)

            def unpacking(self) -> "LightsOut.State":
                board = from_uint8(self.board, (size * size,))
                return State(board=board)

        return State

    def __init__(self, size: int, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return "□" if x == 0 else "■"

        def parser(state: "LightsOut.State", **kwargs):
            return form.format(*map(to_char, state.unpacking().board))

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "LightsOut.State":
        return self._get_suffled_state(solve_config, solve_config.TargetState, key, num_shuffle=8)

    def get_target_state(self, key=None) -> "LightsOut.State":
        return self.State(board=jnp.zeros(self.size**2, dtype=bool)).packing()

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        return self.SolveConfig(TargetState=self.get_target_state(key))

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "LightsOut.State", filled: bool = True
    ) -> tuple["LightsOut.State", chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        board = state.unpacking().board
        # actions - combinations of range(size) with 2 elements
        actions = jnp.stack(
            jnp.meshgrid(jnp.arange(self.size), jnp.arange(self.size), indexing="ij"), axis=-1
        ).reshape(-1, 2)

        def flip(board, action):
            x, y = action
            xs = jnp.clip(jnp.array([x, x, x + 1, x - 1, x]), 0, self.size - 1)
            ys = jnp.clip(jnp.array([y, y + 1, y, y, y - 1]), 0, self.size - 1)
            idxs = xs * self.size + ys
            return board.at[idxs].set(jnp.logical_not(board[idxs]))

        def map_fn(action, filled):
            next_board, cost = jax.lax.cond(
                filled, lambda _: (flip(board, action), 1.0), lambda _: (board, jnp.inf), None
            )
            next_state = self.State(board=next_board).packing()
            return next_state, cost

        next_states, costs = jax.vmap(map_fn, in_axes=(0, None))(actions, filled)
        return next_states, costs

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "LightsOut.State") -> bool:
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return action_to_char(action)

    def _get_visualize_format(self):
        size = self.size
        action_idx = 0
        form = "┏━"
        form += "━Board".center((size - 1) * 2, "━")
        form += "━━┳━"
        form += "━Actions".center((size - 1) * 2, "━")
        form += "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s} "
            form += "┃ "
            for j in range(size):
                form += action_to_char(action_idx) + " "
                action_idx += 1
            form += "┃"
            form += "\n"
        form += "┗━"
        form += "━━" * (size - 1)
        form += "━━┻━"
        form += "━━" * (size - 1)
        form += "━━┛"
        return form

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        def img_func(state: "LightsOut.State", **kwargs):
            imgsize = IMG_SIZE[0]
            # Create a background image with a dark gray base
            img = np.full((imgsize, imgsize, 3), fill_value=30, dtype=np.uint8)
            # Calculate the size of each cell in the grid
            cell_size = imgsize // self.size
            # Reshape the flat board state into a 2D array
            board = np.array(state.unpacking().board).reshape(self.size, self.size)
            # Define colors in BGR: light on → bright yellow, light off → black, and grid lines → gray
            on_color = (255, 255, 0)  # Yellow
            off_color = (0, 0, 0)  # Black
            grid_color = (50, 50, 50)  # Gray for grid lines
            # Draw each cell of the puzzle
            for i in range(self.size):
                for j in range(self.size):
                    top_left = (j * cell_size, i * cell_size)
                    bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
                    # Use lit color if the cell is "on", otherwise use off color
                    cell_color = on_color if board[i, j] else off_color
                    img = cv2.rectangle(img, top_left, bottom_right, cell_color, thickness=-1)
                    img = cv2.rectangle(img, top_left, bottom_right, grid_color, thickness=1)
            return img

        return img_func


class LightsOutHard(LightsOut):
    """
    This class is a extension of LightsOut, it will generate the hardest state for the puzzle.
    """

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> LightsOut.State:
        return self._get_suffled_state(solve_config, solve_config.TargetState, key, num_shuffle=50)


class LightsOutRandom(LightsOut):
    """
    This class is a extension of LightsOut, it will generate the random state for the puzzle.
    """

    @property
    def fixed_target(self) -> bool:
        return False

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        solve_config = super().get_solve_config(key, data)
        solve_config.TargetState = self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=1000
        )
        return solve_config

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> LightsOut.State:
        return self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=1000
        )
