import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle, state_dataclass

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

    @state_dataclass
    class State:
        board: chex.Array

    @property
    def has_target(self) -> bool:
        return True

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return "□" if x == 0 else "■"

        def parser(state):
            return form.format(*map(to_char, self.from_uint8(state.board)))

        return parser

    def get_default_gen(self) -> callable:
        def gen():
            return self.State(board=self.to_uint8(jnp.ones(self.size**2, dtype=bool)))

        return gen

    def get_initial_state(self, key=None) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key=None) -> State:
        return self.State(board=self.to_uint8(jnp.zeros(self.size**2, dtype=bool)))

    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        board = self.from_uint8(state.board)
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
            return self.to_uint8(next_board), cost

        next_boards, costs = jax.vmap(map_fn, in_axes=(0, None))(actions, filled)
        return self.State(board=next_boards), costs

    def is_solved(self, state: State, target: State) -> bool:
        return self.is_equal(state, target)

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

    def _get_random_state(self, key, num_shuffle=8):
        """
        This function should return a random state.
        """
        init_state = self.get_target_state()

        def random_flip(carry, _):
            state, key = carry
            neighbor_states, _ = self.get_neighbours(state, filled=True)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(self.size**2))
            next_state = neighbor_states[idx]
            return (next_state, key), None

        (last_state, _), _ = jax.lax.scan(random_flip, (init_state, key), None, length=num_shuffle)
        return last_state

    def to_uint8(self, board: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        return jnp.packbits(board, axis=-1, bitorder="little")

    def from_uint8(self, board: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        return jnp.unpackbits(board, axis=-1, count=self.size**2, bitorder="little")

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        def img_func(state: "LightsOut.State"):
            imgsize = IMG_SIZE[0]
            # Create a background image with a dark gray base
            img = np.full((imgsize, imgsize, 3), fill_value=30, dtype=np.uint8)
            # Calculate the size of each cell in the grid
            cell_size = imgsize // self.size
            # Reshape the flat board state into a 2D array
            board = np.array(self.from_uint8(state.board)).reshape((self.size, self.size))
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

    def get_initial_state(self, key=None) -> LightsOut.State:
        return self._get_random_state(key, num_shuffle=50)
