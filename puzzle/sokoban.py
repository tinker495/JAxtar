from enum import Enum

import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8


class Object(Enum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    BOX = 3


class Sokoban(Puzzle):
    size: int = 10

    @state_dataclass
    class State:
        board: chex.Array  # Now stores a packed board representation of shape (25,)

    def __init__(self, size: int = 10):
        self.size = size
        assert size == 10, "Boxoban dataset only supports size 10"
        self.init_puzzles = jnp.load("init.npy")  # bring boxoban dataset here
        self.target_puzzles = jnp.load("target.npy")  # bring boxoban dataset here
        self.num_puzzles = self.init_puzzles.shape[0]
        super().__init__()

    @property
    def has_target(self) -> bool:
        return True

    @staticmethod
    def pack_board(board: jnp.ndarray) -> jnp.ndarray:
        """
        Pack a board array of shape (100,) with cell values in {0, 1, 2, 3}
        into a compact representation using 25 uint8 values.
        """
        reshaped = jnp.reshape(board, (-1, 4))  # shape: (25, 4)
        shifts = jnp.array([0, 2, 4, 6], dtype=board.dtype)
        packed = jnp.sum(reshaped * (2**shifts), axis=1).astype(jnp.uint8)
        return packed

    @staticmethod
    def unpack_board(packed: jnp.ndarray) -> jnp.ndarray:
        """
        Unpack a compact board representation (25,)-shaped array back
        to a board of shape (100,) with cell values in {0, 1, 2, 3}.
        """
        shifts = jnp.array([0, 2, 4, 6], dtype=jnp.uint8)
        cells = jnp.stack([(packed >> shift) & 3 for shift in shifts], axis=1)
        board = jnp.reshape(cells, (-1,))
        return board

    def get_default_gen(self) -> callable:
        def gen():
            # Create a default flat board and pack it.
            board = jnp.ones(self.size**2, dtype=TYPE)
            packed_board = self.pack_board(board)
            return self.State(board=packed_board)

        return gen

    def get_initial_state(self, key=None) -> State:
        # Initialize the board with the player, boxes, and walls from level1 and pack it.
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        packed_board = self.init_puzzles[idx, ...]
        return self.State(board=packed_board)

    def get_target_state(self, key=None) -> State:
        # Define the target state and pack it.
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        packed_board = self.target_puzzles[idx, ...]
        return self.State(board=packed_board)

    def is_solved(self, state: State, target: State) -> bool:
        # Unpack boards for comparison.
        board = self.unpack_board(state.board)
        t_board = self.unpack_board(target.board)
        # Remove the player from the current board.
        rm_player = jnp.where(board == Object.PLAYER.value, Object.EMPTY.value, board)
        return jnp.all(rm_player == t_board)

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        if action == 0:
            return "←"
        elif action == 1:
            return "→"
        elif action == 2:
            return "↑"
        elif action == 3:
            return "↓"
        else:
            raise ValueError(f"Invalid action: {action}")

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            if x == Object.EMPTY.value:  # empty
                return " "
            elif x == Object.WALL.value:  # wall
                return colored("■", "white")
            elif x == Object.PLAYER.value:  # player
                return colored("●", "red")
            elif x == Object.BOX.value:  # box
                return colored("■", "yellow")
            else:
                return "?"

        def parser(state):
            # Unpack the board before visualization.
            board = self.unpack_board(state.board)
            return form.format(*map(to_char, board))

        return parser

    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        Returns neighbour states along with the cost for each move.
        If a move isn't possible, it returns the original state with an infinite cost.
        """
        # Unpack the board so that we work on a flat representation.
        board = self.unpack_board(state.board)
        x, y = self._getPlayerPosition(state)
        current_pos = jnp.array([x, y])
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        # Helper: convert (row, col) to flat index
        def flat_idx(i, j):
            return i * self.size + j

        def is_empty(i, j):
            return board[flat_idx(i, j)] == Object.EMPTY.value

        def is_valid_pos(i, j):
            return jnp.logical_and(
                jnp.logical_and(i >= 0, i < self.size), jnp.logical_and(j >= 0, j < self.size)
            )

        def move(direction):
            new_pos = (current_pos + direction).astype(current_pos.dtype)
            new_x, new_y = new_pos[0], new_pos[1]
            valid_move = is_valid_pos(new_x, new_y)

            def invalid_case(_):
                return state, jnp.inf

            def process_move(_):
                target = board[flat_idx(new_x, new_y)]
                # Case when target cell is empty: simply move the player.

                def move_empty(_):
                    new_board = board.at[flat_idx(current_pos[0], current_pos[1])].set(
                        Object.EMPTY.value
                    )
                    new_board = new_board.at[flat_idx(new_x, new_y)].set(Object.PLAYER.value)
                    # Pack the updated board.
                    return self.State(board=self.pack_board(new_board)), 1.0

                # Case when target cell contains a box: attempt to push it.
                def push_box(_):
                    push_pos = (new_pos + direction).astype(current_pos.dtype)
                    push_x, push_y = push_pos[0], push_pos[1]
                    valid_push = jnp.logical_and(
                        is_valid_pos(push_x, push_y), is_empty(push_x, push_y)
                    )

                    def do_push(_):
                        new_board = board.at[flat_idx(current_pos[0], current_pos[1])].set(
                            Object.EMPTY.value
                        )
                        new_board = new_board.at[flat_idx(new_x, new_y)].set(Object.PLAYER.value)
                        new_board = new_board.at[flat_idx(push_x, push_y)].set(Object.BOX.value)
                        return self.State(board=self.pack_board(new_board)), 1.0

                    return jax.lax.cond(valid_push, do_push, invalid_case, operand=None)

                return jax.lax.cond(
                    jnp.equal(target, Object.EMPTY.value),
                    move_empty,
                    lambda _: jax.lax.cond(
                        jnp.equal(target, Object.BOX.value), push_box, invalid_case, operand=None
                    ),
                    operand=None,
                )

            return jax.lax.cond(valid_move, process_move, invalid_case, operand=None)

        new_states, costs = jax.vmap(move)(moves)
        return new_states, costs

    def _get_visualize_format(self):
        size = self.size
        top_border = "┏━" + "━━" * size + "┓\n"
        middle = ""
        for _ in range(size):
            middle += "┃ " + " ".join(["{:s}"] * size) + " ┃\n"
        bottom_border = "┗━" + "━━" * size + "┛"
        return top_border + middle + bottom_border

    def _getPlayerPosition(self, state: State):
        board = self.unpack_board(state.board)
        flat_index = jnp.argmax(board == Object.PLAYER.value)
        return jnp.unravel_index(flat_index, (self.size, self.size))
