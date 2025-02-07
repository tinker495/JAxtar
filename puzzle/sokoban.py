from enum import Enum

import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8

level1 = [
    "##########",
    "# @      #",
    "# $    . #",
    "#  $# .  #",
    "#  .#$  # ",
    "# . # $ # ",
    "#        #",
    "##########",
    "##########",
    "##########",
]

level2 = [
    "##########",
    "#        #",
    "#$ #   . #",
    "# # $ # .#",
    "#  .# $  #",
    "# @ # . $#",
    "#        #",
    "##########",
    "##########",
    "##########",
]

# Puzzles are grouped into sets of one thousand puzzles, each set encoded as a text file.
# Each puzzle is a 10 by 10 ASCII string which uses the following encoding:
# '#' for wall, '@' for the player character, '$' for a box, and '.' for a goal position.


class Object(Enum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    BOX = 3


def convert_level(level: list[str]) -> tuple[chex.Array, chex.Array]:
    init_level = jnp.zeros((10 * 10,), dtype=TYPE)
    target_level = jnp.zeros((10 * 10,), dtype=TYPE)
    for i, row in enumerate(level):
        for j, col in enumerate(row):
            idx = i * 10 + j
            if col == "#":
                init_level = init_level.at[idx].set(Object.WALL.value)
                target_level = target_level.at[idx].set(Object.WALL.value)
            elif col == "@":
                init_level = init_level.at[idx].set(Object.PLAYER.value)
            elif col == "$":
                init_level = init_level.at[idx].set(Object.BOX.value)
            elif col == ".":
                target_level = target_level.at[idx].set(Object.BOX.value)
    return init_level, target_level


level1_init, level1_target = convert_level(level1)
level2_init, level2_target = convert_level(level2)


class Sokoban(Puzzle):
    size: int = 10

    @state_dataclass
    class State:
        board: chex.Array
        # 0: empty
        # 1: wall
        # 2: player
        # 3: box
        # goal is not included, it could be using the box on target states
        # total 2 bits
        # 2 x size x size

    def __init__(self, size: int = 10):
        self.size = size
        assert size == 10, "Boxoban dataset only supports size 10"
        super().__init__()

    @property
    def has_target(self) -> bool:
        return True

    def get_default_gen(self) -> callable:
        def gen():
            return self.State(board=jnp.ones(self.size**2, dtype=bool))

        return gen

    def get_initial_state(self, key=None) -> State:
        # Initialize the board with the player, boxes, and walls
        return self.State(board=level1_init)

    def get_target_state(self, key=None) -> State:
        # Define the target state where all boxes are on goal positions
        return self.State(board=level1_target)

    def is_solved(self, state: State, target: State) -> bool:
        rm_player = jnp.where(state.board == Object.PLAYER.value, Object.EMPTY.value, state.board)
        return jnp.all(rm_player == target.board)

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
            return form.format(*map(to_char, state.board))

        return parser

    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        Returns neighbour states along with the cost for each move.
        If a move isn't possible, it returns the original state with an infinite cost.
        """
        # Retrieve player's current position and the board state
        x, y = self._getPlayerPosition(state)
        current_pos = jnp.array([x, y])
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        board = state.board

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
                # Case when target cell is empty: simply move the player

                def move_empty(_):
                    new_board = board.at[flat_idx(current_pos[0], current_pos[1])].set(
                        Object.EMPTY.value
                    )
                    new_board = new_board.at[flat_idx(new_x, new_y)].set(Object.PLAYER.value)
                    return self.State(board=new_board), 1.0

                # Case when target cell contains a box: attempt to push it
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
                        return self.State(board=new_board), 1.0

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
        flat_index = jnp.argmax(state.board == Object.PLAYER.value)
        return jnp.unravel_index(flat_index, (self.size, self.size))
