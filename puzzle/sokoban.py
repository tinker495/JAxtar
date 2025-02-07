import chex

# import jax
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

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            if x == 0:  # empty
                return " "
            elif x == 1:  # wall
                return "■"
            elif x == 2:  # player
                return colored("●", "red")
            elif x == 3:  # box
                return colored("■", "yellow")
            else:
                return "?"

        def parser(state):
            return form.format(*map(to_char, self.from_uint8(state.board)))

        return parser

    def _get_visualize_format(self):
        size = self.size
        top_border = "┏" + "━━━" * size + "┓\n"
        middle = ""
        for _ in range(size):
            middle += "┃ " + " ".join(["{:s}"] * size) + " ┃\n"
        bottom_border = "┗" + "━━━" * size + "┛"
        return top_border + middle + bottom_border

    def get_default_gen(self) -> callable:
        def gen():
            return self.State(board=jnp.ones(self.size**2, dtype=bool))

        return gen

    def get_initial_state(self, key=None) -> State:
        # Initialize the board with the player, boxes, and walls
        board = jnp.zeros((self.size, self.size), dtype=TYPE)
        return self.State(board=board)

    def get_target_state(self, key=None) -> State:
        # Define the target state where all boxes are on goal positions
        board = jnp.zeros((self.size, self.size), dtype=TYPE)
        return self.State(board=board)
