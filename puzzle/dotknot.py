import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8
COLORS = [
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
    "white",
]  # 13 colors


class DotKnot(Puzzle):

    size: int

    @state_dataclass
    class State:
        board: chex.Array
        # 0 : empty
        # 1 ~ color_num : point a
        # color_num + 1 ~ 2 * color_num + 1 : point b
        # 2 * color_num + 2 ~ 3 * color_num + 2 : lines

    @property
    def has_target(self) -> bool:
        return False

    def __init__(self, size: int, color_num: int = 4):
        assert size >= 4, "Size must be at least 4 for packing"
        self.size = size
        self.color_num = color_num
        super().__init__()

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            if x == 0:
                return " "
            elif x <= 2 * self.color_num:
                color_idx = (x - 1) % self.color_num
                return colored("●", COLORS[color_idx])
            elif x <= 3 * self.color_num:
                color_idx = (x - 1) % self.color_num
                return colored("■", COLORS[color_idx])
            else:
                return "?"  # for debug and target

        def parser(state):
            unpacked = self.unpack_board(state.board)
            return form.format(*map(to_char, unpacked))

        return parser

    def pack_board(self, board: jnp.ndarray) -> jnp.ndarray:
        """
        Pack a board array of shape (size * size) with cell values in 0 ~ 16
        into a compact representation.
        """
        if board.shape[0] % 2 == 1:
            board = jnp.concatenate([board, jnp.array([0], dtype=board.dtype)])
        reshaped = jnp.reshape(board, (-1, 2))  # reshape to (num_pairs, 2)
        shifts = jnp.array([0, 4], dtype=board.dtype)
        packed = jnp.sum(reshaped * (2**shifts), axis=1).astype(jnp.uint8)
        return packed

    def unpack_board(self, packed: jnp.ndarray) -> jnp.ndarray:
        """
        Unpack a compact board representation back to a board of shape (size * size)
        with cell values in 0 ~ 16.
        """
        shifts = jnp.array([0, 4], dtype=jnp.uint8)
        cells = jnp.stack([(packed >> shift) & 15 for shift in shifts], axis=1)
        board = jnp.reshape(cells, (-1,))
        if board.shape[0] % 2 == 1:
            board = board[:-1]
        return board

    def get_default_gen(self) -> callable:
        def gen():
            board = jnp.zeros((self.size * self.size), dtype=TYPE)
            packed_board = self.pack_board(board)
            return self.State(board=packed_board)

        return gen

    def get_initial_state(self, key=jax.random.PRNGKey(128)) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key=None) -> State:
        board = jnp.full((self.size * self.size), -1, dtype=TYPE)
        packed_board = self.pack_board(board)
        return self.State(board=packed_board)  # this puzzle no target

    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return neighbours, and the cost of the move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Define possible moves: up, down, left, right
        points = jnp.arange(self.color_num, dtype=TYPE) + 1  # 1 ~ color_num: point a, etc.
        moves = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        points, moves_idx = jnp.meshgrid(points, jnp.arange(4))
        points = points.reshape(-1)
        moves = moves[moves_idx.reshape(-1)]

        # Unpack the board for processing
        unpacked_board = self.unpack_board(state.board)

        def is_valid(new_pos, color_idx):
            index = new_pos[0] * self.size + new_pos[1]
            not_blocked = unpacked_board[index] == 0
            new_pos_color_idx = (unpacked_board[index] - 1) % self.color_num
            new_pos_is_point = unpacked_board[index] <= 2 * self.color_num
            is_merge = (new_pos_color_idx == color_idx) & new_pos_is_point & ~not_blocked
            valid = (
                (new_pos >= 0).all()
                & (new_pos < self.size).all()
                & (not_blocked | is_merge)
                & filled
            )
            return is_merge, valid

        def point_move(board, pos, new_pos, point_idx, color_idx, is_merge):
            flat_index = pos[0] * self.size + pos[1]
            next_flat_index = new_pos[0] * self.size + new_pos[1]
            board = jnp.where(
                is_merge,
                board.at[next_flat_index].set(color_idx + 2 * self.color_num + 1),
                board.at[next_flat_index].set(point_idx),
            )
            return board.at[flat_index].set(color_idx + 2 * self.color_num + 1)

        def move(state, point, move_vector):
            point_idx = point
            color_idx = (point_idx - 1) % self.color_num
            available, pos = self._getBlankPosition(state, point_idx)
            new_pos = (pos + move_vector).astype(TYPE)
            is_merge, valid_move = is_valid(new_pos, color_idx)
            valid_move = valid_move & available
            new_board = jax.lax.cond(
                valid_move,
                lambda _: point_move(unpacked_board, pos, new_pos, point_idx, color_idx, is_merge),
                lambda _: unpacked_board,
                operand=None,
            )
            new_state = self.State(board=self.pack_board(new_board))
            cost = jnp.where(valid_move, 1.0, jnp.inf)
            return new_state, cost

        new_states, costs = jax.vmap(move, in_axes=(None, 0, 0))(state, points, moves)
        return new_states, costs

    def is_solved(self, state: State, target: State) -> bool:
        unpacked = self.unpack_board(state.board)
        empty = jnp.all(unpacked == 0)  # ALL empty is not solved condition
        gr = jnp.greater_equal(unpacked, 1)  # ALL point a is solved condition
        le = jnp.less_equal(unpacked, self.color_num * 2)  # ALL point b is solved condition
        points = gr & le
        no_point = ~jnp.any(points)  # if there is no point, it is solved
        return no_point & ~empty

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        color = action // 4
        direction = action % 4
        if direction == 0:
            return f"{COLORS[color]}↑"
        elif direction == 1:
            return f"{COLORS[color]}↓"
        elif direction == 2:
            return f"{COLORS[color]}←"
        elif direction == 3:
            return f"{COLORS[color]}→"

    def _get_visualize_format(self):
        size = self.size
        form = "┏━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s} "
            form += "┃" + "\n"
        form += "┗━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def _getBlankPosition(self, state: State, idx: int):
        unpacked_board = self.unpack_board(state.board)
        one_hot = unpacked_board == idx
        available = jnp.any(one_hot)
        flat_index = jnp.argmax(one_hot)
        pos = jnp.stack(jnp.unravel_index(flat_index, (self.size, self.size)))
        return available, pos

    def _get_random_state(self, key, num_shuffle=30):
        """
        This function should return a random state.
        """
        init_board = jnp.zeros((self.size * self.size), dtype=TYPE)

        def _while_loop(val):
            board, key, idx = val
            key, subkey = jax.random.split(key)
            pos = jax.random.randint(
                subkey, minval=0, maxval=self.size - 2, shape=(2,)
            ) + jnp.array([1, 1])
            random_index = pos[0] * self.size + pos[1]
            is_already_filled = board[random_index] != 0
            board = jax.lax.cond(
                is_already_filled,
                lambda _: board,
                lambda _: board.at[random_index].set(idx),
                operand=None,
            )
            next_idx = jnp.where(is_already_filled, idx, idx + 1)
            return board, key, next_idx

        board, _, _ = jax.lax.while_loop(
            lambda val: val[2] < self.color_num * 2 + 1, _while_loop, (init_board, key, 1)
        )
        packed_board = self.pack_board(board)
        return self.State(board=packed_board)
