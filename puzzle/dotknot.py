import chex
import jax
import jax.numpy as jnp

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puzzle.util import coloring_str

TYPE = jnp.uint8

COLOR_MAP = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 255, 0),  # Yellow
    3: (0, 0, 255),  # Blue
}


class DotKnot(Puzzle):

    size: int

    def get_solve_config_default_gen(self):
        def gen():
            return self.SolveConfig()

        return gen

    def define_solve_config_class(self) -> PuzzleState:
        @state_dataclass
        class SolveConfig:
            pass

        return SolveConfig

    def define_state_class(self) -> PuzzleState:
        str_parser = self.get_string_parser()
        board = jnp.zeros((self.size * self.size), dtype=TYPE)
        packed_board = self.pack_board(board)

        @state_dataclass
        class State:
            board: FieldDescriptor[TYPE, packed_board.shape, packed_board]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int, color_num: int = 4, **kwargs):
        assert size >= 4, "Size must be at least 4 for packing"
        self.size = size
        self.color_num = color_num
        super().__init__(**kwargs)

    def get_solve_config_string_parser(self):
        def parser(solve_config: "DotKnot.SolveConfig"):
            return ""

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            if x == 0:
                return " "
            elif x <= 2 * self.color_num:
                color_idx = int((x - 1) % self.color_num)
                return coloring_str("●", COLOR_MAP[color_idx])
            elif x <= 3 * self.color_num:
                color_idx = int((x - 1) % self.color_num)
                return coloring_str("■", COLOR_MAP[color_idx])
            else:
                return "?"  # for debug and target

        def parser(state, **kwargs):
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

    def get_initial_state(
        self, solve_config: "DotKnot.SolveConfig", key=jax.random.PRNGKey(128), data=None
    ) -> "DotKnot.State":
        return self._get_random_state(key)

    def get_solve_config(self, key=jax.random.PRNGKey(128), data=None) -> "DotKnot.SolveConfig":
        return self.SolveConfig()

    def get_neighbours(
        self, solve_config: "DotKnot.SolveConfig", state: "DotKnot.State", filled: bool = True
    ) -> tuple["DotKnot.State", chex.Array]:
        """
        This function returns neighbours and the cost of each move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Unpack the board for processing.
        unpacked_board = self.unpack_board(state.board)

        # Determine the smallest available color among {1, 2, ..., self.color_num}.
        colors = jnp.arange(1, self.color_num + 1, dtype=TYPE)
        # For each candidate color, check if that point is present in the board.
        available_mask = jnp.any(unpacked_board[None, :] == colors[:, None], axis=1)
        # If a color is not available, we replace it with a value greater than any valid color;
        # then, taking the minimum gives us the smallest valid color.
        selected_color = jnp.min(jnp.where(available_mask, colors, self.color_num + 1))
        # Define the 4 directional moves: up, down, left, right.
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

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

        def move(state, move_vector):
            point_idx = selected_color
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

        new_states, costs = jax.vmap(move, in_axes=(None, 0))(state, moves)
        return new_states, costs

    def is_solved(self, solve_config: "DotKnot.SolveConfig", state: "DotKnot.State") -> bool:
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
        match action:
            case 0:
                return "←"
            case 1:
                return "→"
            case 2:
                return "↑"
            case 3:
                return "↓"
            case _:
                raise ValueError(f"Invalid action: {action}")

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

    def _getBlankPosition(self, state: "DotKnot.State", idx: int):
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

    def get_solve_config_img_parser(self):
        def parser(solve_config: "DotKnot.SolveConfig"):
            pass

        return parser

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        def img_func(state: "DotKnot.State", **kwargs):
            imgsize = IMG_SIZE[0]
            img = np.zeros(IMG_SIZE + (3,), np.uint8)
            img[:] = (190, 190, 190)  # Background color (R144,G96,B8)
            img = cv2.rectangle(
                img,
                (int(imgsize * 0.03), int(imgsize * 0.03)),
                (int(imgsize - imgsize * 0.02), int(imgsize - imgsize * 0.02)),
                (50, 50, 50),
                -1,
            )
            board_flat = self.unpack_board(state.board)
            knot_max = 2 * self.color_num  # Values <= knot_max represent knots
            for idx, val in enumerate(board_flat):
                if val == 0:
                    continue
                stx = int(imgsize * 0.04 + (imgsize * 0.95 / self.size) * (idx % self.size))
                sty = int(imgsize * 0.04 + (imgsize * 0.95 / self.size) * (idx // self.size))
                bs = int(imgsize * 0.87 / self.size)
                center = (stx + bs // 2, sty + bs // 2)
                color = COLOR_MAP[(int(val) - 1) % len(COLOR_MAP)]
                if val <= knot_max:
                    # Draw knot as a filled circle
                    radius = int(bs * 0.6)
                    img = cv2.circle(img, center, radius, color, -1)
                else:
                    # Draw path as a filled rectangle
                    img = cv2.rectangle(img, (stx, sty), (stx + bs, sty + bs), color, -1)
            return img

        return img_func
