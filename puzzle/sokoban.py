from enum import Enum

import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

TYPE = jnp.uint8


class Object(Enum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    BOX = 3
    TARGET = 4
    PLAYER_ON_TARGET = 5
    BOX_ON_TARGET = 6


class Sokoban(Puzzle):
    size: int = 10

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for Sokoban using xtructure."""
        str_parser = self.get_string_parser()
        board = jnp.ones(self.size * self.size, dtype=TYPE)
        packed_board = self.pack_board(board)

        @state_dataclass
        class State:
            board: FieldDescriptor[TYPE, packed_board.shape, packed_board]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 10, **kwargs):
        self.size = size
        assert size == 10, "Boxoban dataset only supports size 10"
        super().__init__(**kwargs)

    @property
    def fixed_target(self) -> bool:
        return False

    def data_init(self):
        self.init_puzzles = jnp.load("puzzle/data/sokoban/init.npy")  # bring boxoban dataset here
        self.target_puzzles = jnp.load(
            "puzzle/data/sokoban/target.npy"
        )  # bring boxoban dataset here
        self.num_puzzles = self.init_puzzles.shape[0]

    def pack_board(self, board: jnp.ndarray) -> jnp.ndarray:
        """
        Pack a board array of shape (100,) with cell values in {0, 1, 2, 3}
        into a compact representation using 25 uint8 values.
        """
        reshaped = jnp.reshape(board, (-1, 4))  # shape: (25, 4)
        shifts = jnp.array([0, 2, 4, 6], dtype=board.dtype)
        packed = jnp.sum(reshaped * (2**shifts), axis=1).astype(jnp.uint8)
        return packed

    def unpack_board(self, packed: jnp.ndarray) -> jnp.ndarray:
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

    def get_data(self, key: jax.random.PRNGKey) -> tuple[chex.Array, chex.Array]:
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        return self.target_puzzles[idx, ...], self.init_puzzles[idx, ...]

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "Sokoban.State":
        # Initialize the board with the player, boxes, and walls from level1 and pack it.
        _, init_data = data
        return self.State(board=init_data)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        target_data, _ = data
        return self.SolveConfig(TargetState=self.State(board=target_data))

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State") -> bool:
        # Unpack boards for comparison.
        board = self.unpack_board(state.board)
        t_board = self.unpack_board(solve_config.TargetState.board)
        # Remove the player from the current board.
        rm_player = jnp.where(board == Object.PLAYER.value, Object.EMPTY.value, board)
        return jnp.all(rm_player == t_board)

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

    def get_solve_config_string_parser(self):
        def parser(solve_config: "Sokoban.SolveConfig", **kwargs):
            return solve_config.TargetState.str(solve_config=solve_config)

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            match x:
                case Object.EMPTY.value:
                    return " "
                case Object.WALL.value:
                    return colored("■", "white")
                case Object.PLAYER.value:
                    return colored("●", "red")
                case Object.BOX.value:
                    return colored("■", "yellow")
                case Object.TARGET.value:
                    return colored("x", "red")
                case Object.PLAYER_ON_TARGET.value:
                    return colored("ⓧ", "red")
                case Object.BOX_ON_TARGET.value:
                    return colored("■", "green")
                case _:
                    return "?"

        def parser(state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs):
            # Unpack the board before visualization.
            board = self.unpack_board(state.board)
            if solve_config is not None:
                goal = self.unpack_board(solve_config.TargetState.board)
                for i in range(self.size):
                    for j in range(self.size):
                        if goal[i * self.size + j] == Object.BOX.value:
                            match board[i * self.size + j]:
                                case Object.PLAYER.value:
                                    board = board.at[i * self.size + j].set(
                                        Object.PLAYER_ON_TARGET.value
                                    )
                                case Object.BOX.value:
                                    board = board.at[i * self.size + j].set(
                                        Object.BOX_ON_TARGET.value
                                    )
                                case Object.EMPTY.value:
                                    board = board.at[i * self.size + j].set(Object.TARGET.value)

            return form.format(*map(to_char, board))

        return parser

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State", filled: bool = True
    ) -> tuple["Sokoban.State", chex.Array]:
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
                return state, 1.0

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
                    valid_push = jnp.logical_and(valid_push, filled)

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
        costs = jnp.where(filled, costs, jnp.inf)
        return new_states, costs

    def _get_visualize_format(self):
        size = self.size
        top_border = "┏━" + "━━" * size + "┓\n"
        middle = ""
        for _ in range(size):
            middle += "┃ " + " ".join(["{:s}"] * size) + " ┃\n"
        bottom_border = "┗━" + "━━" * size + "┛"
        return top_border + middle + bottom_border

    def _getPlayerPosition(self, state: "Sokoban.State"):
        board = self.unpack_board(state.board)
        flat_index = jnp.argmax(board == Object.PLAYER.value)
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import os

        import cv2
        import numpy as np

        cell_w = IMG_SIZE[0] // self.size
        cell_h = IMG_SIZE[1] // self.size

        image_dir = os.path.join("puzzle", "data", "sokoban", "imgs")
        assets = {
            0: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "floor.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            1: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "wall.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            2: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "agent.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            3: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            4: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            5: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "agent_on_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            6: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box_on_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
        }

        def img_func(state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs):
            img = np.zeros(IMG_SIZE + (3,), np.uint8)

            board = np.array(self.unpack_board(state.board))
            if solve_config is not None:
                goal = np.array(self.unpack_board(solve_config.TargetState.board))
            else:
                goal = None
            for i in range(self.size):
                for j in range(self.size):
                    cell_val = int(board[i * self.size + j])
                    if (
                        goal is not None and goal[i * self.size + j] == Object.BOX.value
                    ):  # If this cell is marked as a target
                        match cell_val:
                            case Object.PLAYER.value:
                                asset = assets.get(Object.PLAYER_ON_TARGET.value)  # agent on target
                            case Object.BOX.value:
                                asset = assets.get(Object.BOX_ON_TARGET.value)  # box on target
                            case Object.EMPTY.value:
                                asset = assets.get(Object.TARGET.value)  # target floor (box target)
                            case _:
                                asset = assets.get(cell_val)
                    else:
                        asset = assets.get(cell_val)
                    if asset is not None:
                        img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = asset
                    else:
                        # Fallback: fill with a gray square if image not found
                        img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = (
                            127,
                            127,
                            127,
                        )
            return img

        return img_func

    def _get_box_positions(self, board: jnp.ndarray, number_of_boxes: int = 4) -> jnp.ndarray:
        """
        Get the positions of all boxes on the board.
        """
        flat_index = jnp.argsort(board == Object.BOX.value)[-number_of_boxes:]
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def _place_agent_randomly(self, board: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Place the agent randomly on the board.
        """
        board = self.unpack_board(board)
        box_xs, box_ys = self._get_box_positions(board)
        box_positions = jnp.expand_dims(jnp.stack([box_xs, box_ys], axis=1), 0)  # shape: (1, 4, 2)
        _near = jnp.expand_dims(
            jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), 1
        )  # shape: (4, 1, 2)
        near_positions = box_positions + _near  # shape: (4, 4, 2)
        near_positions = near_positions.reshape((-1, 2))  # shape: (16, 2)
        mask = jnp.ones(near_positions.shape[0])
        mask = jnp.where(near_positions[:, 0] < 0, 0, mask)
        mask = jnp.where(near_positions[:, 0] >= self.size, 0, mask)
        mask = jnp.where(near_positions[:, 1] < 0, 0, mask)
        mask = jnp.where(near_positions[:, 1] >= self.size, 0, mask)
        mask = jnp.where(
            board[near_positions[:, 0] * self.size + near_positions[:, 1]] != Object.EMPTY.value,
            0,
            mask,
        )
        prob = mask / jnp.sum(mask)
        idx = jax.random.choice(key, jnp.arange(near_positions.shape[0]), p=prob)
        new_board = board.at[near_positions[idx, 0] * self.size + near_positions[idx, 1]].set(
            Object.PLAYER.value
        )
        packed_board = self.pack_board(new_board)
        return packed_board

    def solve_config_to_state_transform(
        self, solve_config: "Sokoban.SolveConfig", key: jax.random.PRNGKey = None
    ) -> "Sokoban.State":
        """
        This function shoulde transformt the solve config to the state.
        """
        packed_board = self._place_agent_randomly(solve_config.TargetState.board, key)
        return self.State(board=packed_board)

    def hindsight_transform(
        self, solve_config: "Sokoban.SolveConfig", state: "Sokoban.State"
    ) -> "Sokoban.SolveConfig":
        """
        This function shoulde transformt the state to the solve config.
        """
        board = self.unpack_board(state.board)
        rm_player = jnp.where(board == Object.PLAYER.value, Object.EMPTY.value, board)
        solve_config.TargetState = self.State(board=self.pack_board(rm_player))
        return solve_config

    def get_inverse_neighbours(
        self, solve_config: "Sokoban.SolveConfig", state: "Sokoban.State", filled: bool = True
    ) -> tuple["Sokoban.State", chex.Array]:
        """
        Returns possible previous states and their associated costs.
        In Sokoban, inverse moves correspond to 'pulling' a box or simply moving back
        to the previous position if no box is involved.
        If an inverse move is not possible, it returns the original state with an infinite cost.
        """

        board = self.unpack_board(state.board)
        x, y = self._getPlayerPosition(state)
        current_pos = jnp.array([x, y])
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        def flat_idx(i, j):
            return i * self.size + j

        def is_empty(i, j):
            return board[flat_idx(i, j)] == Object.EMPTY.value

        def is_valid_pos(i, j):
            return jnp.logical_and(
                jnp.logical_and(i >= 0, i < self.size), jnp.logical_and(j >= 0, j < self.size)
            )

        def inv_move(direction):
            direction = direction.astype(current_pos.dtype)
            # Candidate previous player's position (i.e. where the player came from)
            prev_pos = current_pos - direction
            valid_prev = is_valid_pos(prev_pos[0], prev_pos[1])
            # For a pull move, consider the cell in front of the current player.
            front_pos = current_pos + direction
            valid_front = is_valid_pos(front_pos[0], front_pos[1])
            box_at_front = jnp.logical_and(
                valid_front, board[flat_idx(front_pos[0], front_pos[1])] == Object.BOX.value
            )
            empty_prev = jnp.logical_and(valid_prev, is_empty(prev_pos[0], prev_pos[1]))

            def invalid(_):
                return state, 1.0

            # Inverse pull: the forward push is inverted by pulling the box from ahead.
            # In a forward push, the player moved from prev_pos
            # to current_pos and pushed the box from current_pos to front_pos.
            # Here, we "pull" the box from front_pos back into current_pos while moving the player to prev_pos.
            def do_pull(_):
                new_board = board
                new_board = new_board.at[flat_idx(front_pos[0], front_pos[1])].set(
                    Object.EMPTY.value
                )
                new_board = new_board.at[flat_idx(current_pos[0], current_pos[1])].set(
                    Object.BOX.value
                )
                new_board = new_board.at[flat_idx(prev_pos[0], prev_pos[1])].set(
                    Object.PLAYER.value
                )
                return self.State(board=self.pack_board(new_board)), 1.0

            # Inverse simple move: simply move the player back if no box is involved.
            def do_simple(_):
                new_board = board
                new_board = new_board.at[flat_idx(current_pos[0], current_pos[1])].set(
                    Object.EMPTY.value
                )
                new_board = new_board.at[flat_idx(prev_pos[0], prev_pos[1])].set(
                    Object.PLAYER.value
                )
                return self.State(board=self.pack_board(new_board)), 1.0

            def branch_fn(_):
                # Only allow a move if the previous cell is empty.
                return jax.lax.cond(
                    empty_prev,
                    lambda _: jax.lax.cond(
                        box_at_front, do_pull, lambda _: do_simple(None), operand=None
                    ),
                    invalid,
                    operand=None,
                )

            return jax.lax.cond(valid_prev, branch_fn, invalid, operand=None)

        new_states, costs = jax.vmap(inv_move)(moves)
        costs = jnp.where(filled, costs, jnp.inf)
        return new_states, costs


class SokobanHard(Sokoban):
    def data_init(self):
        self.init_puzzles = jnp.load("puzzle/data/sokoban/init_hard.npy")
        self.target_puzzles = jnp.load("puzzle/data/sokoban/target_hard.npy")
        self.num_puzzles = self.init_puzzles.shape[0]


class SokobanDS(Sokoban):
    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        target_board, _ = data
        packed_board = self._place_agent_randomly(target_board, key)
        return self.SolveConfig(TargetState=self.State(board=packed_board))
