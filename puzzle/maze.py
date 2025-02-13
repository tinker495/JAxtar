import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint16


class Maze(Puzzle):

    size: int
    maze: chex.Array

    @state_dataclass
    class State:
        pos: chex.Array

    @state_dataclass
    class SolveConfig:
        TargetState: "Maze.State"
        Maze: chex.Array

    def __init__(self, size: int, p=0.3, key=jax.random.PRNGKey(0), **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def get_solve_config_string_parser(self):
        def parser(solve_config: "Maze.SolveConfig", **kwargs):
            return solve_config.TargetState.str(solve_config=solve_config)

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            match x:
                case 0:
                    return " "
                case 1:
                    return "■"
                case 2:
                    return colored("●", "red")  # player
                case 3:
                    return colored("x", "red")  # target
                case 4:
                    return colored("●", "green")  # player on target
                case _:
                    raise ValueError(f"Invalid value: {x}")

        def parser(state: "Maze.State", solve_config: "Maze.SolveConfig" = None, **kwargs):
            if solve_config is not None:
                maze_with_pos = self.from_uint8(solve_config.Maze)
                maze_with_pos = maze_with_pos.at[
                    solve_config.TargetState.pos[0] * self.size + solve_config.TargetState.pos[1]
                ].set(3)
            else:
                maze_with_pos = jnp.zeros((self.size**2), dtype=jnp.bool_)
            idx = state.pos[0] * self.size + state.pos[1]
            if maze_with_pos[idx] == 0:
                maze_with_pos = maze_with_pos.at[idx].set(2)
            else:
                maze_with_pos = maze_with_pos.at[idx].set(4)
            return form.format(*map(to_char, maze_with_pos))

        return parser

    def get_solve_config_default_gen(self):
        def gen():
            maze = jnp.zeros((self.size**2), dtype=jnp.bool_)
            maze = self.to_uint8(maze)
            return self.SolveConfig(
                TargetState=self.State(pos=jnp.array([0, 0], dtype=TYPE)), Maze=maze
            )

        return gen

    def get_default_gen(self) -> callable:
        def gen():
            return self.State(pos=jnp.array([0, 0], dtype=TYPE))

        return gen

    def get_initial_state(
        self, solve_config: "Maze.SolveConfig", key=jax.random.PRNGKey(0)
    ) -> State:
        return self._get_random_state(solve_config.Maze, key)

    def get_solve_config(self, key=jax.random.PRNGKey(128)) -> Puzzle.SolveConfig:
        maze = jax.random.bernoulli(key, p=jnp.float32(0.3), shape=(self.size**2,)).astype(
            jnp.uint8
        )
        maze = self.to_uint8(maze)
        key1, _ = jax.random.split(key)
        return self.SolveConfig(TargetState=self._get_random_state(maze, key1), Maze=maze)

    def get_neighbours(
        self, solve_config: "Maze.SolveConfig", state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return neighbours, and the cost of the move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Define possible moves: up, down, left, right
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        def move(state, move):
            new_pos = (state.pos + move).astype(TYPE)

            maze = self.from_uint8(solve_config.Maze)
            # Check if the new position is within the maze bounds and not a wall
            valid_move = (
                (new_pos >= 0).all()
                & (new_pos < self.size).all()
                & (maze[new_pos[0] * self.size + new_pos[1]] == 0)
                & filled
            )

            # If the move is valid, update the position. Otherwise, keep the old position.
            new_state = self.State(pos=jnp.where(valid_move, new_pos, state.pos))

            # Cost is 1 for valid moves, inf for invalid moves
            cost = jnp.where(valid_move, 1.0, jnp.inf)

            return new_state, cost

        # Apply the move function to all possible moves
        new_states, costs = jax.vmap(lambda m: move(state, m))(moves)

        return new_states, costs

    def is_solved(self, solve_config: "Maze.SolveConfig", state: State) -> bool:
        return self.is_equal(state, solve_config.TargetState)

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        if action == 0:
            return "↑"
        elif action == 1:
            return "↓"
        elif action == 2:
            return "←"
        elif action == 3:
            return "→"
        else:
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
            form += "┃"
            form += "\n"
        form += "┗━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, maze: chex.Array, key):
        """
        This function should return a random state.
        """
        maze = self.from_uint8(maze)

        def get_random_state(key):
            return self.State(pos=jax.random.randint(key, (2,), 0, self.size, dtype=TYPE))

        def is_not_wall(x):
            state = x[0]
            return maze[state.pos[0] * self.size + state.pos[1]] != 0

        def while_loop(x):
            state, key = x
            next_key, key = jax.random.split(key)
            state = get_random_state(key)
            return state, next_key

        next_key, key = jax.random.split(key)
        state = get_random_state(key)
        state, _ = jax.lax.while_loop(is_not_wall, while_loop, (state, next_key))
        return state

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

        def img_func(state: "Maze.State", solve_config: "Maze.SolveConfig" = None, **kwargs):
            assert solve_config is not None, "This puzzle requires a solve_config"
            imgsize = IMG_SIZE[0]
            # Create a white background image for the maze
            img = np.full((imgsize, imgsize, 3), 255, np.uint8)

            # Unpack the maze layout from state's board.
            # Assume that each cell is represented by a binary value:
            # 1 indicates a wall, 0 indicates an open path.
            maze_flat = self.from_uint8(solve_config.Maze)
            cell_size = imgsize / self.size

            # Draw the maze walls as filled black rectangles.
            for i in range(self.size):
                for j in range(self.size):
                    idx = i * self.size + j
                    if maze_flat[idx] == 1:
                        top_left = (int(j * cell_size), int(i * cell_size))
                        bottom_right = (int((j + 1) * cell_size), int((i + 1) * cell_size))
                        img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), thickness=-1)

            # Optionally, draw grid lines to highlight the maze cell boundaries.
            for i in range(self.size + 1):
                pt1 = (0, int(i * cell_size))
                pt2 = (imgsize, int(i * cell_size))
                cv2.line(img, pt1, pt2, (200, 200, 200), 1)
            for j in range(self.size + 1):
                pt1 = (int(j * cell_size), 0)
                pt2 = (int(j * cell_size), imgsize)
                cv2.line(img, pt1, pt2, (200, 200, 200), 1)

            # Draw the player's current position.
            pos_player = state.pos
            pos_target = solve_config.TargetState.pos
            player_center = (
                int((pos_player[1] + 0.5) * cell_size),
                int((pos_player[0] + 0.5) * cell_size),
            )
            player_radius = max(1, int(cell_size / 3))
            if (state.pos == solve_config.TargetState.pos).all():
                img = cv2.circle(img, player_center, player_radius, (0, 255, 0), thickness=-1)
            else:
                img = cv2.circle(img, player_center, player_radius, (255, 0, 0), thickness=-1)
                # Otherwise, draw the target as an "X".
                top_left = (int(pos_target[1] * cell_size), int(pos_target[0] * cell_size))
                bottom_right = (
                    int((pos_target[1] + 1) * cell_size),
                    int((pos_target[0] + 1) * cell_size),
                )
                top_right = (int((pos_target[1] + 1) * cell_size), int(pos_target[0] * cell_size))
                bottom_left = (int(pos_target[1] * cell_size), int((pos_target[0] + 1) * cell_size))
                img = cv2.line(img, top_left, bottom_right, (255, 0, 0), thickness=2)
                img = cv2.line(img, top_right, bottom_left, (255, 0, 0), thickness=2)

            return img

        return img_func

    def get_solve_config_img_parser(self):
        def parser(solve_config: "Maze.SolveConfig"):
            return self.get_img_parser()(solve_config.TargetState, solve_config)

        return parser
