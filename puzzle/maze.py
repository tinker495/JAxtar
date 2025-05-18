import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puzzle.util import from_uint8, to_uint8

TYPE = jnp.uint16


class Maze(Puzzle):

    size: int

    def define_solve_config_class(self) -> PuzzleState:
        dummy_maze = jnp.zeros((self.size**2), dtype=jnp.bool_)
        dummy_maze = to_uint8(dummy_maze)
        size = self.size

        @state_dataclass
        class SolveConfig:
            TargetState: FieldDescriptor[self.State]
            Maze: FieldDescriptor[jnp.uint8, (dummy_maze.shape[0],), dummy_maze]

            def __str__(self, **kwargs):
                return self.TargetState.str(solve_config=self, **kwargs)

            def packing(self):
                packed_maze = to_uint8(self.Maze)
                return SolveConfig(TargetState=self.TargetState, Maze=packed_maze)

            def unpacking(self):
                maze = from_uint8(self.Maze, (size * size,))
                return SolveConfig(TargetState=self.TargetState, Maze=maze)

        return SolveConfig

    def define_state_class(self) -> PuzzleState:

        str_parser = self.get_string_parser()

        @state_dataclass
        class State:
            pos: FieldDescriptor[TYPE, (2,)]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int, **kwargs):
        # Parameter p is no longer used for maze generation
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

        def parser(state: "Maze.State", solve_config: "Maze.SolveConfig", **kwargs):
            assert solve_config is not None, "This puzzle requires a solve_config"

            # 1. Unpack the maze to boolean (True=wall, False=path)
            bool_maze_flat = solve_config.unpacking().Maze

            # 2. Create an integer representation (0=path, 1=wall)
            # Ensure correct shape for intermediate calculations
            int_maze_flat = jnp.where(bool_maze_flat, 1, 0).astype(jnp.int8)

            # 3. Get target and player positions and calculate flat indices
            target_pos = solve_config.TargetState.pos
            player_pos = state.pos

            if self.size > 30:
                return (
                    f"Is too big to visualize - player at {player_pos} and target at {target_pos}"
                )

            target_idx = target_pos[0] * self.size + target_pos[1]
            player_idx = player_pos[0] * self.size + player_pos[1]

            # 4. Place target marker (3) onto the integer maze
            # Important: only place target if it's not a wall (should always be true with DFS gen)
            int_maze_flat = jnp.where(
                bool_maze_flat[target_idx], int_maze_flat, int_maze_flat.at[target_idx].set(3)
            )

            # 5. Check if player is on target
            is_on_target = target_idx == player_idx

            # 6. Place player marker (4 if on target, 2 otherwise)
            # Important: only place player if it's not a wall (should always be true)
            player_marker = jnp.where(is_on_target, 4, 2)
            int_maze_flat = jnp.where(
                bool_maze_flat[player_idx],
                int_maze_flat,
                int_maze_flat.at[player_idx].set(player_marker),
            )

            # 7. Format the string using the final integer maze
            return form.format(*map(to_char, int_maze_flat))

        return parser

    def get_initial_state(
        self, solve_config: "Maze.SolveConfig", key=jax.random.PRNGKey(0), data=None
    ) -> "Maze.State":
        # Start state should also be chosen from valid path locations
        bool_maze = solve_config.unpacking().Maze.reshape((self.size, self.size))
        return self._get_random_state(bool_maze, key)

    def get_solve_config(self, key=jax.random.PRNGKey(128), data=None) -> Puzzle.SolveConfig:
        # Generate maze using DFS
        key, maze_key, target_key = jax.random.split(key, 3)
        bool_maze = self._generate_maze_dfs(maze_key, self.size)  # Returns bool array (True=wall)
        bool_maze = bool_maze.ravel()

        # Get target state on a valid path cell
        target_state = self._get_random_state(bool_maze, target_key)

        return self.SolveConfig(TargetState=target_state, Maze=bool_maze).packing()

    def _generate_maze_dfs(self, key, size):
        """Generates a maze using Randomized Depth-First Search."""
        maze = jnp.ones((size, size), dtype=jnp.bool_)  # Start with all walls (True)
        stack = jnp.zeros((size * size, 2), dtype=TYPE)  # Max possible stack depth
        stack_ptr = 0

        # Choose starting cell - always start at (0, 0)
        # key, start_key = jax.random.split(key) # No longer needed for random start
        start_pos = jnp.array([0, 0], dtype=TYPE)
        maze = maze.at[start_pos[0], start_pos[1]].set(False)  # Mark start (0,0) as path
        stack = stack.at[stack_ptr].set(start_pos)
        stack_ptr += 1

        # Directions: N, S, E, W (relative row, col changes)
        # We check cells 2 steps away to ensure walls remain between paths
        dr = jnp.array([-2, 2, 0, 0], dtype=jnp.int8)
        dc = jnp.array([0, 0, 2, -2], dtype=jnp.int8)
        # Wall between cells
        wall_dr = jnp.array([-1, 1, 0, 0], dtype=jnp.int8)
        wall_dc = jnp.array([0, 0, 1, -1], dtype=jnp.int8)

        def _cond_fun(state):
            # Continue while stack is not empty
            _, _, stack_ptr, _ = state
            return stack_ptr > 0

        def _body_fun(state):
            maze, stack, stack_ptr, key = state
            key, shuffle_key, loop_key = jax.random.split(key, 3)

            # Current position (top of stack)
            curr_pos = stack[stack_ptr - 1]
            cr, cc = curr_pos[0], curr_pos[1]

            # Find unvisited neighbours (cells that are walls 2 steps away)
            potential_nr = cr + dr
            potential_nc = cc + dc

            # Check bounds
            in_bounds = (
                (potential_nr >= 0)
                & (potential_nr < size)
                & (potential_nc >= 0)
                & (potential_nc < size)
            )

            # Check if potential neighbour is a wall (i.e., unvisited)
            # Need to handle OOB indexing safely for maze lookup
            safe_nr = jnp.clip(potential_nr, 0, size - 1)
            safe_nc = jnp.clip(potential_nc, 0, size - 1)
            is_wall = maze[safe_nr, safe_nc]

            valid_neighbors_mask = in_bounds & is_wall
            valid_indices = jnp.where(valid_neighbors_mask, size=4, fill_value=-1)[
                0
            ]  # Get indices [0,1,2,3] of valid moves
            num_valid_neighbors = jnp.sum(valid_neighbors_mask)

            # --- Jax control flow: choose a branch ---
            def _visit_neighbor(state):
                maze, stack, stack_ptr, key, valid_indices, num_valid_neighbors = state
                key, choice_key = jax.random.split(key)

                # Randomly choose one valid neighbor
                chosen_idx_in_valid = jax.random.randint(
                    choice_key, (), 0, num_valid_neighbors, dtype=jnp.int32
                )
                chosen_dir_idx = valid_indices[
                    chosen_idx_in_valid
                ]  # Map back to original direction index [0,1,2,3]

                nr, nc = potential_nr[chosen_dir_idx], potential_nc[chosen_dir_idx]
                wall_r, wall_c = cr + wall_dr[chosen_dir_idx], cc + wall_dc[chosen_dir_idx]

                # Carve path to neighbor and wall between
                maze = maze.at[nr, nc].set(False)
                maze = maze.at[wall_r, wall_c].set(False)

                # Push neighbor onto stack
                new_pos = jnp.array([nr, nc], dtype=TYPE)
                stack = stack.at[stack_ptr].set(new_pos)
                stack_ptr += 1
                return maze, stack, stack_ptr, key

            def _backtrack(state):
                maze, stack, stack_ptr, key, _, _ = state
                # Pop from stack
                stack_ptr -= 1
                return maze, stack, stack_ptr, key

            # Use jax.lax.cond to either visit a neighbor or backtrack
            maze, stack, stack_ptr, key = jax.lax.cond(
                num_valid_neighbors > 0,
                _visit_neighbor,
                _backtrack,
                (
                    maze,
                    stack,
                    stack_ptr,
                    loop_key,
                    valid_indices,
                    num_valid_neighbors,
                ),  # Pass necessary state
            )
            return maze, stack, stack_ptr, key

        # Initial state for the loop
        init_state = (maze, stack, stack_ptr, key)
        # Run the DFS loop
        maze, _, _, _ = jax.lax.while_loop(_cond_fun, _body_fun, init_state)

        return maze  # Return the boolean maze grid

    def get_neighbours(
        self, solve_config: "Maze.SolveConfig", state: "Maze.State", filled: bool = True
    ) -> tuple["Maze.State", chex.Array]:
        """
        This function should return neighbours, and the cost of the move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Define possible moves: up, down, left, right
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        bool_maze = solve_config.unpacking().Maze.reshape((self.size, self.size))

        def move(state, move):
            new_pos = (state.pos + move).astype(TYPE)

            # Check if the new position is within the maze bounds and not a wall (True)
            valid_move = (
                (new_pos >= 0).all()
                & (new_pos < self.size).all()
                & (~bool_maze[new_pos[0], new_pos[1]])  # Check against False (path)
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

    def is_solved(self, solve_config: "Maze.SolveConfig", state: "Maze.State") -> bool:
        return state == solve_config.TargetState

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
            form += "┃"
            form += "\n"
        form += "┗━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, bool_maze: chex.Array, key):
        """
        This function should return a random state on a path cell (False).
        Accepts a boolean maze directly.
        """
        # bool_maze is now passed directly
        # Ensure bool_maze is 2D
        if bool_maze.ndim == 1:
            bool_maze = bool_maze.reshape((self.size, self.size))

        def get_random_pos(key):
            return jax.random.randint(key, (2,), 0, self.size, dtype=TYPE)

        def is_wall(carry):
            pos, _ = carry
            # Check if the position is a wall (True)
            return bool_maze[pos[0], pos[1]]

        def while_body(carry):
            _, key = carry
            key, split_key = jax.random.split(key)
            new_pos = get_random_pos(split_key)
            return new_pos, key

        # Initial random position
        key, pos_key, loop_key = jax.random.split(key, 3)
        initial_pos = get_random_pos(pos_key)
        initial_carry = (initial_pos, loop_key)

        # Loop until we find a position that is not a wall
        final_pos, _ = jax.lax.while_loop(is_wall, while_body, initial_carry)

        return self.State(pos=final_pos)

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        def img_func(state: "Maze.State", solve_config: "Maze.SolveConfig" = None, **kwargs):
            assert solve_config is not None, "This puzzle requires a solve_config"
            imgsize = IMG_SIZE[0]

            # --- Optimized Wall Rendering ---
            # 1. Unpack maze to boolean (True=wall)
            maze_bool_jax = solve_config.unpacking().Maze.reshape((self.size, self.size))
            maze_bool_np = np.array(maze_bool_jax)  # Convert JAX array to NumPy array

            # 2. Create monochrome image (0=wall, 255=path) using NumPy array
            walls_mono_np = (~maze_bool_np).astype(np.uint8) * 255

            # 3. Resize the NumPy array to target image size
            img_resized = cv2.resize(
                walls_mono_np, (imgsize, imgsize), interpolation=cv2.INTER_NEAREST
            )

            # 4. Convert to 3-channel BGR
            img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            # --- End Optimized Wall Rendering ---

            cell_size = imgsize / self.size  # Still needed for grid lines and object placement

            # Draw grid lines (remains the same)
            grid_color = (200, 200, 200)  # Light grey
            for i in range(self.size + 1):
                pt1 = (0, int(i * cell_size))
                pt2 = (imgsize, int(i * cell_size))
                cv2.line(img, pt1, pt2, grid_color, 1)
            for j in range(self.size + 1):
                pt1 = (int(j * cell_size), 0)
                pt2 = (int(j * cell_size), imgsize)
                cv2.line(img, pt1, pt2, grid_color, 1)

            # Draw player and target (remains the same)
            pos_player = state.pos
            pos_target = solve_config.TargetState.pos
            player_center = (
                int((pos_player[1] + 0.5) * cell_size),
                int((pos_player[0] + 0.5) * cell_size),
            )
            player_radius = max(1, int(cell_size / 3))

            if (state.pos == solve_config.TargetState.pos).all():
                # Player on target: Green circle
                img = cv2.circle(img, player_center, player_radius, (0, 255, 0), thickness=-1)
            else:
                # Player not on target: Red circle
                img = cv2.circle(img, player_center, player_radius, (255, 0, 0), thickness=-1)

                # Draw target 'X' (Red)
                target_top_left = (int(pos_target[1] * cell_size), int(pos_target[0] * cell_size))
                target_bottom_right = (
                    int((pos_target[1] + 1) * cell_size),
                    int((pos_target[0] + 1) * cell_size),
                )
                target_top_right = (
                    int((pos_target[1] + 1) * cell_size),
                    int(pos_target[0] * cell_size),
                )
                target_bottom_left = (
                    int(pos_target[1] * cell_size),
                    int((pos_target[0] + 1) * cell_size),
                )

                target_color = (255, 0, 0)  # Red in BGR
                thickness = 2
                img = cv2.line(img, target_top_left, target_bottom_right, target_color, thickness)
                img = cv2.line(img, target_top_right, target_bottom_left, target_color, thickness)

            return img

        return img_func

    def get_solve_config_img_parser(self):
        def parser(solve_config: "Maze.SolveConfig"):
            return self.get_img_parser()(solve_config.TargetState, solve_config)

        return parser
