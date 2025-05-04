import jax
import jax.numpy as jnp

from puzzle.maze import TYPE, Maze  # Inherit from Maze

# Removed Fixed Map Constants - Map is now generated dynamically


class Room(Maze):
    """A Maze subclass representing a fixed 3x3 grid of rooms.
    The internal dimension of each room is configurable via the total grid size.
    Size must be of the form 3*N+2, where N>=1 is the room dimension.
    If an invalid size is provided, it adjusts to the nearest valid size."""

    room_dim: int  # Internal dimension of each room

    def __init__(self, size: int = 11, **kwargs):
        """Initialize with a specified size, calculating room dimension and
        adjusting to the nearest valid size if necessary."""
        if size < 5:
            raise ValueError(
                f"Input size {size} is too small. Minimum valid size is 5 (for 1x1 rooms)."
            )

        # Check if size fits the 3*N+2 formula
        if (size - 2) % 3 == 0:
            actual_size = size
            room_dim = (size - 2) // 3
        else:
            # Calculate nearest room_dim (must be >= 1)
            target_room_dim_float = (size - 2) / 3
            room_dim = max(1, round(target_room_dim_float))
            actual_size = 3 * room_dim + 2
            print(
                f"[Room Puzzle] Input size {size} is invalid."
                f"Using closest valid size {actual_size} (room dimension {room_dim})."
            )

        self.room_dim = room_dim
        # Pass the final valid size to the Maze constructor
        super().__init__(size=actual_size, **kwargs)

    # --- Map Generation --- #

    def _generate_room_map(self) -> jnp.ndarray:
        """Generates the 3x3 room structure map based on self.size and self.room_dim."""
        size = self.size
        room_dim = self.room_dim
        num_rooms_dim = 3  # Fixed at 3x3 rooms
        maze = jnp.ones((size, size), dtype=jnp.bool_)  # Start with all walls

        for r in range(num_rooms_dim):
            for c in range(num_rooms_dim):
                # Top-left corner of the room area
                # Accounts for room dim and the 1-unit wall/corridor
                room_r = (room_dim + 1) * r
                room_c = (room_dim + 1) * c
                # Carve out the room_dim x room_dim room
                maze = maze.at[room_r : room_r + room_dim, room_c : room_c + room_dim].set(False)

                # Add connection (door) to the room on the right
                if c < num_rooms_dim - 1:
                    # Door row is middle of room height, door col is the wall coord
                    door_r = room_r + room_dim // 2
                    door_c = room_c + room_dim
                    maze = maze.at[door_r, door_c].set(False)

                # Add connection (door) to the room below
                if r < num_rooms_dim - 1:
                    # Door row is the wall coord, door col is middle of room width
                    door_r = room_r + room_dim
                    door_c = room_c + room_dim // 2
                    maze = maze.at[door_r, door_c].set(False)

        return maze

    def get_solve_config(self, key=jax.random.PRNGKey(128), data=None) -> Maze.SolveConfig:
        """Generates a SolveConfig using the generated 3x3 room map."""
        bool_maze_jax = self._generate_room_map()
        target_state = self._get_random_state(bool_maze_jax, key)
        packed_maze = self.to_uint8(bool_maze_jax.flatten())
        return self.SolveConfig(TargetState=target_state, Maze=packed_maze)

    def _generate_maze_dfs(self, key):
        """Overrides DFS generation to return the generated 3x3 room map."""
        return self._generate_room_map()

    def get_solve_config_default_gen(self):
        """Default SolveConfig with target at (0,0) and the generated 3x3 room map."""

        def gen():
            default_map = self._generate_room_map()
            packed_map = self.to_uint8(default_map.flatten())
            return self.SolveConfig(
                TargetState=self.State(pos=jnp.array([0, 0], dtype=TYPE)), Maze=packed_map
            )

        return gen
