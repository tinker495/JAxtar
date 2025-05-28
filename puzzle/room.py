import jax
import jax.numpy as jnp

from puzzle.maze import Maze  # Inherit from Maze

# Removed Fixed Map Constants - Map is now generated dynamically


# JAX-compatible Disjoint Set Union (DSU) find operation
def dsu_find_jax(parent_array: jnp.ndarray, i: int) -> int:
    """
    Finds the representative (root) of the set containing element i.
    Uses a JAX-compatible iterative approach (lax.while_loop).
    No path compression during this specific find pass for simplicity in JAX tracing.
    """
    # parent_array: shape (N,)
    # i: scalar index
    # Returns scalar root_idx

    def _find_cond(curr_idx_val):
        # Condition to continue loop: current element is not its own parent
        return parent_array[curr_idx_val] != curr_idx_val

    def _find_body(curr_idx_val):
        # Move to the parent of the current element
        return parent_array[curr_idx_val]

    # Loop until the root (element that is its own parent) is found
    root_idx = jax.lax.while_loop(_find_cond, _find_body, i)
    return root_idx


class Room(Maze):
    """A Maze subclass representing a fixed 3x3 grid of rooms.
    The internal dimension of each room is configurable via the total grid size.
    Size must be of the form 3*N+2, where N>=1 is the room dimension.
    If an invalid size is provided, it adjusts to the nearest valid size.
    Doors between rooms are now randomly opened/closed while ensuring solvability.
    """

    room_dim: int  # Internal dimension of each room
    # Probability to open an additional door beyond those needed for basic connectivity
    _prob_open_extra_door: float = 0.1

    def __init__(self, size: int = 11, prob_open_extra_door: float = 1.0, **kwargs):
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
        self._prob_open_extra_door = prob_open_extra_door
        # Pass the final valid size to the Maze constructor
        super().__init__(size=actual_size, **kwargs)

    # --- Map Generation --- #

    def _generate_maze_dfs(self, key: jax.random.PRNGKey, size_param: int) -> jnp.ndarray:
        """
        Generates the 3x3 room structure map with randomly opened/closed doors.
        Ensures all rooms are connected using a Kruskal-like algorithm on the room graph.
        Overrides the DFS generation from the parent Maze class.
        `size_param` is `self.size`, passed from superclass call.
        """
        grid_size = self.size  # Actual grid dimensions
        room_dim = self.room_dim
        num_rooms_dim = 3  # Fixed at 3x3 rooms

        # Start with all walls
        maze = jnp.ones((grid_size, grid_size), dtype=jnp.bool_)

        # 1. Carve out the room_dim x room_dim room interiors
        for r_idx in range(num_rooms_dim):
            for c_idx in range(num_rooms_dim):
                room_r_start = (room_dim + 1) * r_idx
                room_c_start = (room_dim + 1) * c_idx
                maze = maze.at[
                    room_r_start : room_r_start + room_dim, room_c_start : room_c_start + room_dim
                ].set(False)

        # 2. Define all potential doors and their properties
        # Each entry: ((door_r, door_c), room_idx1_flat, room_idx2_flat)
        _potential_doors_list_py = []

        def room_to_flat_idx(r, c):
            return r * num_rooms_dim + c

        # Horizontal doors (connecting rooms in the same row, e.g., (0,0) to (0,1))
        for r_idx in range(num_rooms_dim):
            for c_idx in range(num_rooms_dim - 1):
                room_r_start = (room_dim + 1) * r_idx
                door_r_coord = room_r_start + room_dim // 2
                door_c_coord = (room_dim + 1) * c_idx + room_dim  # Wall coordinate

                idx1 = room_to_flat_idx(r_idx, c_idx)
                idx2 = room_to_flat_idx(r_idx, c_idx + 1)
                _potential_doors_list_py.append(((door_r_coord, door_c_coord), idx1, idx2))

        # Vertical doors (connecting rooms in the same col, e.g., (0,0) to (1,0))
        for c_idx in range(num_rooms_dim):
            for r_idx in range(num_rooms_dim - 1):
                room_c_start = (room_dim + 1) * c_idx
                door_r_coord = (room_dim + 1) * r_idx + room_dim  # Wall coordinate
                door_c_coord = room_c_start + room_dim // 2

                idx1 = room_to_flat_idx(r_idx, c_idx)
                idx2 = room_to_flat_idx(r_idx + 1, c_idx)
                _potential_doors_list_py.append(((door_r_coord, door_c_coord), idx1, idx2))

        num_potential_doors = len(_potential_doors_list_py)
        # Convert Python list of door data to JAX arrays
        door_maze_coords_jax = jnp.array([d[0] for d in _potential_doors_list_py], dtype=jnp.int32)
        door_room_pairs_jax = jnp.array([d[1:] for d in _potential_doors_list_py], dtype=jnp.int32)

        # 3. Ensure all rooms are connected using a Kruskal-like algorithm (DSU)
        key_shuffle, key_extra_doors = jax.random.split(key)

        # Shuffle the order of considering potential doors
        shuffled_door_indices = jax.random.permutation(key_shuffle, jnp.arange(num_potential_doors))

        # Initialize DSU state for rooms
        # parent_array[i] = parent of room i. Initially each room is its own parent.
        initial_parent_array = jnp.arange(num_rooms_dim * num_rooms_dim)
        # Mask to track which doors are opened to form the spanning tree
        initial_st_doors_mask = jnp.zeros(num_potential_doors, dtype=jnp.bool_)
        # Number of edges added to the spanning tree
        initial_edges_count = 0

        # State for scan: (parent_array, st_doors_mask, edges_added_count)
        initial_kruskal_carry = (initial_parent_array, initial_st_doors_mask, initial_edges_count)

        # Kruskal's algorithm: iterate through shuffled doors, add if connects different components
        def kruskal_scan_body(carry_state, current_shuffled_door_idx):
            parent_arr, st_mask, edges_added = carry_state

            # Get the two rooms this door connects
            room1_idx = door_room_pairs_jax[current_shuffled_door_idx, 0]
            room2_idx = door_room_pairs_jax[current_shuffled_door_idx, 1]

            # Find representatives (roots) of the sets these rooms belong to
            root1 = dsu_find_jax(parent_arr, room1_idx)
            root2 = dsu_find_jax(parent_arr, room2_idx)

            # If roots are different and we still need edges for spanning tree, unite them
            def _perform_union_and_add_door(op_state):
                p_arr, current_mask, e_count = op_state
                # Union: make root1's parent root2 (or vice-versa)
                p_arr_updated = p_arr.at[root1].set(root2)
                # Mark this door as part of the spanning tree
                mask_updated = current_mask.at[current_shuffled_door_idx].set(True)
                e_count_updated = e_count + 1
                return p_arr_updated, mask_updated, e_count_updated

            def _do_nothing(op_state):
                return op_state  # No change

            # Condition for union: roots differ AND spanning tree is not yet complete
            # Spanning tree for N rooms needs N-1 edges. Here, 9 rooms need 8 edges.
            max_st_edges = (num_rooms_dim * num_rooms_dim) - 1

            new_parent_arr, new_st_mask, new_edges_added = jax.lax.cond(
                (root1 != root2) & (edges_added < max_st_edges),
                _perform_union_and_add_door,
                _do_nothing,
                (parent_arr, st_mask, edges_added),
            )
            return (
                new_parent_arr,
                new_st_mask,
                new_edges_added,
            ), None  # No per-iteration output needed

        # Run the scan to determine which doors form the spanning tree
        final_kruskal_state, _ = jax.lax.scan(
            kruskal_scan_body, initial_kruskal_carry, shuffled_door_indices
        )
        _, spanning_tree_doors_mask, _ = final_kruskal_state

        # Open the doors identified for the spanning tree
        maze_after_st = maze  # Start with maze where only rooms are carved

        def open_st_doors_loop_body(i, current_maze_state):
            # If this door is in the spanning tree mask, open it
            door_coord_tuple = (door_maze_coords_jax[i, 0], door_maze_coords_jax[i, 1])
            return jax.lax.cond(
                spanning_tree_doors_mask[i],
                lambda m: m.at[door_coord_tuple].set(False),  # Open the door
                lambda m: m,  # Keep as is
                current_maze_state,
            )

        maze_after_st = jax.lax.fori_loop(
            0, num_potential_doors, open_st_doors_loop_body, maze_after_st
        )

        # 4. Randomly open additional doors (those not in the spanning tree)
        # Generate random numbers for each potential door
        extra_door_rand_probs = jax.random.uniform(key_extra_doors, (num_potential_doors,))

        final_maze = maze_after_st

        def open_extra_doors_loop_body(i, current_maze_state):
            is_spanning_tree_door = spanning_tree_doors_mask[i]
            # Decide to open if it's NOT an ST door AND random chance passes
            should_open_randomly = extra_door_rand_probs[i] < self._prob_open_extra_door

            door_coord_tuple = (door_maze_coords_jax[i, 0], door_maze_coords_jax[i, 1])

            return jax.lax.cond(
                (~is_spanning_tree_door) & should_open_randomly,
                lambda m: m.at[door_coord_tuple].set(False),  # Open the door
                lambda m: m,  # Keep as is
                current_maze_state,
            )

        final_maze = jax.lax.fori_loop(
            0, num_potential_doors, open_extra_doors_loop_body, final_maze
        )

        return final_maze

    # Removed _generate_room_map method as its logic is now integrated and enhanced
    # in _generate_maze_dfs above.
