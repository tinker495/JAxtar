import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

TYPE = jnp.uint8


class TowerOfHanoi(Puzzle):
    """
    Tower of Hanoi Puzzle

    In this puzzle, there are three pegs and a number of disks of different sizes.
    The disks are initially stacked on the first peg in order of decreasing size (largest at bottom).
    The goal is to move all disks to the third peg, following these rules:
    1. Only one disk can be moved at a time
    2. Each move consists of taking the upper disk from one stack and placing it on top of another stack
    3. No disk may be placed on top of a smaller disk
    """

    num_disks: int
    num_pegs: int = 3  # Classic Tower of Hanoi has 3 pegs
    max_disk_value: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for Tower of Hanoi using xtructure."""
        str_parser = self.get_string_parser()
        # Default pegs value for FieldDescriptor, initialized when class is defined
        # self.num_pegs and self.num_disks are available from TowerOfHanoi.__init__
        default_pegs_val = jnp.zeros((self.num_pegs, self.num_disks + 1), dtype=TYPE)

        @state_dataclass
        class State:
            pegs: FieldDescriptor[TYPE, default_pegs_val.shape, default_pegs_val]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int, **kwargs):
        """
        Initialize the Tower of Hanoi puzzle

        Args:
            num_disks: The number of disks in the puzzle
        """
        self.num_disks = size
        self.max_disk_value = size
        super().__init__(**kwargs)

    def get_string_parser(self):
        """Returns a function to convert a state to a string representation"""

        def parser(state: "TowerOfHanoi.State", **kwargs):
            result = []

            # Get the pegs array - has shape (num_pegs, num_disks + 1)
            pegs = state.pegs

            # Find the maximum height
            max_height = self.num_disks

            # For each level from top to bottom
            for level in range(max_height):
                row = []

                # For each peg
                for peg_idx in range(self.num_pegs):
                    peg = pegs[peg_idx]
                    num_disks_on_peg = int(peg[0])

                    # Calculate position from the top
                    pos_from_top = level

                    # If there's a disk at this position
                    if pos_from_top < num_disks_on_peg:
                        # Get the disk at this position (index 1 + pos_from_top has the disk size)
                        disk_size = int(peg[1 + pos_from_top])
                        disk_str = "=" * (2 * disk_size - 1)
                        colored_disk = colored(
                            disk_str.center(2 * self.num_disks + 1), get_color(disk_size)
                        )
                        row.append(colored_disk)
                    else:
                        # No disk, just show the peg
                        row.append("|".center(2 * self.num_disks + 1))

                result.append("   ".join(row))

            # Add base
            base_row = []
            for _ in range(self.num_pegs):
                base = "-" * (2 * self.num_disks + 1)
                base_row.append(base)

            result.append("   ".join(base_row))

            # Add peg numbers
            label_row = []
            for i in range(self.num_pegs):
                label = f"Peg {i+1}".center(2 * self.num_disks + 1)
                label_row.append(label)

            result.append("   ".join(label_row))

            return "\n".join(result)

        return parser

    def get_img_parser(self) -> callable:
        """Returns a function to convert a state to an image representation"""
        import cv2
        import numpy as np

        def img_func(state: "TowerOfHanoi.State", **kwargs):
            # Create blank image with correct dimensions
            image = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
            image.fill(240)  # Light gray background

            # Get dimensions
            width, height = IMG_SIZE

            # Parameters for visualization
            peg_width = 10
            peg_height = height * 0.6
            base_height = 20
            base_width = width * 0.8

            # Bottom of pegs (y-coordinate)
            base_y = height - 80

            # Draw base
            base_x = (width - base_width) / 2
            cv2.rectangle(
                image,
                (int(base_x), int(base_y)),
                (int(base_x + base_width), int(base_y + base_height)),
                (120, 80, 40),  # Brown color
                -1,  # Filled
            )

            # Calculate peg positions
            peg_xs = [
                base_x + base_width * (i + 1) / (self.num_pegs + 1) for i in range(self.num_pegs)
            ]

            # Draw pegs
            for peg_x in peg_xs:
                cv2.rectangle(
                    image,
                    (int(peg_x - peg_width / 2), int(base_y - peg_height)),
                    (int(peg_x + peg_width / 2), int(base_y)),
                    (120, 80, 40),  # Brown color
                    -1,  # Filled
                )

            # Draw disks on pegs
            max_disk_width = base_width / (self.num_pegs + 1) * 0.9
            disk_height = 20

            # Get the pegs array
            pegs = state.pegs

            # For each peg
            for peg_idx, peg_x in enumerate(peg_xs):
                peg = pegs[peg_idx]
                num_disks_on_peg = int(peg[0])

                # For each disk on this peg (from bottom to top)
                for disk_idx in range(num_disks_on_peg):
                    disk_size = int(peg[1 + disk_idx])
                    disk_width = max_disk_width * disk_size / self.max_disk_value

                    # Position from bottom
                    pos_from_bottom = num_disks_on_peg - disk_idx - 1
                    disk_y = base_y - (pos_from_bottom + 1) * disk_height

                    # Generate color based on disk size
                    color = get_disk_color(disk_size, self.max_disk_value)

                    # Draw disk
                    cv2.rectangle(
                        image,
                        (int(peg_x - disk_width / 2), int(disk_y)),
                        (int(peg_x + disk_width / 2), int(disk_y + disk_height)),
                        color,
                        -1,  # Filled
                    )

                    # Add disk size text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = str(disk_size)
                    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
                    text_x = int(peg_x - text_size[0] / 2)
                    text_y = int(disk_y + disk_height - 5)
                    cv2.putText(image, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)

            return image

        return img_func

    def get_initial_state(
        self, solve_config: "TowerOfHanoi.SolveConfig", key=None, data=None
    ) -> "TowerOfHanoi.State":
        """Generate the initial state for the puzzle with all disks on the first peg"""
        # Create an array with all disks on the first peg
        pegs = jnp.zeros((self.num_pegs, self.num_disks + 1), dtype=TYPE)

        # Set the number of disks on the first peg
        pegs = pegs.at[0, 0].set(self.num_disks)

        # Place disks on the first peg in ascending order (smallest at top)
        # In this arrangement, index 1 = top disk, index num_disks = bottom disk
        # For example, with 3 disks:
        # pegs[0, 1] = 1 (smallest, at the top)
        # pegs[0, 2] = 2 (medium, in the middle)
        # pegs[0, 3] = 3 (largest, at the bottom)
        for i in range(self.num_disks):
            disk_size = i + 1  # Smallest disk size first (1), then increasing
            # Top disk at index 1, bottom disk at highest index
            pegs = pegs.at[0, i + 1].set(disk_size)

        return self.State(pegs=pegs)

    def get_solve_config(self, key=None, data=None) -> "TowerOfHanoi.SolveConfig":
        """Create the solving configuration (target state) - all disks on third peg"""
        # Create an array with all disks on the third peg
        pegs = jnp.zeros((self.num_pegs, self.num_disks + 1), dtype=TYPE)

        # Set the number of disks on the third peg
        pegs = pegs.at[2, 0].set(self.num_disks)

        # Place disks on the third peg in ascending order (smallest at top)
        # In this arrangement, index 1 = top disk, index num_disks = bottom disk
        # For example, with 3 disks:
        # pegs[2, 1] = 1 (smallest, at the top)
        # pegs[2, 2] = 2 (medium, in the middle)
        # pegs[2, 3] = 3 (largest, at the bottom)
        for i in range(self.num_disks):
            disk_size = i + 1  # Smallest disk size first (1), then increasing
            # Top disk at index 1, bottom disk at highest index
            pegs = pegs.at[2, i + 1].set(disk_size)

        return self.SolveConfig(TargetState=self.State(pegs=pegs))

    def get_neighbours(
        self,
        solve_config: "TowerOfHanoi.SolveConfig",
        state: "TowerOfHanoi.State",
        filled: bool = True,
    ) -> tuple["TowerOfHanoi.State", chex.Array]:
        """
        Get all neighboring states by moving a disk from one peg to another

        Returns:
            tuple: (neighboring states, costs of moves)
        """
        pegs = state.pegs

        # Generate all possible moves: (from_peg, to_peg)
        possible_moves = [
            (from_peg, to_peg)
            for from_peg in range(self.num_pegs)
            for to_peg in range(self.num_pegs)
            if from_peg != to_peg
        ]

        def is_valid_move(pegs, from_peg, to_peg):
            # Check if the from_peg has disks
            disks_on_from = pegs[from_peg, 0]
            valid_from = disks_on_from > 0

            # Get the top disk size from from_peg (if there are disks)
            # Top disk is at index 1 (smallest disk)
            from_top_disk = jax.lax.cond(
                disks_on_from > 0,
                lambda _: pegs[from_peg, 1],
                lambda _: jnp.array(0, dtype=TYPE),
                None,
            )

            # Check if the to_peg has space and the top disk on to_peg is larger
            disks_on_to = pegs[to_peg, 0]

            # If to_peg is empty, it's valid. Otherwise, compare disk sizes:
            # Only allow placing a smaller disk on top of a larger disk
            valid_to = jax.lax.cond(
                disks_on_to == 0,
                lambda _: jnp.array(True, dtype=bool),
                lambda _: from_top_disk < pegs[to_peg, 1],
                None,
            )

            return jnp.logical_and(valid_from, valid_to)

        def make_move(pegs, from_peg, to_peg):
            # Get the number of disks on the from_peg
            disks_on_from = pegs[from_peg, 0]

            # Get the top disk size from from_peg (smallest disk at top = index 1)
            from_top_disk = pegs[from_peg, 1]

            # Create a copy of the pegs array
            new_pegs = pegs.copy()

            # Remove the top disk from from_peg
            # Shift all disks up (disk at position n moves to position n-1)
            new_pegs = new_pegs.at[from_peg, 1:-1].set(new_pegs[from_peg, 2:])
            new_pegs = new_pegs.at[from_peg, -1].set(0)  # Clear the last position

            # Decrement the disk count on from_peg
            new_pegs = new_pegs.at[from_peg, 0].set(disks_on_from - 1)

            # Get the number of disks on the to_peg
            disks_on_to = new_pegs[to_peg, 0]

            # Add the disk to to_peg (at the top position = index 1)
            # Shift all disks down to make room at index 1
            new_pegs = new_pegs.at[to_peg, 2:].set(new_pegs[to_peg, 1:-1])
            new_pegs = new_pegs.at[to_peg, 1].set(from_top_disk)

            # Increment the disk count on to_peg
            new_pegs = new_pegs.at[to_peg, 0].set(disks_on_to + 1)

            return new_pegs

        # Function to apply to each possible move
        def map_fn(move, filled):
            from_peg, to_peg = move

            def move_disk(_):
                # Check if the move is valid
                valid = is_valid_move(pegs, from_peg, to_peg)

                # If valid, make the move; otherwise, keep the original pegs
                new_pegs = jax.lax.cond(
                    valid, lambda _: make_move(pegs, from_peg, to_peg), lambda _: pegs, None
                )

                # Cost is 1 if valid, infinity if invalid
                cost = jax.lax.cond(
                    valid, lambda _: jnp.array(1.0), lambda _: jnp.array(jnp.inf), None
                )

                return self.State(pegs=new_pegs), cost

            def no_move(_):
                return self.State(pegs=pegs), jnp.inf

            return jax.lax.cond(filled, move_disk, no_move, None)

        # Apply the mapping function to all possible moves
        # Convert moves to JAX array for vmap
        moves = jnp.array(possible_moves)
        next_states, costs = jax.vmap(map_fn, in_axes=(0, None))(moves, filled)

        return next_states, costs

    def is_solved(
        self, solve_config: "TowerOfHanoi.SolveConfig", state: "TowerOfHanoi.State"
    ) -> bool:
        """Check if the current state matches the target state"""
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        """Return a string representation of the action"""
        # action maps to (from_peg, to_peg) pair in possible_moves
        possible_moves = [
            (from_peg, to_peg)
            for from_peg in range(self.num_pegs)
            for to_peg in range(self.num_pegs)
            if from_peg != to_peg
        ]

        from_peg, to_peg = possible_moves[action]
        return f"Move disk from peg {from_peg+1} to peg {to_peg+1}"


def get_color(size):
    """Get color based on disk size"""
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
    return colors[(size - 1) % len(colors)]


def get_disk_color(size, max_size):
    """Get disk color as RGB based on size"""
    # Create a rainbow gradient
    hue = 240 * (1 - size / max_size)  # From blue (240) to red (0)

    # Convert HSV to RGB
    h = hue / 60
    i = int(h)
    f = h - i

    v = 0.9  # Value
    s = 0.8  # Saturation
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)
