import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

TYPE = jnp.uint8


def get_color(size):
    """Get color based on pancake size"""
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
    return colors[int(size) % len(colors)]


class PancakeSorting(Puzzle):
    """
    Pancake Sorting Puzzle

    In this puzzle, there's a stack of pancakes with different sizes.
    The goal is to sort the pancakes by size with the largest at the bottom.
    The only operation allowed is to insert a spatula at any position and flip all pancakes above it.
    """

    size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for PancakeSorting using xtructure."""
        str_parser = self.get_string_parser()

        @state_dataclass
        class State:
            stack: FieldDescriptor[TYPE, (self.size,)]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int, **kwargs):
        """
        Initialize the Pancake Sorting puzzle

        Args:
            size: The number of pancakes in the stack
        """
        self.size = size
        super().__init__(**kwargs)

    def get_string_parser(self):
        """Returns a function to convert a state to a string representation"""

        def parser(state: "PancakeSorting.State", **kwargs):
            result = []
            for i, pancake in enumerate(state.stack):
                size_str = "=" * (2 * (int(pancake) - 1) + 1)
                result.append(
                    f"{i+1:02d}:{pancake:02d} - "
                    + colored(f"{size_str.center(self.size * 2)}", get_color(pancake))
                )
            result.append("Plate " + "┗━" + "━━" * self.size + "┛")
            return "\n".join(result)

        return parser

    def get_img_parser(self) -> callable:
        """Returns a function to convert a state to an image representation"""
        import cv2
        import numpy as np

        def img_func(state: "PancakeSorting.State", **kwargs):
            # Create blank image with correct dimensions
            # IMG_SIZE is actually a tuple (width, height)
            image = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
            image.fill(240)  # Light gray background
            stack = state.stack
            max_size = self.size

            # Calculate parameters for visualization
            img_height = IMG_SIZE[1]  # Height from the tuple
            pancake_height = img_height // (self.size + 4)
            max_width = IMG_SIZE[0] - 40  # Width from the tuple

            # Draw a plate at the bottom - moved to the bottom of the image
            plate_y = img_height - 50  # Position plate at the bottom with some margin
            plate_height = pancake_height // 2
            plate_width = int(max_width * 1.1)
            cv2.ellipse(
                image,
                (IMG_SIZE[0] // 2, plate_y + plate_height // 2),
                (plate_width // 2, plate_height // 2),
                0,
                0,
                180,
                (150, 150, 150),
                -1,
            )

            def draw_pancake(img, y_pos, size):
                width = int(max_width * (size / max_size))
                x_start = (IMG_SIZE[0] - width) // 2
                x_end = x_start + width

                # Generate color based on pancake size using a gradient
                # Map the size to a position in the gradient (0 to 1)
                gradient_pos = (size - 1) / max_size

                # Create a smooth gradient from light orange to dark brown
                r = int(255 - (95 * gradient_pos))  # 255 -> 160
                g = int(200 - (100 * gradient_pos))  # 200 -> 100
                b = int(100 - (100 * gradient_pos))  # 100 -> 0

                # Ensure values are within valid range
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                color = (r, g, b)

                # Draw pancake with rounded corners and gradient
                # Create a rounded rectangle for the pancake
                rect_points = np.array(
                    [
                        [x_start, y_pos],
                        [x_end, y_pos],
                        [x_end, y_pos + pancake_height],
                        [x_start, y_pos + pancake_height],
                    ]
                )

                # Draw filled pancake with rounded corners
                cv2.fillPoly(img, [rect_points], color)

                # Add a highlight on top of the pancake
                highlight_y = y_pos + 2
                highlight_height = pancake_height // 4
                cv2.rectangle(
                    img,
                    (x_start + 5, highlight_y),
                    (x_end - 5, highlight_y + highlight_height),
                    (min(color[0] + 40, 255), min(color[1] + 40, 255), min(color[2] + 40, 255)),
                    -1,
                )

                # Add a shadow at the bottom
                shadow_y = y_pos + pancake_height - highlight_height - 2
                cv2.rectangle(
                    img,
                    (x_start + 5, shadow_y),
                    (x_end - 5, shadow_y + highlight_height),
                    (max(color[0] - 40, 0), max(color[1] - 40, 0), max(color[2] - 40, 0)),
                    -1,
                )

                # Add size text in the middle of the pancake
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(int(size))
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                text_x = (x_start + x_end - text_size[0]) // 2
                text_y = y_pos + (pancake_height + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)

                return img

            # Calculate starting position for the stack (from bottom to top)
            base_y_pos = plate_y - int(pancake_height * 0.75)

            # Draw each pancake from bottom to top
            for i, size in enumerate(reversed(stack)):
                y_pos = base_y_pos - (i * pancake_height)
                image = draw_pancake(image, y_pos, size)

            # Convert to JAX array and return
            return image

        return img_func

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "PancakeSorting.State":
        """Generate a random initial state for the puzzle"""
        return self._get_random_state(key)

    def get_solve_config(self, key=None, data=None) -> "PancakeSorting.SolveConfig":
        """Create the solving configuration (target state)"""
        # Target is the sorted order, largest at the bottom (index size-1)
        target_stack = jnp.arange(1, self.size + 1, dtype=TYPE)
        return self.SolveConfig(TargetState=self.State(stack=target_stack))

    def get_neighbours(
        self,
        solve_config: "PancakeSorting.SolveConfig",
        state: "PancakeSorting.State",
        filled: bool = True,
    ) -> tuple["PancakeSorting.State", chex.Array]:
        """
        Get all neighboring states by flipping pancakes at different positions

        Returns:
            tuple: (neighboring states, costs of moves)
        """
        stack = state.stack

        # Generate all possible flip positions (we can flip at positions 0 to size-2)
        # Position i means flip pancakes from index 0 to index i
        possible_flips = (
            jnp.arange(self.size - 1) + 1
        )  # +1 because we flip at positions 1 to size-1

        def flip_stack(stack, flip_pos):
            """Flip the pancakes from index 0 to flip_pos (inclusive)"""
            # For each valid flip position, we need to create a new stack with the top portion flipped
            indices = jnp.arange(stack.shape[0])

            # Create masks for the section to flip vs keep unchanged
            flip_section_mask = indices <= flip_pos

            # Create a new array with same shape as stack
            new_stack = jnp.zeros_like(stack)

            # For the flip section (0 to flip_pos), we need to copy elements in reverse order
            # For each position i in the flip section, we want stack[flip_pos - i]
            def body_fun(i, new_stack):
                new_pos = jnp.where(
                    flip_section_mask[i],
                    flip_pos - i,  # Reverse the order within the flip section
                    i,  # Keep the same position for the rest
                )
                return new_stack.at[i].set(stack[new_pos])

            new_stack = jax.lax.scan(
                lambda new_s, i: (body_fun(i, new_s), None), new_stack, jnp.arange(stack.shape[0])
            )[0]

            return new_stack

        def map_fn(flip_pos, filled):
            next_stack, cost = jax.lax.cond(
                filled,
                lambda _: (flip_stack(stack, flip_pos), 1.0),
                lambda _: (stack, jnp.inf),
                None,
            )
            return next_stack, cost

        next_stacks, costs = jax.vmap(map_fn, in_axes=(0, None))(possible_flips, filled)
        return self.State(stack=next_stacks), costs

    def is_solved(
        self, solve_config: "PancakeSorting.SolveConfig", state: "PancakeSorting.State"
    ) -> bool:
        """Check if the current state matches the target state (sorted)"""
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        """Return a string representation of the action"""
        return f"Flip at position {action + 1}"

    def _get_random_state(self, key):
        """Generate a random initial state"""
        if key is None:
            key = jax.random.PRNGKey(0)

        # Create a shuffled arrangement of pancakes
        stack = jax.random.permutation(key, jnp.arange(1, self.size + 1, dtype=TYPE))
        return self.State(stack=stack)
