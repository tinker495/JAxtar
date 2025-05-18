import chex
import jax
import jax.numpy as jnp

from puzzle.annotate import IMG_SIZE  # Assuming IMG_SIZE is defined
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

TYPE = jnp.uint8


class TopSpin(Puzzle):
    """
    Top Spin puzzle implementation.

    State: A permutation of numbers from 1 to n_discs.
    Actions:
        0: Shift the entire ring one step to the left.
        1: Shift the entire ring one step to the right.
        2: Reverse the order of the discs within the turnstile (first `turnstile_size` discs).
    Goal: Arrange the discs in ascending order [1, 2, ..., n_discs].
    """

    n_discs: int
    turnstile_size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for TopSpin using xtructure."""
        str_parser = self.get_string_parser()

        @state_dataclass
        class State:
            permutation: FieldDescriptor[TYPE, (self.n_discs,)]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 20, turnstile_size: int = 4, **kwargs):
        if turnstile_size > size:
            raise ValueError("Turnstile size cannot be larger than the number of discs.")
        self.n_discs = size
        self.turnstile_size = turnstile_size
        super().__init__(**kwargs)

    def get_string_parser(self) -> callable:
        def parser(state: "TopSpin.State", **kwargs):
            # Highlight the turnstile
            turnstile_str = " ".join(
                map(lambda x: f"{x:2d}", state.permutation[: self.turnstile_size])
            )
            rest_str = " ".join(map(lambda x: f"{x:2d}", state.permutation[self.turnstile_size :]))
            return f"[{turnstile_str}] {rest_str}"

        return parser

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        # The target state is the sorted permutation
        target_state = self.State(permutation=jnp.arange(1, self.n_discs + 1, dtype=TYPE))
        return self.SolveConfig(TargetState=target_state)

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "TopSpin.State":
        # Start from solved state and apply random moves
        return self._get_suffled_state(solve_config, solve_config.TargetState, key, 18)

    def _get_neighbors_internal(self, state: "TopSpin.State") -> tuple["TopSpin.State", chex.Array]:
        """Internal function to compute neighbors without vmap."""
        p = state.permutation

        # 1. Shift Left
        state_left = self.State(permutation=jnp.roll(p, -1))

        # 2. Shift Right
        state_right = self.State(permutation=jnp.roll(p, 1))

        # 3. Reverse Turnstile
        turnstile = p[: self.turnstile_size]
        reversed_turnstile = jnp.flip(turnstile)
        perm_reversed = p.at[: self.turnstile_size].set(reversed_turnstile)
        state_reversed = self.State(permutation=perm_reversed)

        # Combine states - use jax.tree_util.tree_map to stack arrays within the dataclass
        all_states = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args), state_left, state_right, state_reversed
        )

        costs = jnp.ones(3)  # All moves have cost 1

        return all_states, costs

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "TopSpin.State", filled: bool = True
    ) -> tuple["TopSpin.State", chex.Array]:
        """
        Returns neighbour states and costs for the 3 possible moves.
        If filled is False, costs are infinity.
        """
        all_states, costs = self._get_neighbors_internal(state)
        final_costs = jnp.where(filled, costs, jnp.inf)

        return all_states, final_costs

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "TopSpin.State") -> bool:
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        match action:
            case 0:
                return "Shift Left (<<)"
            case 1:
                return "Shift Right (>>)"
            case 2:
                return f"Reverse Turnstile (R{self.turnstile_size})"
            case _:
                raise ValueError(f"Invalid action: {action}")

    def get_img_parser(self):
        import cv2
        import numpy as np

        def img_func(state: "TopSpin.State", **kwargs):
            imgsize = IMG_SIZE[0]
            img = np.zeros(IMG_SIZE + (3,), np.uint8)
            img[:] = (240, 240, 240)  # White background

            n = self.n_discs
            ts = self.turnstile_size
            center_x, center_y = imgsize // 2, imgsize // 2
            radius = int(imgsize * 0.4)
            font_scale = 1.0
            font_thickness = 2
            disc_radius = int(imgsize * 0.04)

            # Find the position of the first turnstile element to align it at the top
            # This ensures the turnstile is always at the top (12 o'clock position)
            offset = -(
                self.turnstile_size // 2
            )  # No offset needed as we'll place the first ts elements at the top

            # Draw the ring and discs
            for i, val in enumerate(state.permutation):
                # Calculate angle to place turnstile at the top (12 o'clock position)
                # First ts elements will be in the turnstile area
                angle = (2 * np.pi * ((i + offset + 0.5) / n)) - (
                    np.pi / 2
                )  # Start from top (12 o'clock)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                # Determine if this position is part of the turnstile
                is_turnstile = i < ts
                color = (
                    (0, 0, 200) if is_turnstile else (50, 50, 50)
                )  # Blue for turnstile, gray otherwise
                cv2.circle(img, (x, y), disc_radius, color, -1)

                text = str(val)
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(
                    img,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )

            # Draw the turnstile area indicator at the top
            start_angle_rad = -np.pi / 2 - (np.pi * ts / n)  # Start angle for turnstile area
            end_angle_rad = -np.pi / 2 + (np.pi * ts / n)  # End angle for turnstile area
            cv2.ellipse(
                img,
                (center_x, center_y),
                (radius + disc_radius + 5, radius + disc_radius + 5),
                0,
                np.degrees(start_angle_rad),
                np.degrees(end_angle_rad),
                (200, 0, 0),
                2,
            )

            return img

        return img_func
