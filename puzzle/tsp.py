import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import ACTION_DTYPE  # ensure consistency with search utilities
from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle
from puzzle.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puzzle.util import from_uint8, to_uint8

# Use 16-bit unsigned integers so that problem sizes >255 are handled without overflow.
TYPE = jnp.uint16


class TSP(Puzzle):

    size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for TSP using xtructure."""
        str_parser = self.get_string_parser()
        mask = jnp.zeros(self.size, dtype=jnp.bool_)
        packed_mask = to_uint8(mask)
        size = self.size

        @state_dataclass
        class State:
            mask: FieldDescriptor[jnp.uint8, packed_mask.shape, packed_mask]
            point: FieldDescriptor[TYPE]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            def packing(self):
                packed_mask = to_uint8(self.mask)
                return State(mask=packed_mask, point=self.point)

            def unpacking(self):
                mask = from_uint8(self.mask, (size,))
                return State(mask=mask, point=self.point)

        return State

    def define_solve_config_class(self) -> PuzzleState:
        """Defines the solve config class for TSP using xtructure."""
        str_parser = self.get_solve_config_string_parser()

        @state_dataclass
        class SolveConfig:
            points: FieldDescriptor[jnp.float32, (self.size, 2)]
            distance_matrix: FieldDescriptor[jnp.float32, (self.size, self.size)]
            start: FieldDescriptor[TYPE]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return SolveConfig

    def __init__(self, size: int, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def get_solve_config_string_parser(self) -> callable:
        def parser(solve_config: "TSP.SolveConfig", **kwargs):
            return print(solve_config.points)

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x, true_char="■", false_char="☐"):
            return true_char if x else false_char

        def parser(state: "TSP.State", **kwargs):
            mask = state.unpacking().mask
            point_mask = jnp.zeros_like(mask).at[state.point].set(True)
            maps = [to_char(x, true_char="↓", false_char=" ") for x in point_mask]
            maps += [to_char(x, true_char="■", false_char="☐") for x in mask]
            return form.format(*maps)

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=jax.random.PRNGKey(0), data=None
    ) -> Puzzle.State:
        mask = jnp.zeros(self.size, dtype=jnp.bool_)
        point = solve_config.start
        mask = mask.at[point].set(True)
        return self.State(mask=mask, point=point).packing()

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        # Split PRNG key so that the start index is independent of point positions.
        key_points, key_start = jax.random.split(key)

        points = jax.random.uniform(
            key_points, shape=(self.size, 2), minval=0, maxval=1, dtype=jnp.float32
        )
        distance_matrix = jnp.linalg.norm(points[:, None] - points[None, :], axis=-1).astype(
            jnp.float32
        )
        start = jax.random.randint(
            key_start, shape=(), minval=0, maxval=self.size, dtype=TYPE
        )
        return self.SolveConfig(points=points, distance_matrix=distance_matrix, start=start)

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: Puzzle.State, filled: bool = True
    ) -> tuple[Puzzle.State, chex.Array]:
        """
        This function returns neighbours and the cost of the move.
        If moving to a point already visited, the cost is infinity.
        """
        # Define possible moves: up, down, left, right
        mask = state.unpacking().mask
        point = state.point

        def move(idx):
            masked = mask[idx] & filled
            new_mask = mask.at[idx].set(True)
            all_visited = jnp.all(new_mask)
            cost = solve_config.distance_matrix[point, idx]
            cost = jnp.where(masked, jnp.inf, cost) + jnp.where(
                all_visited,
                jnp.linalg.norm(
                    solve_config.points[solve_config.start] - solve_config.points[idx], axis=-1
                ),
                0,
            )
            new_state = self.State(mask=new_mask, point=idx).packing()
            return new_state, cost

        # Apply the move function to all possible moves
        new_states, costs = jax.vmap(move)(jnp.arange(self.size, dtype=TYPE))
        costs = jnp.where(filled, costs, jnp.inf)
        return new_states, costs

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: Puzzle.State) -> bool:
        """
        TSP is solved when all points have been visited.
        """
        return jnp.all(state.unpacking().mask)

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"{action:02d}"

    def _get_visualize_format(self):
        size = self.size
        form = " " + "{:s} " * size + "\n"
        form += "[" + "{:s} " * size + "]"
        return form

    def get_solve_config_img_parser(self) -> callable:
        def parser(solve_config: "TSP.SolveConfig", **kwargs):
            raise NotImplementedError("TSP does not support image visualization")

        return parser

    def get_img_parser(self):
        """
        This function returns an img_parser that visualizes the TSP problem.
        It draws all the points scaled to fit into the image, highlights the start point in green,
        marks visited points in blue and unvisited in red, and outlines the current point with a black border.
        If all points are visited, it draws a line from the current point back to the start point.
        """
        import cv2
        import numpy as np

        def img_func(
            state: "TSP.State",
            path: list["TSP.State"],
            idx: int,
            solve_config: "TSP.SolveConfig",
            **kwargs,
        ):
            imgsize = IMG_SIZE[0]
            # Create a white background image
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255

            # Get the visited mask as booleans
            visited = state.unpacking().mask
            # Convert the TSP points (assumed to be an array of shape [number_of_points, 2]) to a numpy array
            points_np = np.array(solve_config.points)

            # Compute scaling parameters to fit all points within the image with a margin
            margin = 20
            if points_np.size > 0:
                xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
                ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()
            else:
                xmin, xmax, ymin, ymax = 0, 1, 0, 1

            # Scale points to image coordinates
            scaled_points = []
            for pt in points_np:
                if xmax > xmin:
                    x_coord = margin + int((pt[0] - xmin) / (xmax - xmin) * (imgsize - 2 * margin))
                else:
                    x_coord = imgsize // 2
                if ymax > ymin:
                    y_coord = margin + int((pt[1] - ymin) / (ymax - ymin) * (imgsize - 2 * margin))
                else:
                    y_coord = imgsize // 2
                scaled_points.append((x_coord, y_coord))

            # Visualize the given path by drawing lines connecting the successive points from 'paths'
            # up to the current index 'idx'
            if path and len(path) > 1:
                route_points = [scaled_points[path[i].point] for i in range(idx + 1)]
                cv2.polylines(
                    img,
                    [np.array(route_points, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 0, 0),
                    thickness=2,
                )

            # Draw each point with different colors based on status
            for i, (x, y) in enumerate(scaled_points):  # Renamed idx to i for clarity
                # Color: green for start, blue for visited, red for unvisited
                if i == solve_config.start:
                    color = (0, 255, 0)
                elif visited[i]:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                cv2.circle(img, (x, y), 5, color, -1)

                # Highlight the current point with an outer black circle
                if i == state.point:
                    cv2.circle(img, (x, y), 8, (0, 0, 0), 2)

                # Optionally, label the point with its index
                cv2.putText(
                    img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1
                )

            # If all points are visited, draw a line from the current point to the start point to close the tour
            if np.all(visited):
                cv2.line(
                    img, scaled_points[state.point], scaled_points[solve_config.start], (0, 0, 0), 2
                )

            return img

        return img_func
