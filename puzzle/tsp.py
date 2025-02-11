import chex
import jax
import jax.numpy as jnp

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8


class TSP(Puzzle):

    number_of_points: int
    points: chex.Array

    @state_dataclass
    class State:
        mask: chex.Array  # 1D array of size number_of_points, 0 if not visited, 1 if visited
        start: chex.Array  # idx of the point that is the start
        point: chex.Array  # idx of the point that is currently visited

    @property
    def has_target(self) -> bool:
        return False

    def __init__(self, number_of_points: int, key=jax.random.PRNGKey(0)):
        self.number_of_points = number_of_points
        self.points = self.create_points(number_of_points, key)
        super().__init__()

    def create_points(self, number_of_points, key=None):
        return jax.random.uniform(
            key, shape=(number_of_points, 2), minval=0, maxval=1, dtype=jnp.float16
        )

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return "☐" if x == 0 else "■"  # 0: not visited, 1: visited

        def parser(state):
            mask = self.from_uint8(state.mask)
            return form.format(*map(to_char, mask), state.point)

        return parser

    def get_default_gen(self) -> callable:

        number_of_points = self.number_of_points

        def gen():
            mask = jnp.zeros(number_of_points, dtype=jnp.bool_)
            start = jnp.array(0, dtype=TYPE)
            point = jnp.array(0, dtype=TYPE)
            return self.State(mask=self.to_uint8(mask), start=start, point=point)

        return gen

    def get_initial_state(self, key=jax.random.PRNGKey(0)) -> State:
        mask = jnp.zeros(self.number_of_points, dtype=jnp.bool_)
        point = jax.random.randint(
            key, shape=(), minval=0, maxval=self.number_of_points, dtype=TYPE
        )
        mask = mask.at[point].set(True)
        return self.State(mask=self.to_uint8(mask), start=point, point=point)

    def get_target_state(self, key=jax.random.PRNGKey(128)) -> State:
        # this puzzle no target state
        mask = jnp.zeros(self.number_of_points, dtype=jnp.bool_)
        point = jnp.array(0, dtype=TYPE)
        return self.State(mask=self.to_uint8(mask), start=point, point=point)

    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return neighbours, and the cost of the move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Define possible moves: up, down, left, right
        mask = self.from_uint8(
            state.mask
        )  # 1D array of size number_of_points, 0 if not visited, 1 if visited
        point = state.point

        def move(idx):
            masked = mask[idx] & filled
            new_mask = mask.at[idx].set(True)
            all_visited = jnp.all(new_mask)

            cost = jnp.linalg.norm(self.points[point] - self.points[idx], axis=-1)
            cost = jnp.where(masked, jnp.inf, cost) + jnp.where(
                all_visited,
                jnp.linalg.norm(self.points[state.start] - self.points[idx], axis=-1),
                0,
            )
            new_state = self.State(mask=self.to_uint8(new_mask), start=state.start, point=idx)
            return new_state, cost

        # Apply the move function to all possible moves
        new_states, costs = jax.vmap(move)(jnp.arange(self.number_of_points, dtype=TYPE))
        return new_states, costs

    def is_solved(self, state: State, target: State) -> bool:
        return jnp.all(self.from_uint8(state.mask))

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"{action:02d}"

    def _get_visualize_format(self):
        size = self.number_of_points
        form = "[" + "{:s} " * size + "]\n" + "point : [{:02d}]"
        return form

    def to_uint8(self, board: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        return jnp.packbits(board, axis=-1, bitorder="little")

    def from_uint8(self, board: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        return jnp.unpackbits(board, axis=-1, count=self.number_of_points, bitorder="little")

    def get_img_parser(self):
        """
        This function returns an img_parser that visualizes the TSP problem.
        It draws all the points scaled to fit into the image, highlights the start point in green,
        marks visited points in blue and unvisited in red, and outlines the current point with a black border.
        If all points are visited, it draws a line from the current point back to the start point.
        """
        import cv2
        import numpy as np

        def img_func(state: "TSP.State", path: list["TSP.State"], idx: int, **kwargs):
            imgsize = IMG_SIZE[0]
            # Create a white background image
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255

            # Get the visited mask as booleans
            visited = self.from_uint8(state.mask)
            # Convert the TSP points (assumed to be an array of shape [number_of_points, 2]) to a numpy array
            points_np = np.array(self.points)

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
                if i == state.start:
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
                cv2.line(img, scaled_points[state.point], scaled_points[state.start], (0, 0, 0), 2)

            return img

        return img_func
