import chex
import jax
import jax.numpy as jnp

from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8


class TSP(Puzzle):

    number_of_points: int
    points: chex.Array

    @state_dataclass
    class State:
        mask: chex.Array  # 1D array of size number_of_points, 0 if not visited, 1 if visited
        point: chex.Array  # idx of the point that is currently visited

    @property
    def has_target(self) -> bool:
        return False

    def __init__(self, number_of_points: int, key=jax.random.PRNGKey(0)):
        self.number_of_points = number_of_points
        self.points = self.create_points(number_of_points, key)
        super().__init__()

    def create_points(self, number_of_points, key=None):
        return jax.random.uniform(key, shape=(number_of_points, 2), minval=0, maxval=1)

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return " " if x == 0 else "■"  # 0: empty, 1: wall, 2: player

        def parser(state):
            mask = self.from_uint8(state.mask)
            return form.format(*map(to_char, mask))

        return parser

    def get_default_gen(self) -> callable:

        number_of_points = self.number_of_points

        def gen():
            mask = jnp.zeros(number_of_points, dtype=jnp.bool_)
            point = jnp.array([-1], dtype=TYPE)
            return self.State(mask=self.to_uint8(mask), point=point)

        return gen

    def get_initial_state(self, key=jax.random.PRNGKey(0)) -> State:
        mask = jnp.zeros(self.number_of_points, dtype=jnp.bool_)
        point = jax.random.randint(
            key, shape=(1,), minval=0, maxval=self.number_of_points, dtype=TYPE
        )
        return self.State(mask=self.to_uint8(mask), point=point)

    def get_target_state(self, key=jax.random.PRNGKey(128)) -> State:
        # this puzzle no target state
        mask = jnp.ones(self.number_of_points, dtype=jnp.bool_)
        point = jnp.array([-1], dtype=TYPE)
        return self.State(mask=self.to_uint8(mask), point=point)

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
            masked = mask[idx]
            new_mask = mask.at[idx].set(1)

            cost = jnp.linalg.norm(self.points[point] - self.points[idx])
            cost = jnp.where(masked, jnp.inf, cost)
            new_state = self.State(mask=self.to_uint8(new_mask), point=idx)
            return new_state, cost

        # Apply the move function to all possible moves
        new_states, costs = jax.vmap(move)(jnp.arange(self.number_of_points))
        return new_states, costs

    def is_solved(self, state: State, target: State) -> bool:
        return self.is_equal(state, target)

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"{action:02d}"

    def _get_visualize_format(self):
        size = self.number_of_points
        form = "[" + "{:s}" * size + "]\n" + "point : [{:02d}]"
        return form

    def to_uint8(self, board: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        return jnp.packbits(board, axis=-1, bitorder="little")

    def from_uint8(self, board: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        return jnp.unpackbits(board, axis=-1, count=self.number_of_points, bitorder="little")
