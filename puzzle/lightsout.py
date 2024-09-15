import chex
import jax
import jax.numpy as jnp
from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8

class LightsOut(Puzzle):

    size: int

    @state_dataclass
    class State:
        board: chex.Array

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return "□" if x == 0 else "■"

        def parser(state):
            return form.format(*map(to_char, state.board))
        return parser
    
    def get_default_gen(self) -> callable:
        def gen():
            return self.State(board=jnp.full(self.size**2, -1, dtype=TYPE))
        return gen

    def get_initial_state(self, key = None) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key = None) -> State:
        return self.State(board=jnp.zeros(self.size**2, dtype=TYPE))

    def get_neighbours(self, state:State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        board = state.board
        # actions - combinations of range(size) with 2 elements
        actions = jnp.stack(jnp.meshgrid(jnp.arange(self.size), jnp.arange(self.size)), axis=-1).reshape(-1, 2)

        def flip(board, action):
            x, y = action
            xs = jnp.clip(jnp.array([x, x, x + 1, x - 1, x]), 0, self.size - 1)
            ys = jnp.clip(jnp.array([y, y + 1, y, y, y - 1]), 0, self.size - 1)
            idxs = xs * self.size + ys
            return board.at[idxs].set(~board[idxs])

        def map_fn(action, filled):
            next_board, cost = jax.lax.cond(
                filled,
                lambda _: (flip(board, action), 1.0),
                lambda _: (board, jnp.inf),
                None
            )
            return next_board, cost

        next_boards, costs = jax.vmap(map_fn, in_axes=(0, None))(actions, filled)
        return self.State(board=next_boards), costs

    def is_solved(self, state:State, target:State) -> bool:
        return self.is_equal(state, target)

    def _get_visualize_format(self):
        size = self.size
        form = "┏"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃"
            for j in range(size):
                form += "{:s} "
            form += "┃"
            form += "\n"
        form += "┗"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, key, num_shuffle=100):
        """
        This function should return a random state.
        """
        init_state = self.State(board=jnp.zeros(self.size**2, dtype=TYPE))
        def random_flip(carry, _):
            state, key = carry
            neighbor_states, _ = self.get_neighbours(state, filled=True)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(self.size**2))
            next_state = neighbor_states[idx]
            return (next_state, key), None
        (last_state, _), _ = jax.lax.scan(random_flip, (init_state, key), None, length=num_shuffle)
        return last_state
