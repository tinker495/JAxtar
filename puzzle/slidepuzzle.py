import chex
import jax
import jax.numpy as jnp
from puzzle.puzzle_base import Puzzle, state_dataclass

TYPE = jnp.uint8

class SlidePuzzle(Puzzle):

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
            if x == 0:
                return " "
            if x > 9:
                return chr(x + 55)
            return str(x)

        def parser(state):
            return form.format(*map(to_char, state.board))
        return parser
    
    def get_default_gen(self) -> callable:
        def gen():
            return self.State(board=jnp.zeros(self.size**2, dtype=TYPE))
        return gen

    def get_initial_state(self, key = None) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key = None) -> State:
        return self.State(board=jnp.array([*range(1, self.size**2), 0], dtype=TYPE))
    
    def get_neighbours(self, state:State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        x, y = self._getBlankPosition(state)
        pos = jnp.asarray((x, y))
        next_pos = pos + jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        board = state.board

        def is_valid(x, y):
            return jnp.logical_and(x >= 0, jnp.logical_and(x < self.size, jnp.logical_and(y >= 0, y < self.size)))
        
        def swap(board, x, y, next_x, next_y):
            flat_index = x * self.size + y
            next_flat_index = next_x * self.size + next_y
            old_board = board
            board = board.at[next_flat_index].set(board[flat_index])
            return board.at[flat_index].set(old_board[next_flat_index])
        
        def map_fn(next_pos, filled):
            next_x, next_y = next_pos
            next_board, cost = jax.lax.cond(
                jnp.logical_and(is_valid(next_x, next_y),filled),
                lambda _: (swap(board, x, y, next_x, next_y), 1.0),
                lambda _: (board, jnp.inf),
                None
            )
            return next_board, cost

        next_boards, costs = jax.vmap(
            map_fn, in_axes=(0, None)
        )(next_pos, filled)
        return self.State(board=next_boards), costs

    def is_solved(self, state:State, target:State) -> bool:
        return self.is_equal(state, target)

    def _get_visualize_format(self):
        size = self.size
        form = "┏━"
        for i in range(size):
            form += "━━┳━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s}"
                form += " ┃ " if j != size - 1 else " ┃"
            form += "\n"
            if i != size - 1:
                form += "┣━"
                for j in range(size):
                    form += "━━╋━" if j != size - 1 else "━━┫"
                form += "\n"
        form += "┗━"
        for i in range(size):
            form += "━━┻━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, key):
        """
        This function should return a random state.
        """
        def get_random_state(key):
            return self.State(board=jax.random.permutation(key, jnp.arange(0, self.size**2, dtype=TYPE)))
        
        def not_solverable(x):
            state = x[0]
            return ~self._solvable(state)
        
        def while_loop(x):
            state, key = x
            next_key, key = jax.random.split(key)
            state = get_random_state(key)
            return state, next_key

        next_key, key = jax.random.split(key)
        state = get_random_state(key)
        state, _ = jax.lax.while_loop(not_solverable, while_loop, (state, next_key))
        return state

    def _solvable(self, state:State):
        """Check if the state is solvable"""
        N = self.size
        inv_count = self._getInvCount(state)
        return jax.lax.cond(
            N % 2 == 1,
            lambda inv_count: inv_count % 2 == 0,
            lambda inv_count: jnp.logical_xor(self._getBlankRow(state) % 2 == 0, inv_count % 2 == 0),
            inv_count
        )
    
    def _getBlankPosition(self, state:State):
        flat_index = jnp.argmax(state.board == 0)
        return jnp.unravel_index(flat_index, (self.size, self.size))
    
    def _getBlankRow(self, state:State):
        return self._getBlankPosition(state)[0]
    
    def _getBlankCol(self, state:State):
        return self._getBlankPosition(state)[1]

    def _getInvCount(self, state:State):
        
        def is_inv(a, b):
            return jnp.logical_and(a > b, jnp.logical_and(a != 0, b != 0))
        
        n = self.size
        arr = state.board
        inv_count = 0
        for i in range(n * n):
            for j in range(i + 1, n * n):
                inv_count += is_inv(arr[i], arr[j])
        return inv_count

class SlidePuzzleHard(SlidePuzzle):
    """
    This class is a extension of SlidePuzzle, it will generate the hardest state for the puzzle.
    """
    def __init__(self, size: int):
        super().__init__(size)
        self.size = size
        if size not in [3, 4]:
            raise ValueError("Size of the puzzle must be 3 or 4")
        
        if size == 3:
            self.hardest_state = self.State(board=jnp.array([3, 1, 2, 0, 4, 5, 6, 7, 8], dtype=TYPE))
        elif size == 4:
            self.hardest_state = self.State(board=jnp.array([0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 2, 5, 4, 8, 6, 1], dtype=TYPE))

    def get_initial_state(self, key = None) -> SlidePuzzle.State:
        return self.hardest_state