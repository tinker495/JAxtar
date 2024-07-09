import chex
import jax
import jax.numpy as jnp
from puzzle_base import Puzzle, state_dataclass

class SlidePuzzle(Puzzle):

    size: int
    actions = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    @state_dataclass
    class State:
        board: chex.Array

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_string_parser(self):
        form = self.get_visualize_format()
        def parser(state):
            return form.format(*state.board)
        return parser
    
    def get_initial_state(self, key = None) -> State:
        return self.get_random_state(key)

    def get_target_state(self, key = None) -> State:
        return self.get_random_state(key)
    
    def get_neighbours(self, state:State) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        x, y = self.getBlankPosition(state)
        pos = jnp.asarray((x, y))
        next_pos = pos + self.actions
        board = state.board

        def is_valid(x, y):
            return jnp.logical_and(x >= 0, jnp.logical_and(x < self.size, jnp.logical_and(y >= 0, y < self.size)))
        
        def swap(board, x, y, next_x, next_y):
            flat_index = x * self.size + y
            next_flat_index = next_x * self.size + next_y
            old_board = board
            board = board.at[next_flat_index].set(board[flat_index])
            return board.at[flat_index].set(old_board[next_flat_index])
        
        def map(next_pos):
            next_x, next_y = next_pos
            next_board, cost = jax.lax.cond(
                is_valid(next_x, next_y),
                lambda _: (swap(board, x, y, next_x, next_y), 1.0),
                lambda _: (board, jnp.inf),
                None
            )
            return next_board, cost

        next_boards, costs = jax.vmap(
            map, in_axes=(0)
        )(next_pos)
        return self.State(board=next_boards), costs

    def get_visualize_format(self):
        hexa = False
        size = self.size
        if size >= 4:
            hexa = True
        form = "┏━"
        for i in range(size):
            form += "━━┳━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:d}" if not hexa else "{:x}"
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

    def get_random_state(self, key):
        
        def get_random_state(key):
            return self.State(board=jax.random.permutation(key, jnp.arange(0, self.size**2)))
        
        def not_solverable(x):
            state = x[0]
            return ~self.solverable(state)
        
        def while_loop(x):
            state, key = x
            next_key, key = jax.random.split(key)
            state = get_random_state(key)
            return state, next_key

        next_key, key = jax.random.split(key)
        state = get_random_state(key)
        state, _ = jax.lax.while_loop(not_solverable, while_loop, (state, next_key))
        return state

    def solverable(self, state:State):
        """Check if the state is solverable"""
        N = self.size
        inv_count = self.getInvCount(state)
        return jax.lax.cond(
            N % 1,
            lambda _: inv_count % 2 == 0,
            lambda _: jnp.logical_xor(self.getBlankRow(state) % 2 == 0, inv_count % 2 == 0),
            None
        )
    
    def getBlankPosition(self, state:State):
        flat_index = jnp.argmax(state.board == 0)
        return jnp.unravel_index(flat_index, (self.size, self.size))
    
    def getBlankRow(self, state:State):
        return self.getBlankPosition(state)[0]
    
    def getBlankCol(self, state:State):
        return self.getBlankPosition(state)[1]

    def getInvCount(self, state:State):
        
        def is_inv(a, b):
            return jnp.logical_and(a < b, jnp.logical_and(a != 0, b != 0))
        
        n = self.size
        arr = state.board
        def count_inv_i(count, i):
            def count_inv_j(count, j):
                return count + is_inv(arr[i], arr[j])
            return count + jax.lax.fori_loop(i+1, n * n, count_inv_j, 0)
        return jax.lax.fori_loop(0, n * n - 1, count_inv_i, 0)

if __name__ == "__main__":
    puzzle = SlidePuzzle(4)
    states = puzzle.get_random_state(jax.random.PRNGKey(0))
    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(0),10))
    print(states[0])
    print("Solverable : ", puzzle.solverable(states[0]))

    #check solverable is working
    states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]))
    print(states)
    print("Solverable : ", puzzle.solverable(states))
    states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,15,14,0]))
    print(states)
    print("Solverable : ", puzzle.solverable(states))

    #check neighbours
    states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]))
    print(states)
    next_states, costs = puzzle.get_neighbours(states)
    for i in range(4):
        print(next_states[i])
        print(costs[i])

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(0),10))
    next_states, costs = jax.vmap(puzzle.get_neighbours, in_axes=0)(states)
    print(next_states.shape)
    print(costs.shape)