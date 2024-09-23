import chex
import jax
import jax.numpy as jnp
from puzzle.puzzle_base import Puzzle, state_dataclass
from termcolor import colored

TYPE = jnp.uint16

class Maze(Puzzle):

    size: int
    maze: chex.Array

    @state_dataclass
    class State:
        pos: chex.Array

    def __init__(self, size: int, p=0.3, key = jax.random.PRNGKey(0)):
        self.size = size
        self.maze = self.create_maze(size, p, key)
        super().__init__()

    def create_maze(self, size, p=0.3, key=None):
        maze = jax.random.bernoulli(key, p=jnp.float32(p), shape=(size**2,)).astype(jnp.uint8)
        return self.to_uint8(maze)

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return " " if x == 0 else "■" if x == 1 else colored("●", "red") # 0: empty, 1: wall, 2: player

        def parser(state):
            maze_with_pos = self.from_uint8(self.maze)
            maze_with_pos = maze_with_pos.at[state.pos[0] * self.size + state.pos[1]].set(2)
            return form.format(*map(to_char, maze_with_pos))
        return parser
    
    def get_default_gen(self) -> callable:
        def gen():
            return self.State(pos=jnp.array([-1, -1], dtype=TYPE))
        return gen

    def get_initial_state(self, key = jax.random.PRNGKey(0)) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key = jax.random.PRNGKey(128)) -> State:
        return self._get_random_state(key)

    def get_neighbours(self, state:State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return neighbours, and the cost of the move.
        If impossible to move in a direction, cost should be inf and State should be same as input state.
        """
        # Define possible moves: up, down, left, right
        moves = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        
        def move(state, move):
            new_pos = (state.pos + move).astype(TYPE)
            
            maze = self.from_uint8(self.maze)
            # Check if the new position is within the maze bounds and not a wall
            valid_move = (new_pos >= 0).all() & (new_pos < self.size).all() & (maze[new_pos[0] * self.size + new_pos[1]] == 0) & filled
            
            # If the move is valid, update the position. Otherwise, keep the old position.
            new_state = self.State(
                pos=jnp.where(valid_move, new_pos, state.pos)
            )
            
            # Cost is 1 for valid moves, inf for invalid moves
            cost = jnp.where(valid_move, 1.0, jnp.inf)
            
            return new_state, cost

        # Apply the move function to all possible moves
        new_states, costs = jax.vmap(lambda m: move(state, m))(moves)
        
        return new_states, costs

    def is_solved(self, state:State, target:State) -> bool:
        return self.is_equal(state, target)

    def _get_visualize_format(self):
        size = self.size
        form = "┏━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s} "
            form += "┃"
            form += "\n"
        form += "┗━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, key):
        """
        This function should return a random state.
        """
        def get_random_state(key):
            return self.State(pos=jax.random.randint(key, (2,), 0, self.size, dtype=TYPE))
        
        def is_not_wall(x):
            state = x[0]
            maze = self.from_uint8(self.maze)
            return maze[state.pos[0] * self.size + state.pos[1]] != 0
        
        def while_loop(x):
            state, key = x
            next_key, key = jax.random.split(key)
            state = get_random_state(key)
            return state, next_key

        next_key, key = jax.random.split(key)
        state = get_random_state(key)
        state, _ = jax.lax.while_loop(is_not_wall, while_loop, (state, next_key))
        return state

    def to_uint8(self, board: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        return jnp.packbits(board, axis=-1, bitorder='little')
    
    def from_uint8(self, board: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        return jnp.unpackbits(board, axis=-1, count=self.size**2, bitorder='little')