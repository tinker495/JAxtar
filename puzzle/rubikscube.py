import chex
import jax
import jax.numpy as jnp
from functools import partial
from puzzle.puzzle_base import Puzzle, state_dataclass
from tabulate import tabulate
from termcolor import colored

TYPE = jnp.uint8

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
FRONT = 4
BACK = 5
face_map = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right',
    4: 'front',
    5: 'back'
}
color_map = {
    0: 'white',
    1: 'yellow',
    2: 'red',
    3: 'magenta',  # orange
    4: 'green',
    5: 'blue'
}

def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])
# (rolled_faces, rotate_axis_for_rolled_faces) 
# 0: x-axis(left), 1: y-axis(up), 2: z-axis(front)

class RubiksCube(Puzzle):
    size: int
    index_grid: chex.Array

    @state_dataclass
    class State:
        # 6 faces, size x size
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - front, 5 - back
        faces: chex.Array

    def __init__(self, size:int = 3):
        self.size = size
        is_even = size % 2 == 0
        self.index_grid = jnp.asarray([
            i for i in range(size) if is_even or not i == (size // 2)
        ], dtype=jnp.uint8)
        super().__init__()

    def get_string_parser(self):
        def parser(state):
            # Helper function to get face string
            def get_empty_face_string():
                return "\n".join(["  " * (self.size+2) for _ in range(self.size+2)])
            def color_legend():
                return "\n".join([f"{face_map[i]:<6}:{colored('■', color_map[i])}" for i in range(6)])
            def get_face_string(face):
                face_str = face_map[face]
                string = f"┏━{face_str.center(self.size * 2 - 1, '━')}━┓\n"
                for j in range(self.size):
                    string += "┃ " + ' '.join([colored('■', color_map[int(state.faces[face, j*self.size + i])])
                                 for i in range(self.size)]) + " ┃\n"
                string += "┗━" + "━━" * (self.size - 1) + "━━┛\n"
                return string
            
            # Create the cube string representation
            cube_str = tabulate([[color_legend(), get_face_string(0)],
                [get_face_string(2), get_face_string(4), get_face_string(3), get_face_string(5)],
                [get_empty_face_string(), get_face_string(1)]],
                tablefmt="plain", rowalign="center")
            return cube_str
        return parser
    
    def get_default_gen(self) -> callable:
        def gen():
            return self.State(faces=jnp.full((6, self.size * self.size), -1).astype(TYPE))
        return gen

    def get_initial_state(self, key = None) -> State:
        return self._get_random_state(key)

    def get_target_state(self, key = None) -> State:
        return self.State(faces=jnp.repeat(jnp.arange(6)[:, None], self.size * self.size, axis=1).astype(TYPE)) # 6 faces, 3x3 each

    def get_neighbours(self, state:State, filled: bool = True) -> tuple[State, chex.Array]:
        def map_fn(face, axis, index, clockwise):
            return jax.lax.cond(
                filled,
                lambda _: (self._rotate(face, axis, index, clockwise), 1.0),
                lambda _: (face, jnp.inf),
                None
            )
        axis_grid, index_grid, clockwise_grid = jnp.meshgrid(jnp.arange(3), self.index_grid, jnp.arange(2))
        axis_grid = axis_grid.reshape(-1)
        index_grid = index_grid.reshape(-1)
        clockwise_grid = clockwise_grid.reshape(-1)
        shaped_faces = state.faces.reshape((6, self.size, self.size))
        shaped_faces, costs = jax.vmap(map_fn, in_axes=(None, 0, 0, 0))(shaped_faces, axis_grid, index_grid, clockwise_grid)
        return self.State(faces=shaped_faces.reshape((-1, 6, self.size * self.size))), costs

    def is_solved(self, state:State, target:State) -> bool:
        return self.is_equal(state, target)

    def _get_random_state(self, key, num_shuffle=12):
        """
        This function should return a random state.
        """
        init_state = self.get_target_state()
        def random_flip(carry, _):
            state, key = carry
            neighbor_states, costs = self.get_neighbours(state, filled=True)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(costs.shape[0]))
            next_state = neighbor_states[idx]
            return (next_state, key), None
        (last_state, _), _ = jax.lax.scan(random_flip, (init_state, key), None, length=num_shuffle)
        return last_state

    @staticmethod
    def _rotate_face(shaped_faces: chex.Array, clockwise: bool, mul: int):
        return rot90_traceable(shaped_faces, jnp.where(clockwise, mul,-mul))

    def _rotate(self, shaped_faces: chex.Array, axis: int, index: int, clockwise: bool = True):
        # rotate the edge clockwise or counterclockwise
        # axis is the axis of the rotation, 0 for x, 1 for y, 2 for z
        # index is the index of the edge to rotate
        # clockwise is a boolean, True for clockwise, False for counterclockwise
        rotate_edge_map = jnp.array([
            [UP    , FRONT  , DOWN  , BACK  ], # x-axis
            [LEFT  , FRONT  , RIGHT , BACK  ], # y-axis
            [UP    , LEFT   , DOWN  , RIGHT ], # z-axis
        ])
        rotate_edge_rot = jnp.array([
            [-1, -1, -1, -1],   # x-axis
            [2, 2, 2, 0],       # y-axis  
            [2, 1, 0, 3],       # z-axis
        ])
        edge_faces = rotate_edge_map[axis]
        edge_rot = rotate_edge_rot[axis]
        shaped_faces = shaped_faces.at[BACK].set(jnp.flip(jnp.flip(shaped_faces[BACK], axis=0), axis=1))
        rolled_faces = shaped_faces[edge_faces]
        rolled_faces = jax.vmap(lambda face, rot: rot90_traceable(face, k=rot))(rolled_faces, edge_rot)
        rolled_faces = rolled_faces.at[:, index, :].set(jnp.roll(rolled_faces[:, index, :], jnp.where(clockwise, 1, -1), axis=0))
        rolled_faces = jax.vmap(lambda face, rot: rot90_traceable(face, k=-rot))(rolled_faces, edge_rot)
        shaped_faces = shaped_faces.at[edge_faces].set(rolled_faces)
        shaped_faces = shaped_faces.at[BACK].set(jnp.flip(jnp.flip(shaped_faces[BACK], axis=1), axis=0))
        is_edge = jnp.isin(index, jnp.array([0, self.size - 1]))
        switch_num = jnp.where(is_edge, 1 + 2 * axis + index // (self.size - 1), 0) # 0: None, 1: left, 2: right, 3: up, 4: down, 5: front, 6: back
        shaped_faces = jax.lax.switch(switch_num, [
            lambda: shaped_faces, # 0: None
            lambda: shaped_faces.at[LEFT].set(self._rotate_face(shaped_faces[LEFT], clockwise, -1)), # 1: left
            lambda: shaped_faces.at[RIGHT].set(self._rotate_face(shaped_faces[RIGHT], clockwise, 1)), # 2: right
            lambda: shaped_faces.at[DOWN].set(self._rotate_face(shaped_faces[DOWN], clockwise, -1)), # 3: down
            lambda: shaped_faces.at[UP].set(self._rotate_face(shaped_faces[UP], clockwise, 1)), # 4: up
            lambda: shaped_faces.at[FRONT].set(self._rotate_face(shaped_faces[FRONT], clockwise, 1)), # 5: front
            lambda: shaped_faces.at[BACK].set(self._rotate_face(shaped_faces[BACK], clockwise, -1)), # 6: back
        ])
        return shaped_faces