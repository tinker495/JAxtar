import chex
import jax
import jax.numpy as jnp
from puzzle.puzzle_base import Puzzle, state_dataclass
from tabulate import tabulate
from termcolor import colored

TYPE = jnp.uint8

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

class RubiksCube(Puzzle):
    size = 3

    @state_dataclass
    class State:
        # 6 faces, size x size
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - front, 5 - back
        faces: chex.Array

    def __init__(self, size:int = 3):
        self.size = size
        super().__init__()

    def get_string_parser(self):
        def parser(state):
            # Helper function to get face string
            def get_empty_face_string():
                return "\n".join(["  " * (self.size+2) for _ in range(self.size+2)])
            def get_face_string(face):
                face_str = face_map[face]
                string = f"┏━{face_str.center(self.size * 2 - 1, '━')}━┓\n"
                for j in range(self.size):
                    string += "┃ " + ' '.join([colored('■', color_map[int(state.faces[face, j*self.size + i])])
                                 for i in range(self.size)]) + " ┃\n"
                string += "┗━" + "━━" * (self.size - 1) + "━━┛\n"
                return string
            
            # Create the cube string representation
            cube_str = tabulate([[get_empty_face_string(), get_face_string(0), get_empty_face_string(), get_empty_face_string()],
                [get_face_string(2), get_face_string(4), get_face_string(3), get_face_string(5)],
                [get_empty_face_string(), get_face_string(1), get_empty_face_string(), get_empty_face_string()]],
                tablefmt="plain")
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
        pass

    def is_solved(self, state:State, target:State) -> bool:
        return self.is_equal(state, target)

    def _get_random_state(self, key):
        pass

    def _rotate_face(self, state, face, clockwise=True):
        # rotate the face clockwise or counterclockwise
        # face is the index of the face to rotate
        # clockwise is a boolean, True for clockwise, False for counterclockwise
        state.faces[face] = jnp.reshape(
                jnp.rot90(jnp.reshape(state.faces[face], (self.size, self.size)),
                k=1 if clockwise else -1), (self.size * self.size))
        return state
    
    def _rotate_edge(self, state, axis, index, clockwise=True):
        # rotate the edge clockwise or counterclockwise
        # axis is the axis of the rotation, 0 for x, 1 for y, 2 for z
        # index is the index of the edge to rotate
        # clockwise is a boolean, True for clockwise, False for counterclockwise
        pass

    def _rotate_corner(self, state, axis, index, clockwise=True):
        # rotate the corner clockwise or counterclockwise
        # axis is the axis of the rotation, 0 for x, 1 for y, 2 for z
        # index is the index of the corner to rotate
        # clockwise is a boolean, True for clockwise, False for counterclockwise
        pass