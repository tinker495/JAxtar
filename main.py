import jax
import jax.numpy as jnp

from JAxtar.hash import dataclass_hashing, dataclass_hashing_batch
from puzzle.slidepuzzle import SlidePuzzle
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic

puzzle = SlidePuzzle(4)
heuristic = SlidePuzzleHeuristic(puzzle)
defualt_state = jax.vmap(puzzle.State.default)(jnp.zeros(10))
print(defualt_state[0])
states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(0),10))
print(states[0])
print("Solverable : ", puzzle._solverable(states[0]))