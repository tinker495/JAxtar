import jax
import jax.numpy as jnp

from JAxtar.hash import cuckooHash
from puzzle.slidepuzzle import SlidePuzzle

puzzle = SlidePuzzle(4)
states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(0),10))
print(states[0])
print("Solverable : ", puzzle._solverable(states[0]))

#check solverable is working
states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]))
print(states)
print("Solverable : ", puzzle._solverable(states))
states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,15,14,0]))
print(states)
print("Solverable : ", puzzle._solverable(states))

#check neighbours
states = puzzle.State(board=jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]))
print(states)
next_states, costs = puzzle.get_neighbours(states)
for i in range(4):
    print(next_states[i])
    print(costs[i])

states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(0),10000))
next_states, costs = jax.vmap(puzzle.get_neighbours, in_axes=0)(states)
# vstack (4, 10000, 16) -> (40000, 16)
next_states = jax.tree_util.tree_map(lambda x: jnp.vstack(x), next_states)
costs = jnp.vstack(costs)
print(next_states.shape)
print(next_states.dtype)
print(costs.shape)