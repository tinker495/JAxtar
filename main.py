import jax
import jax.numpy as jnp

from JAxtar.hash import dataclass_hashing, dataclass_hashing_batch
from puzzle.slidepuzzle import SlidePuzzle
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic

puzzle = SlidePuzzle(4)
heuristic = SlidePuzzleHeuristic(puzzle)
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
first_flat = lambda x: jnp.reshape(x, (-1, *x.shape[2:]))
next_states = jax.tree_util.tree_map(first_flat, next_states)
costs = first_flat(costs)
print(next_states.shape)
print(next_states.dtype)
print(costs.shape)

#check hashing
print("States Hashing, this should be not collision")
hashes = dataclass_hashing_batch(states, 1)
#count hash collision
print(hashes.shape)
print(hashes.dtype)
print(jnp.unique(hashes).shape) # No collision

#check hashing
print("Next states Hashing, this should be collision")
hashes = dataclass_hashing_batch(next_states, 1)
#count hash collision
print(hashes.shape)
print(hashes.dtype)
print(jnp.unique(hashes).shape) # Collision

#check heuristic
print("Heuristic")
print(states[0])
print(next_states[0])
diff, _ = heuristic._diff_pos(states[0], next_states[0])
print(diff.shape)
print(diff[:,0])
print(diff[:,1])
dist = heuristic.distance(states[0], next_states[0])
print(dist)