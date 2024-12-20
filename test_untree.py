import jax
import jax.numpy as jnp

from JAxtar.hash import hash_func_builder, untree_builder
from puzzle.puzzle_base import Puzzle, state_dataclass
from puzzle.slidepuzzle import SlidePuzzle

puzzle = SlidePuzzle(4)

sample = puzzle.get_initial_state(jax.random.PRNGKey(0))
untree = untree_builder(puzzle.State)
hash_func = hash_func_builder(puzzle.State)
untreed = untree(sample)
hashed = hash_func(sample, 0)

print(untreed.shape)
print(untreed.dtype)
print(hashed)
print(hashed.shape)
print(hashed.dtype)


@state_dataclass
class Dummy(Puzzle.State):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray

    def default():
        return Dummy(
            a=jnp.array([1, 2, 3]),
            b=jnp.array([4, 5, 6]),
            c=jnp.array([0.001, 0.002, 0.003], dtype=jnp.float32),
        )


dummy = Dummy(
    a=jnp.array([1, 2, 3]),
    b=jnp.array([4, 5, 6]),
    c=jnp.array([0.001, 0.002, 0.003], dtype=jnp.float32),
)
print(dummy.shape)
print(dummy.dtype)
untree = untree_builder(Dummy)
hash_func = hash_func_builder(Dummy)
untreed = untree(dummy)
hashed = hash_func(dummy, 0)

print(untreed.shape)
print(untreed.dtype)
print(hashed)
print(hashed.shape)
print(hashed.dtype)
