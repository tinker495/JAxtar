import jax
import jax.numpy as jnp
import chex

from puzzle.puzzle_base import Puzzle
from puzzle.slidepuzzle import SlidePuzzle

def davi(puzzle: Puzzle, steps: int):
    """
    DAVI is a heuristic for the sliding puzzle problem.
    """
    
    pass

def create_shuffled_path(puzzle: Puzzle, shuffle_length: int, batch_size: int, key: chex.PRNGKey):
    target = puzzle.get_target_state()

    def get_trajectory_key(key: chex.PRNGKey):
        def _scan(carry, _):
            state, key = carry
            neighbor_states, cost = puzzle.get_neighbours(state, filled=True)
            filled = jnp.isfinite(cost)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(cost.shape[0]), p=prob, replace=False) 
            next_state = neighbor_states[idx]
            return (next_state, key), next_state

        _, moves = jax.lax.scan(_scan, (target, key), None, length=shuffle_length)
        return moves
    
    moves = jax.vmap(get_trajectory_key)(jax.random.split(key, batch_size)) # [batch_size, shuffle_length][state...]
    return moves