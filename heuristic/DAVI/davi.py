import jax
import jax.numpy as jnp
import chex
from functools import partial
from puzzle.puzzle_base import Puzzle
from typing import Callable

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def davi(puzzle: Puzzle, steps: int, total_batch_size: int, shuffle_length: int, heuristic_fn: Callable, key: chex.PRNGKey, heuristic_params: jax.tree_util.PyTreeDef): 
    """
    DAVI is a heuristic for the sliding puzzle problem.
    """

    shuffled_path = create_shuffled_path(puzzle, shuffle_length, total_batch_size, key)

    def train_loop(carry, _):

        return carry, None

    heuristic_params = jax.lax.scan(train_loop, heuristic_params, None, length=steps)
    return heuristic_params

@partial(jax.jit, static_argnums=(0, 1, 2))
def create_shuffled_path(puzzle: Puzzle, shuffle_length: int, batch_size: int, key: chex.PRNGKey):
    targets = jax.vmap(puzzle.get_target_state)(jax.random.split(key, batch_size))

    def get_trajectory_key(target: Puzzle.State, key: chex.PRNGKey):
        def _scan(carry, _):
            state, key = carry
            neighbor_states, cost = puzzle.get_neighbours(state, filled=True)
            filled = jnp.isfinite(cost)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(cost.shape[0]), p=prob, replace=False) 
            next_state = neighbor_states[idx]
            return (next_state, key), next_state

        _, moves = jax.lax.scan(_scan, (target, key), None, length=shuffle_length - 1)
        moves = jax.tree_util.tree_map(lambda t, x: jnp.concatenate([t, x], axis=0), target[jnp.newaxis, ...], moves)
        return moves
    
    moves = jax.vmap(get_trajectory_key)(targets, jax.random.split(key, batch_size)) # [batch_size, shuffle_length][state...]
    moves = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), moves)
    return moves