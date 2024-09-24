import jax
import jax.numpy as jnp
import chex
import optax

from functools import partial
from puzzle.puzzle_base import Puzzle
from typing import Callable

def davi_builder(puzzle: Puzzle, steps: int, total_batch_size: int, shuffle_length: int, heuristic_fn: Callable, heuristic_params: jax.tree_util.PyTreeDef):
    
    create_shuffled_path_fn = partial(create_shuffled_path, puzzle, shuffle_length, total_batch_size // shuffle_length)

    def davi_loss(heuristic_params: jax.tree_util.PyTreeDef, targets: Puzzle.State, current: Puzzle.State, targets_heuristic: chex.Array):
        current_heuristic = jax.vmap(
            jax.vmap(
                heuristic_fn,
                in_axes=(None, 0, None) # axis - None, 1, 0
            ), 
            in_axes=(None, 0, 0) # axis - None, 0, 0
        )(heuristic_params, current, targets)
        diff = current_heuristic - targets_heuristic
        loss = jnp.mean(jnp.square(diff))
        return loss

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(heuristic_params)

    def davi(key: chex.PRNGKey, heuristic_params: jax.tree_util.PyTreeDef, opt_state: optax.OptState): 
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        #print("Starting DAVI")
        targets, shuffled_path = create_shuffled_path_fn(key)
        neighbors, cost = jax.vmap(jax.vmap(puzzle.get_neighbours))(shuffled_path)
        neighbors_heuristic = jax.vmap(
            jax.vmap(
                jax.vmap(heuristic_fn, in_axes=(None, 0, None)), # axis - None, 2, 0
                in_axes=(None, 0, None) # axis - None, 1, 0
            ), 
            in_axes=(None, 0, 0) # axis - None, 0, 0
        )(heuristic_params, neighbors, targets)
        min_neighbors_heuristic = jnp.min(neighbors_heuristic, axis=2)
        target_heuristic = min_neighbors_heuristic + jnp.min(cost, axis=2)

        def train_loop(carry, _):
            heuristic_params, opt_state = carry
            loss, grads = jax.value_and_grad(davi_loss)(heuristic_params, targets, shuffled_path, target_heuristic)
            updates, opt_state = optimizer.update(grads, opt_state)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state), loss

        (heuristic_params, opt_state), losses = jax.lax.scan(train_loop, (heuristic_params, opt_state), None, length=steps)
        loss = jnp.mean(losses)
        mean_target_heuristic = jnp.mean(target_heuristic)
        return heuristic_params, opt_state, loss, mean_target_heuristic
    
    return jax.jit(davi), opt_state

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

        _, moves = jax.lax.scan(_scan, (target, key), None, length=shuffle_length)
        return moves
    
    moves = jax.vmap(get_trajectory_key)(targets, jax.random.split(key, batch_size)) # [batch_size, shuffle_length][state...]
    return targets, moves