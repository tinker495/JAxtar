import jax
import jax.numpy as jnp
import chex
import optax

from functools import partial
from puzzle.puzzle_base import Puzzle
from typing import Callable
from tqdm import trange

def davi_builder(puzzle: Puzzle, steps: int, total_batch_size: int, shuffle_length: int, minibatch_size: int, heuristic_fn: Callable, heuristic_params: jax.tree_util.PyTreeDef):
    
    create_shuffled_path_fn = partial(create_shuffled_path, puzzle, shuffle_length, total_batch_size // shuffle_length)

    def davi_loss(heuristic_params: jax.tree_util.PyTreeDef, targets: Puzzle.State, current: Puzzle.State, targets_heuristic: chex.Array):
        current_heuristic = jax.vmap(
            heuristic_fn, 
            in_axes=(None, 0, 0) # axis - None, 0, 0
        )(heuristic_params, current, targets)
        diff = targets_heuristic - current_heuristic
        loss = jnp.mean(jnp.square(diff))
        return loss

    optimizer = optax.adabelief(1e-4)
    opt_state = optimizer.init(heuristic_params)

    def davi(key: chex.PRNGKey, target_heuristic_params: jax.tree_util.PyTreeDef, heuristic_params: jax.tree_util.PyTreeDef, opt_state: optax.OptState): 
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        #print("Starting DAVI")
        tiled_targets, shuffled_path = create_shuffled_path_fn(key) # [batch_size, shuffle_length] [batch_size, shuffle_length]
        neighbors, cost = jax.vmap(jax.vmap(puzzle.get_neighbours))(shuffled_path) # [batch_size, shuffle_length, 4] [batch_size, shuffle_length, 4]
        original_shape = cost.shape
        neighbor_len = cost.shape[-1]
        next_tiled_targets = jax.tree_util.tree_map(lambda x: jnp.tile(x[:, :, jnp.newaxis, ...], (1, 1, neighbor_len) + (x.ndim - 2) * (1,)), tiled_targets)

        neighbors_flatten = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), neighbors)
        target_flatten = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), next_tiled_targets)
        cost_flatten = cost.reshape((-1,))
        flatten_size = cost_flatten.shape[0]

        neighbors_heuristics = []
        for i in range(0, flatten_size, minibatch_size):
            neighbors_heuristic = jax.vmap(
                heuristic_fn,
                in_axes=(None, 0, 0)
            )(target_heuristic_params, neighbors_flatten[i:i+minibatch_size], target_flatten[i:i+minibatch_size])
            neighbors_heuristics.append(neighbors_heuristic)
        neighbors_heuristics = jnp.concatenate(neighbors_heuristics, axis=0)
        neighbors_heuristics = neighbors_heuristics.reshape(original_shape)
        
        target_heuristic = jnp.min(neighbors_heuristics + cost, axis=2)

        flatten_target_heuristic = jax.lax.stop_gradient(target_heuristic.reshape((-1,)))
        flatten_shuffled_path = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), shuffled_path)
        target_flatten = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), tiled_targets)

        def train_loop(carry, _):
            heuristic_params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            indexs = jax.random.choice(subkey, flatten_target_heuristic.shape[0], (256,))
            loss, grads = jax.value_and_grad(davi_loss)(heuristic_params, target_flatten[indexs], flatten_shuffled_path[indexs], flatten_target_heuristic[indexs])
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state, key), loss

        (heuristic_params, opt_state, key), losses = jax.lax.scan(train_loop, (heuristic_params, opt_state, key), None, length=steps)
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
            filled = jnp.isfinite(cost).astype(jnp.float32)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(cost.shape[0]), p=prob) 
            next_state = neighbor_states[idx]
            return (next_state, key), next_state

        _, moves = jax.lax.scan(_scan, (target, key), None, length=shuffle_length)
        return moves
    
    moves = jax.vmap(get_trajectory_key)(targets, jax.random.split(key, batch_size)) # [batch_size, shuffle_length][state...]
    tiled_targets = jax.tree_util.tree_map(lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)), targets)
    return tiled_targets, moves
