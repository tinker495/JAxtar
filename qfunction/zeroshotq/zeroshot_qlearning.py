from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import get_random_trajectory
from puzzle.puzzle_base import Puzzle
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQModelBase


def zeroshot_qlearning_builder(
    train_steps: int,
    zeroshot_q_model: ZeroshotQModelBase,
    optimizer: optax.GradientTransformation,
    buffer: BUFFER_TYPE,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    n_devices: int = 1,
):
    def loss_fn(params, solve_configs, target_states, next_states, states, costs, actions):
        pass

    def zeroshot_qlearning(
        key: chex.PRNGKey,
        buffer_state: BUFFER_STATE_TYPE,
        params: Any,
        target_params: Any,
        opt_state: optax.OptState,
    ):
        def train_loop(carry, _):
            params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            sample = buffer.sample(buffer_state, subkey)
            solve_configs, states, costs, actions = (
                sample.experience.first["solve_config"],
                sample.experience.first["state"],
                sample.experience.first["cost"],
                sample.experience.first["action"],
            )

            (loss, diff), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, solve_configs, states, costs, actions
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(
            train_loop, (params, opt_state, key), None, length=train_steps
        )
        return params, opt_state

    return jax.jit(zeroshot_qlearning)


def get_zeroshot_qlearning_dataset_builder(
    puzzle: Puzzle,
    buffer: BUFFER_TYPE,
    shuffle_length: int,
    add_size: int,
    max_parallel: int = 8192,
):
    if add_size > max_parallel:
        steps = add_size // max_parallel
        shuffle_parallel = add_size // steps
    else:
        steps = 1
        shuffle_parallel = add_size
    create_shuffled_path_fn = partial(
        get_random_trajectory, puzzle, shuffle_length, shuffle_parallel
    )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    @jax.jit
    def get_datasets(
        buffer_state: BUFFER_STATE_TYPE,
        key: chex.PRNGKey,
    ):
        def scan_fn(state, _):
            key, buffer_state = state
            key, subkey = jax.random.split(key)
            paths = jited_create_shuffled_path(subkey)

            solve_configs = paths["solve_configs"]
            states = paths["states"]
            actions = paths["actions"]
            action_costs = paths["action_costs"]

            states = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), states)
            actions = jnp.swapaxes(actions, 0, 1)
            action_costs = jnp.swapaxes(action_costs, 0, 1)

            insert_dict = {
                "solve_config": solve_configs,
                "state": states,
                "action": actions,
                "cost": action_costs,
            }
            buffer_state = buffer.add(buffer_state, insert_dict)
            return (key, buffer_state), None

        (key, buffer_state), _ = jax.lax.scan(scan_fn, (key, buffer_state), None, length=steps)
        return buffer_state

    return get_datasets
