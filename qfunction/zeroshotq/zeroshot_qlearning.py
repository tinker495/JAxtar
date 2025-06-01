from functools import partial

import chex
import jax
import jax.numpy as jnp

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import get_random_trajectory
from puzzle.puzzle_base import Puzzle

# from typing import Any, Callable


# import jax.numpy as jnp
# import optax


# from qfunction.zeroshotq.zeroshotq_base import GoalProjector, ZeroshotQModelBase


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
