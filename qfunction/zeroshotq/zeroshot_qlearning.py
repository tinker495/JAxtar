import math
from functools import partial

import chex
import jax

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from puzzle.puzzle_base import Puzzle

# from typing import Any, Callable


# import jax.numpy as jnp
# import optax


# from qfunction.zeroshotq.zeroshotq_base import GoalProjector, ZeroshotQModelBase


def get_zeroshot_qlearning_dataset_builder(
    puzzle: Puzzle,
    buffer: BUFFER_TYPE,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_target: bool = False,
):
    if using_hindsight_target:
        # Calculate appropriate shuffle_parallel for hindsight sampling
        # For hindsight, we're sampling from lower triangle with (L*(L+1))/2 elements
        if using_triangular_target:
            triangle_size = shuffle_length * (shuffle_length + 1) // 2
            needed_parallel = math.ceil(dataset_size / triangle_size)
            shuffle_parallel = int(min(needed_parallel, dataset_minibatch_size))
            steps = math.ceil(dataset_size / (shuffle_parallel * triangle_size))
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
            )
        else:
            shuffle_parallel = int(
                min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
            )
            steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
            )
    else:
        shuffle_parallel = int(
            min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
        )
        steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            shuffle_length,
            shuffle_parallel,
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
            buffer_state = buffer.add(buffer_state, paths)

            return (key, buffer_state), None

        (key, buffer_state), _ = jax.lax.scan(scan_fn, (key, buffer_state), None, length=steps)
        return buffer_state

    return get_datasets
