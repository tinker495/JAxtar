import math
from datetime import datetime
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorboardX
from tqdm import trange

from helpers.puzzle_config import default_puzzle_sizes, puzzle_dict, puzzle_q_dict_nn
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_dataset_builder, qlearning_builder

PyTree = Any


@jax.jit
def soft_update(new_tensors: PyTree, old_tensors: PyTree, tau: float):
    return jax.tree_util.tree_map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
    )


@click.command()
@click.option(
    "-p",
    "--puzzle",
    default="n-puzzle",
    type=click.Choice(puzzle_q_dict_nn.keys()),
    help="Puzzle to solve",
)
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--steps", type=int, default=1000000)
@click.option("--shuffle_length", type=int, default=30)
@click.option("--key", type=int, default=0)
@click.option("--reset", is_flag=True, help="Reset the target heuristic params")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("-l", "--loss_threshold", type=float, default=0.05)
def train_qlearning(
    puzzle: str,
    puzzle_size: int,
    steps: int,
    shuffle_length: int,
    key: int,
    reset: bool,
    debug: bool,
    loss_threshold: float,
):
    if debug:
        # disable jit
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle_size)
    puzzle_name = puzzle
    puzzle = puzzle_dict[puzzle_name](puzzle_size)
    qfunc: NeuralQFunctionBase = puzzle_q_dict_nn[puzzle_name](puzzle_size, puzzle, reset)

    # Setup tensorboard logging
    log_dir = (
        f"runs/qlearning_{puzzle_name}_{puzzle_size}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer = tensorboardX.SummaryWriter(log_dir)

    qfunc_fn = qfunc.model.apply
    qfunc_params = qfunc.get_new_params()
    target_qfunc_params = qfunc.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset_size = int(1e5)
    dataset_minibatch_size = 10000
    shuffle_parallel = int(math.ceil(dataset_minibatch_size / shuffle_length))
    minibatch_size = 1000

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients to a maximum global norm of 1.0
        optax.adam(1e-3, nesterov=True),
    )
    opt_state = optimizer.init(qfunc_params)

    qlearning_fn = qlearning_builder(minibatch_size, qfunc_fn, optimizer)
    get_datasets = get_dataset_builder(
        puzzle,
        qfunc.pre_process,
        qfunc_fn,
        dataset_size,
        shuffle_parallel,
        shuffle_length,
        dataset_minibatch_size,
    )

    pbar = trange(steps)
    mean_target_heuristic = 0
    save_count = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(
            target_qfunc_params,
            subkey,
        )
        qfunc_params, opt_state, loss, mean_abs_diff, diffs = qlearning_fn(
            key, dataset, qfunc_params, opt_state
        )
        pbar.set_description(
            f"loss: {loss:.4f}, mean_abs_diff: {mean_abs_diff:.2f}, mean_target_heuristic: {mean_target_heuristic:.4f}"
        )
        if i % 100 == 0:
            target_heuristic = dataset[1]
            mean_target_heuristic = jnp.mean(target_heuristic)
            writer.add_scalar("Mean Target Heuristic", mean_target_heuristic, i)
            writer.add_histogram("Target Heuristic", target_heuristic, i)
            writer.add_histogram("Diff", diffs, i)

            # Log metrics to tensorboard
            writer.add_scalar("Loss", loss, i)
            writer.add_scalar("Mean Abs Diff", mean_abs_diff, i)

        if (i % 500 == 0 and i != 0) and loss <= loss_threshold:
            save_count += 1
            # swap target and current params
            target_qfunc_params, qfunc_params = (qfunc_params, target_qfunc_params)
            opt_state = optimizer.init(qfunc_params)  # reset optimizer state

            if save_count >= 5:
                qfunc.params = target_qfunc_params
                qfunc.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")
                save_count = 0


if __name__ == "__main__":
    train_qlearning()
