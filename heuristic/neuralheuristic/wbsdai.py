from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import get_one_solved_branch_distance_samples
from heuristic.neuralheuristic.neuralheuristic_base import (
    NeuralHeuristicBase as NeuralHeuristic,
)
from JAxtar.astar import astar_builder
from puzzle.puzzle_base import Puzzle


def regression_replay_trainer_builder(
    buffer: BUFFER_TYPE,
    train_steps: int,
    preprocess_fn: Callable,
    heuristic_model: NeuralHeuristic,
    optimizer: optax.GradientTransformation,
) -> Callable:
    def regression_loss(
        heuristic_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        target_heuristic: chex.Array,
    ):
        current_heuristic, variable_updates = heuristic_model.apply(
            heuristic_params, states, training=True, mutable=["batch_stats"]
        )
        heuristic_params["batch_stats"] = variable_updates["batch_stats"]
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        loss = jnp.mean(jnp.square(diff))
        return loss, (heuristic_params, diff)

    def regression(
        key: chex.PRNGKey,
        buffer_state: BUFFER_STATE_TYPE,
        heuristic_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """

        def train_loop(carry, _):
            heuristic_params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            sample = buffer.sample(buffer_state, subkey)
            solve_configs, states, target_heuristic = (
                sample.experience.first["solve_config"],
                sample.experience.first["state"],
                sample.experience.first["distance"],
            )
            preprocessed_solve_configs = jax.vmap(preprocess_fn)(solve_configs, states)
            (loss, (heuristic_params, diff)), grads = jax.value_and_grad(
                regression_loss, has_aux=True
            )(
                heuristic_params,
                preprocessed_solve_configs,
                target_heuristic,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude))
            return (heuristic_params, opt_state, key), (
                loss,
                diff,
                target_heuristic,
                grad_magnitude_mean,
            )

        (heuristic_params, opt_state, key), (
            losses,
            diffs,
            target_heuristics,
            grad_magnitude_means,
        ) = jax.lax.scan(
            train_loop,
            (heuristic_params, opt_state, key),
            None,
            length=train_steps,
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude_means))
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(heuristic_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.array(weights_magnitude))
        sampled_target_heuristics = jnp.reshape(target_heuristics, (-1,))
        return (
            heuristic_params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_target_heuristics,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    return jax.jit(regression)


def wbsdai_dataset_builder(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    buffer: BUFFER_TYPE,
    add_batch_size: int = 8192,
    search_batch_size: int = 8192,
    max_nodes: int = int(2e6),
    cost_weight: float = 1.0 - 1e-3,
    max_depth: int = 100,
    sample_ratio: float = 0.3,
    use_topk_branch: bool = False,
) -> Callable:
    """
    wbsdai_builder is a function that returns a partial function of wbsdai.
    """

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        search_batch_size,
        max_nodes,
        cost_weight,
        use_heuristic_params=True,
        export_last_pops=True,
    )

    jitted_get_one_solved_branch_samples = jax.jit(
        partial(
            get_one_solved_branch_distance_samples,
            puzzle,
            astar_fn,
            max_depth,
            sample_ratio,
            use_topk_branch,
        )
    )

    def get_wbsdai_dataset(
        heuristic_params: jax.tree_util.PyTreeDef,
        buffer_state: BUFFER_STATE_TYPE,
        key: chex.PRNGKey,
    ):
        run = True
        search_count = 0
        solved_count = 0
        while run:
            solve_configs_list = []
            states_list = []
            true_costs_list = []
            data_len = 0
            while data_len < add_batch_size:
                key, subkey = jax.random.split(key)
                (
                    solve_configs,
                    states,
                    true_costs,
                    masks,
                    solved,
                ) = jitted_get_one_solved_branch_samples(heuristic_params, subkey)
                solve_configs = solve_configs[masks]
                states = states[masks]
                true_costs = true_costs[masks]
                size = jnp.sum(masks)
                solve_configs_list.append(solve_configs)
                states_list.append(states)
                true_costs_list.append(true_costs)
                data_len += size
                search_count += 1
                solved_count += solved
            solve_configs = jax.tree_util.tree_map(
                lambda *x: jnp.concatenate(x, axis=0), *solve_configs_list
            )
            states = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *states_list)
            true_costs = jnp.concatenate(true_costs_list, axis=0)
            split_len = data_len // add_batch_size

            for i in range(split_len):
                timestep = {
                    "solve_config": solve_configs[i * add_batch_size : (i + 1) * add_batch_size],
                    "state": states[i * add_batch_size : (i + 1) * add_batch_size],
                    "distance": true_costs[i * add_batch_size : (i + 1) * add_batch_size],
                }
                buffer_state = buffer.add(buffer_state, timestep)

            run = not buffer.can_sample(
                buffer_state
            )  # get datas until the buffer is enough to sample

        return buffer_state, search_count, solved_count, key

    return get_wbsdai_dataset
