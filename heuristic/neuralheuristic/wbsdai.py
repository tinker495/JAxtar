from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from puxle import Puzzle
from xtructure import xtructure_numpy as xnp

from heuristic.neuralheuristic.neuralheuristic_base import (
    NeuralHeuristicBase as NeuralHeuristic,
)
from JAxtar.astar import astar_builder
from train_util.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from train_util.wbsampling import get_one_solved_branch_distance_samples


def regression_replay_trainer_builder(
    buffer: BUFFER_TYPE,
    train_steps: int,
    preprocess_fn: Callable,
    heuristic_model: NeuralHeuristic,
    optimizer: optax.GradientTransformation,
) -> Callable:
    def regression_loss(
        heuristic_params: Any,
        states: chex.Array,
        target_heuristic: chex.Array,
    ):
        current_heuristic, variable_updates = heuristic_model.apply(
            heuristic_params, states, training=True, mutable=["batch_stats"]
        )
        heuristic_params["batch_stats"] = variable_updates["batch_stats"]
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        loss = jnp.mean(optax.log_cosh(current_heuristic.squeeze(), target_heuristic.squeeze()))
        return loss, (heuristic_params, diff)

    def regression(
        key: chex.PRNGKey,
        buffer_state: BUFFER_STATE_TYPE,
        heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        Performs regression training on the heuristic model using experience replay buffer.
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
    max_depth: int = 300,
    sample_ratio: float = 0.3,
    pop_ratio: float = 0.35,
    use_promising_branch: bool = False,
) -> Callable:
    """
    wbsdai_builder is a function that returns a partial function of wbsdai.
    """

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        search_batch_size,
        max_nodes,
        initial_pop_ratio=pop_ratio,
        initial_cost_weight=cost_weight,
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
            use_promising_branch,
        )
    )

    def get_wbsdai_dataset(
        heuristic_params: Any,
        buffer_state: BUFFER_STATE_TYPE,
        key: chex.PRNGKey,
        *,
        pop_ratio_override: Optional[float] = None,
        cost_weight_override: Optional[float] = None,
    ):
        run = True
        search_count = 0
        solved_count = 0
        search_pop_ratio = pop_ratio if pop_ratio_override is None else pop_ratio_override
        search_cost_weight = cost_weight if cost_weight_override is None else cost_weight_override
        while run:
            solve_configs_list = []
            states_list = []
            true_costs_list = []
            data_len = 0
            while data_len < add_batch_size:
                key, subkey = jax.random.split(key)
                data = jitted_get_one_solved_branch_samples(
                    heuristic_params, subkey, search_pop_ratio, search_cost_weight
                )
                masks = data["masks"]
                solve_configs = data["solve_configs"][masks]
                states = data["states"][masks]
                true_costs = data["true_costs"][masks]
                solved = data["solved"]
                size = jnp.sum(masks)
                solve_configs_list.append(solve_configs)
                states_list.append(states)
                true_costs_list.append(true_costs)
                data_len += size
                search_count += 1
                solved_count += solved
            solve_configs = xnp.concatenate(solve_configs_list, axis=0)
            states = xnp.concatenate(states_list, axis=0)
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
