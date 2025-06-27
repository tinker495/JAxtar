from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax
from puxle import Puzzle

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import get_one_solved_branch_q_samples
from JAxtar.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import QModelBase as QModel


def regression_replay_q_trainer_builder(
    buffer: BUFFER_TYPE,
    train_steps: int,
    preprocess_fn: Callable,
    qfunction: QModel,
    optimizer: optax.GradientTransformation,
) -> Callable:
    def qlearning_loss(
        q_params: Any,
        states: chex.Array,
        actions: chex.Array,
        target_q: chex.Array,
    ):
        q_values, variable_updates = qfunction.apply(
            q_params, states, training=True, mutable=["batch_stats"]
        )
        q_params["batch_stats"] = variable_updates["batch_stats"]
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_q.squeeze() - q_values_at_actions.squeeze()
        loss = jnp.mean(jnp.square(diff))
        return loss, (q_params, diff)

    def regression(
        key: chex.PRNGKey,
        buffer_state: BUFFER_STATE_TYPE,
        qfunction_params: Any,
        opt_state: optax.OptState,
    ):
        """
        WBSDQI is a Q-function for the sliding puzzle problem.
        """

        def train_loop(carry, _):
            qfunction_params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            sample = buffer.sample(buffer_state, subkey)
            solve_configs, states, actions, target_q = (
                sample.experience.first["solve_config"],
                sample.experience.first["state"],
                sample.experience.first["action"],
                sample.experience.first["distance"],
            )
            preprocessed_solve_configs = jax.vmap(preprocess_fn)(solve_configs, states)
            (loss, (qfunction_params, diff)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                qfunction_params,
                preprocessed_solve_configs,
                actions,
                target_q,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=qfunction_params)
            qfunction_params = optax.apply_updates(qfunction_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude))
            return (qfunction_params, opt_state, key), (
                loss,
                diff,
                target_q,
                grad_magnitude_mean,
            )

        (qfunction_params, opt_state, key), (
            losses,
            diffs,
            target_q,
            grad_magnitude_means,
        ) = jax.lax.scan(
            train_loop,
            (qfunction_params, opt_state, key),
            None,
            length=train_steps,
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude_means))
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(qfunction_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.array(weights_magnitude))
        sampled_target_q = jnp.reshape(target_q, (-1,))
        return (
            qfunction_params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_target_q,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    return jax.jit(regression)


def wbsdqi_dataset_builder(
    puzzle: Puzzle,
    qfunction: QModel,
    buffer: BUFFER_TYPE,
    add_batch_size: int = 8192,
    search_batch_size: int = 8192,
    max_nodes: int = int(2e6),
    cost_weight: float = 1.0 - 1e-3,
    max_depth: int = 300,
    sample_ratio: float = 0.3,
    use_optimal_branch: bool = False,
) -> Callable:
    """
    wbsdqi_builder is a function that returns a partial function of wbsdqi.
    """

    qstar_fn = qstar_builder(
        puzzle,
        qfunction,
        search_batch_size,
        max_nodes,
        cost_weight,
        use_q_fn_params=True,
        export_last_pops=True,
    )

    jitted_get_one_solved_branch_q_samples = jax.jit(
        partial(
            get_one_solved_branch_q_samples,
            puzzle,
            qstar_fn,
            max_depth,
            sample_ratio,
            use_optimal_branch,
        )
    )

    def get_wbsdqi_dataset(
        qfunction_params: Any,
        buffer_state: BUFFER_STATE_TYPE,
        key: chex.PRNGKey,
    ):
        run = True
        search_count = 0
        solved_count = 0
        while run:
            solve_configs_list = []
            states_list = []
            actions_list = []
            true_costs_list = []
            data_len = 0
            while data_len < add_batch_size:
                key, subkey = jax.random.split(key)
                (
                    solve_configs,
                    states,
                    actions,
                    true_costs,
                    masks,
                    solved,
                ) = jitted_get_one_solved_branch_q_samples(qfunction_params, subkey)
                solve_configs = solve_configs[masks]
                states = states[masks]
                actions = actions[masks]
                true_costs = true_costs[masks]
                size = jnp.sum(masks)
                solve_configs_list.append(solve_configs)
                states_list.append(states)
                actions_list.append(actions)
                true_costs_list.append(true_costs)
                data_len += size
                search_count += 1
                solved_count += solved
            solve_configs = jax.tree_util.tree_map(
                lambda *x: jnp.concatenate(x, axis=0), *solve_configs_list
            )
            states = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *states_list)
            actions = jnp.concatenate(actions_list, axis=0)
            true_costs = jnp.concatenate(true_costs_list, axis=0)
            split_len = data_len // add_batch_size

            for i in range(split_len):
                timestep = {
                    "solve_config": solve_configs[i * add_batch_size : (i + 1) * add_batch_size],
                    "state": states[i * add_batch_size : (i + 1) * add_batch_size],
                    "distance": true_costs[i * add_batch_size : (i + 1) * add_batch_size],
                    "action": actions[i * add_batch_size : (i + 1) * add_batch_size],
                }
                buffer_state = buffer.add(buffer_state, timestep)

            run = not buffer.can_sample(
                buffer_state
            )  # get datas until the buffer is enough to sample

        return buffer_state, search_count, solved_count, key

    return get_wbsdqi_dataset
