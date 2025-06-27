from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax
from puxle import Puzzle

from helpers.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from helpers.sampling import get_random_trajectory
from neural_util.target_update import soft_update
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQModelBase


def zeroshot_qlearning_builder(
    train_steps: int,
    zeroshot_q_model: ZeroshotQModelBase,
    optimizer: optax.GradientTransformation,
    buffer: BUFFER_TYPE,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
):
    def target_fn(target_params, solve_configs, nstate_i, state_j, costs):
        target_z = zeroshot_q_model.apply(
            target_params,
            solve_configs,
            method=zeroshot_q_model.solve_config_projection,
        )  # (batch_size, latent_dim)
        nstate_f = zeroshot_q_model.apply(
            target_params,
            nstate_i,
            target_z,
            method=zeroshot_q_model.forward_projection,
        )  # (batch_size, action_size, latent_dim)
        state_j_b = zeroshot_q_model.apply(
            target_params,
            state_j,
            method=zeroshot_q_model.backward_projection,
        )  # (batch_size, latent_dim)
        target_z = zeroshot_q_model.apply(
            target_params,
            nstate_f,
            target_z,
            method=zeroshot_q_model.distance,
        )  # (batch_size, action_size)
        target_z = jnp.min(target_z, axis=1) + costs  # (batch_size,)
        target_j = zeroshot_q_model.apply(
            target_params,
            nstate_f,
            state_j_b,
            method=zeroshot_q_model.distance,
        )  # (batch_size, action_size)
        target_j = jnp.min(target_j, axis=1) + costs  # (batch_size,)
        return target_z, target_j  # (batch_size,), (batch_size,)

    def loss_fn(params, solve_config, state_i, state_j, action, target_z, target_j):
        batch_size = action.shape[0]

        z = zeroshot_q_model.apply(
            params,
            solve_config,
            method=zeroshot_q_model.solve_config_projection,
        )
        b_i = zeroshot_q_model.apply(
            params,
            state_i,
            method=zeroshot_q_model.backward_projection,
        )
        b_j = zeroshot_q_model.apply(
            params,
            state_j,
            method=zeroshot_q_model.backward_projection,
        )
        f = zeroshot_q_model.apply(
            params,
            state_i,
            z,
            method=zeroshot_q_model.forward_projection,
        )

        q_z = zeroshot_q_model.apply(
            params,
            f,
            z,
            method=zeroshot_q_model.distance,
        )  # (batch_size, action_size)
        q_z = q_z[jnp.arange(batch_size), action]  # (batch_size,)
        q_b = zeroshot_q_model.apply(
            params,
            f,
            b_j,
            method=zeroshot_q_model.distance,
        )  # (batch_size, action_size)
        q_b = q_b[jnp.arange(batch_size), action]  # (batch_size,)
        q_self = zeroshot_q_model.apply(
            params,
            f,
            b_i,
            method=zeroshot_q_model.distance,
        )  # (batch_size, action_size)
        q_self = q_self[jnp.arange(batch_size), action]  # (batch_size,)

        diff_z = target_z - q_z
        diff_j = target_j - q_b

        loss_z = jnp.mean(diff_z**2)
        loss_j = jnp.mean(diff_j**2)
        loss_self = jnp.mean(q_self**2)

        total_loss = 1e-2 * loss_z + 1e-2 * loss_j + loss_self
        return total_loss, (loss_z, loss_j, loss_self)

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
                sample.experience.first["solve_config"],  # (batch_size, ...)
                sample.experience.first["state"],  # (batch_size, traj_len + 1, ...)
                sample.experience.first["cost"],  # (batch_size, traj_len)
                sample.experience.first["action"],  # (batch_size, traj_len)
            )
            traj_len = actions.shape[1]
            batch_size = actions.shape[0]

            idx_i = jax.random.randint(key, (batch_size,), 0, traj_len)  # (batch_size,)
            idx_j = jax.random.randint(key, (batch_size,), idx_i + 1, traj_len + 1)  # (batch_size,)

            states_i = states[jnp.arange(batch_size), idx_i]  # (batch_size, ...)
            nstates_i = states[jnp.arange(batch_size), idx_i + 1]  # (batch_size, ...)
            states_j = states[jnp.arange(batch_size), idx_j]  # (batch_size, ...)
            actions = actions[jnp.arange(batch_size), idx_i]  # (batch_size,)
            costs = costs[jnp.arange(batch_size), idx_i]  # (batch_size,)

            p_solve_configs = jax.vmap(solve_config_preproc_fn)(solve_configs)
            p_states_i = jax.vmap(state_preproc_fn)(states_i)
            p_nstates_i = jax.vmap(state_preproc_fn)(nstates_i)
            p_states_j = jax.vmap(state_preproc_fn)(states_j)

            target_z, target_j = target_fn(
                target_params, p_solve_configs, p_nstates_i, p_states_j, costs
            )

            (total_loss, (loss_z, loss_j, loss_self)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params, p_solve_configs, p_states_i, p_states_j, actions, target_z, target_j)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), (
                total_loss,
                loss_z,
                loss_j,
                loss_self,
                target_z,
                target_j,
            )

        (params, opt_state, key), (
            total_loss,
            loss_z,
            loss_j,
            loss_self,
            target_z,
            target_j,
        ) = jax.lax.scan(train_loop, (params, opt_state, key), None, length=train_steps)
        target_params = soft_update(target_params, params, 0.9)
        total_loss = jnp.mean(total_loss)
        loss_z = jnp.mean(loss_z)
        loss_j = jnp.mean(loss_j)
        loss_self = jnp.mean(loss_self)
        target_z = jnp.reshape(target_z, (-1,))
        target_j = jnp.reshape(target_j, (-1,))
        return (
            params,
            target_params,
            opt_state,
            {
                "total_loss": total_loss,
                "loss_z": loss_z,
                "loss_j": loss_j,
                "loss_self": loss_self,
                "target_z": target_z,
                "target_j": target_j,
            },
        )

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
