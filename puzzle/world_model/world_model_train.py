import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax


def world_model_train_builder(
    minibatch_size: int,
    q_fn: Callable,
    optimizer: optax.GradientTransformation,
):
    def qlearning_loss(
        heuristic_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        target_qs: chex.Array,
    ):
        q_values, variable_updates = q_fn(
            heuristic_params, states, training=True, mutable=["batch_stats"]
        )
        heuristic_params["batch_stats"] = variable_updates["batch_stats"]
        diff = target_qs - q_values
        loss = jnp.mean(jnp.square(diff))
        return loss, (heuristic_params, diff)

    def qlearning(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array],
        heuristic_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        states, target_q = dataset
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            states, target_q = batched_dataset
            (loss, (heuristic_params, diff)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                heuristic_params,
                states,
                target_q,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state), (loss, diff)

        (heuristic_params, opt_state), (losses, diffs) = jax.lax.scan(
            train_loop,
            (heuristic_params, opt_state),
            (batched_states, batched_target_q),
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        return heuristic_params, opt_state, loss, mean_abs_diff, diffs

    return jax.jit(qlearning)
