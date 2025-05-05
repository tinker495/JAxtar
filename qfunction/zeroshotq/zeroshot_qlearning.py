import math
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.puzzle_base import Puzzle
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQModelBase


def zeroshot_qlearning_builder(
    minibatch_size: int,
    zeroshotq_model: ZeroshotQModelBase,
    optimizer: optax.GradientTransformation,
    n_devices: int = 1,
):
    latent_dim = zeroshotq_model.latent_dim

    def qlearning_loss(
        q_params: jax.tree_util.PyTreeDef,
        target_q_params: jax.tree_util.PyTreeDef,
        solve_configs_i: chex.Array,
        states_i: chex.Array,
        actions_i: chex.Array,
        costs_i: chex.Array,
        states_j: chex.Array,
        actions_j: chex.Array,
        key: chex.PRNGKey,
    ):
        batch_size = actions_i.shape[0]

        # ---------- solve config loss ----------

        solve_config_z, _ = zeroshotq_model.apply(
            q_params,
            solve_configs_i,
            training=True,
            method=zeroshotq_model.solve_config_projection,
        )
        solve_config_q_ij, _ = zeroshotq_model.apply(
            q_params,
            solve_config_z,
            states_i,
            states_j,
            actions_j,
            training=True,
            method=zeroshotq_model.get_q_ij,
        )
        solve_config_q_ij_at_actions = jnp.take_along_axis(
            solve_config_q_ij, actions_i[:, jnp.newaxis], axis=1
        )  # (batch_size,)

        solve_config_target_q_ij, _ = zeroshotq_model.apply(
            target_q_params,
            solve_config_z,
            states_i,
            states_j,
            actions_j,
            training=False,
            method=zeroshotq_model.get_q_ij,
        )
        solve_config_target_pi_ij = boltzmann_action_selection(
            solve_config_target_q_ij
        )  # (batch_size, action_size)
        solve_config_target_v_ij = (
            jnp.sum(solve_config_target_pi_ij * solve_config_target_q_ij, axis=1) + costs_i
        )  # (batch_size,)

        solve_config_td_error = solve_config_target_v_ij - solve_config_q_ij_at_actions
        solve_config_td_loss = jnp.mean(jnp.square(solve_config_td_error))

        # ---------- sampled loss ----------

        sampled_z = jax.random.normal(key, (batch_size, latent_dim))
        sampled_q_ij, _ = zeroshotq_model.apply(
            q_params,
            sampled_z,
            states_i,
            states_j,
            actions_j,
            training=True,
            method=zeroshotq_model.get_q_ij,
        )
        sampled_q_ij_at_actions = jnp.take_along_axis(
            sampled_q_ij, actions_i[:, jnp.newaxis], axis=1
        )  # (batch_size,)

        sampled_target_q_ij, _ = zeroshotq_model.apply(
            target_q_params,
            sampled_z,
            states_i,
            states_j,
            actions_j,
            training=False,
            method=zeroshotq_model.get_q_ij,
        )  # (batch_size, action_size)
        sampled_target_pi_ij = boltzmann_action_selection(
            sampled_target_q_ij
        )  # (batch_size, action_size)
        sampled_target_v_ij = (
            jnp.sum(sampled_target_pi_ij * sampled_target_q_ij, axis=1) + costs_i
        )  # (batch_size,)

        sampled_td_error = sampled_target_v_ij - sampled_q_ij_at_actions
        sampled_td_loss = jnp.mean(jnp.square(sampled_td_error))

        # ---------- fb alignment loss ----------

        sampled_align_q, _ = zeroshotq_model.apply(
            q_params,
            sampled_z,
            states_i,
            states_i,
            actions_i,
            training=True,
            method=zeroshotq_model.get_q_ij,
        )  # (batch_size, action_size)
        sampled_align_q_at_actions = jnp.take_along_axis(
            sampled_align_q, actions_i[:, jnp.newaxis], axis=1
        )  # (batch_size,)
        fb_align_loss = jnp.mean(sampled_align_q_at_actions)

        # ---------- orthogonal loss ----------

        b_i = zeroshotq_model.apply(
            q_params, states_i, actions_i, training=True, method=zeroshotq_model.get_b
        )  # (batch_size, latent_dim)
        b_j = zeroshotq_model.apply(
            q_params, states_j, actions_j, training=True, method=zeroshotq_model.get_b
        )  # (batch_size, latent_dim)

        # Term 1: (1/b^2) * sum_{i,j} B(si, ai)^T * sg(B(sj, aj))
        term1_matrix = jnp.einsum("iz,jz->ij", b_i, jax.lax.stop_gradient(b_j))

        # Term 2: (1/b^2) * sum_{i,j} sg(B(si, ai)^T * B(sj, aj))
        term2_matrix = jnp.einsum("iz,jz->ij", b_i, b_j)
        term2_matrix_sg = jax.lax.stop_gradient(term2_matrix)

        # Term 3: (1/b) * sum_i B(si, ai)^T * sg(B(si, ai))
        term3_vector = jnp.einsum("iz,iz->i", b_i, jax.lax.stop_gradient(b_i))

        # Calculate means (equivalent to sum / b^2 or sum / b)
        loss_term1 = jnp.mean(term1_matrix * term2_matrix_sg)
        loss_term2 = jnp.mean(term3_vector)

        ortho_loss = loss_term1 - loss_term2

        # Combine all losses (assuming equal weight for now)
        total_loss = solve_config_td_loss + sampled_td_loss + fb_align_loss + ortho_loss

        return total_loss, q_params

    def qlearning(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: jax.tree_util.PyTreeDef,
        target_q_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        solve_configs = dataset["solve_configs"]
        states = dataset["states"]
        actions = dataset["actions"]
        costs = dataset["cost"]
        data_size = actions.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        key_i, key_j, key = jax.random.split(key, 3)

        batch_indexs_i = jnp.concatenate(
            [
                jax.random.permutation(key_i, jnp.arange(data_size)),
                jax.random.randint(key_i, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs_i = jnp.reshape(batch_indexs_i, (batch_size, minibatch_size))
        batch_indexs_j = jnp.concatenate(
            [
                jax.random.permutation(key_j, jnp.arange(data_size)),
                jax.random.randint(key_j, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs_j = jnp.reshape(batch_indexs_j, (batch_size, minibatch_size))

        batched_solve_configs_i = jnp.take(solve_configs, batch_indexs_i, axis=0)
        batched_states_i = jnp.take(states, batch_indexs_i, axis=0)
        batched_actions_i = jnp.take(actions, batch_indexs_i, axis=0)
        batched_costs_i = jnp.take(costs, batch_indexs_i, axis=0)

        batched_states_j = jnp.take(states, batch_indexs_j, axis=0)
        batched_actions_j = jnp.take(actions, batch_indexs_j, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            (
                solve_configs_i,
                states_i,
                states_j,
                actions_i,
                actions_j,
                costs_i,
            ) = batched_dataset
            (loss, q_params), grads = jax.value_and_grad(qlearning_loss, has_aux=True)(
                q_params,
                target_q_params,
                solve_configs_i,
                states_i,
                actions_i,
                costs_i,
                states_j,
                actions_j,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.concatenate(grad_magnitude))
            return (q_params, opt_state), (loss, grad_magnitude_mean)

        (q_params, opt_state), (losses, grad_magnitude_means) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (
                batched_solve_configs_i,
                batched_states_i,
                batched_states_j,
                batched_actions_i,
                batched_actions_j,
                batched_costs_i,
            ),
        )
        loss = jnp.mean(losses)
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(grad_magnitude_means)
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(q_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.concatenate(weights_magnitude))
        return (
            q_params,
            opt_state,
            loss,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    if n_devices > 1:

        def pmap_qlearning(key, dataset, q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude,) = jax.pmap(
                qlearning, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, q_params, opt_state)
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            return qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude

        return pmap_qlearning
    else:
        return jax.jit(qlearning)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
    mask: chex.Array = None,
) -> chex.Array:
    q_values = -q_values / temperature
    probs = jnp.exp(q_values)
    if mask is not None:
        probs = jnp.where(mask, probs, 0.0)
    else:
        mask = jnp.ones_like(probs)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)
    uniform_prob = mask.astype(jnp.float32) / jnp.sum(mask, axis=1, keepdims=True)
    probs = probs * (1 - epsilon) + uniform_prob * epsilon
    return probs


def _get_datasets(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    zeroshotq_model: ZeroshotQModelBase,
    minibatch_size: int,
    q_params: jax.tree_util.PyTreeDef,
    shuffled_path: tuple[Puzzle.SolveConfig, Puzzle.State, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs, target_state, shuffled_path, move_costs = shuffled_path

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_shuffled_path = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), shuffled_path
    )
    minibatched_move_costs = jnp.reshape(move_costs, (-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, shuffled_path, move_costs = vals

        solve_config_preproc = jax.vmap(solve_config_preproc_fn)(solve_configs)
        state_preproc = jax.vmap(state_preproc_fn)(shuffled_path)

        solve_config_z, _ = zeroshotq_model.apply(
            q_params,
            solve_config_preproc,
            training=False,
            mutable=["batch_stats"],
            method=zeroshotq_model.solve_config_projection,
        )
        f_a = zeroshotq_model.apply(
            q_params,
            state_preproc,
            solve_config_z,
            training=False,
            mutable=["batch_stats"],
            method=zeroshotq_model.forward_projection,
        )
        q_values, _ = zeroshotq_model.apply(
            q_params,
            f_a,
            solve_config_z,
            training=False,
            mutable=["batch_stats"],
            method=zeroshotq_model.distance,
        )
        _, cost = puzzle.batched_get_neighbours(
            solve_configs, shuffled_path, filleds=jnp.ones_like(move_costs), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        mask = jnp.isfinite(jnp.transpose(cost, (1, 0)))

        probs = boltzmann_action_selection(q_values, mask=mask)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(key, q_values.shape[0]), probs
        )

        return key, (solve_config_preproc, state_preproc, actions, cost)

    _, (solve_config_preproc, state_preproc, actions, cost) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_shuffled_path, minibatched_move_costs),
    )

    solve_config_preproc = solve_config_preproc.reshape((-1, *solve_config_preproc.shape[2:]))
    state_preproc = state_preproc.reshape((-1, *state_preproc.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    cost = cost.reshape((-1, *cost.shape[2:]))
    return {
        "solve_configs": solve_config_preproc,
        "states": state_preproc,
        "actions": actions,
        "cost": cost,
    }


def get_zeroshot_qlearning_dataset_builder(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    zeroshotq_model: ZeroshotQModelBase,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_target: bool = False,
    n_devices: int = 1,
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

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            solve_config_preproc_fn,
            state_preproc_fn,
            zeroshotq_model,
            dataset_minibatch_size,
        )
    )

    @jax.jit
    def get_datasets(
        target_q_params: jax.tree_util.PyTreeDef,
        q_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        def scan_fn(key, _):
            key, subkey = jax.random.split(key)
            paths = jited_create_shuffled_path(subkey)
            return key, paths

        key, paths = jax.lax.scan(scan_fn, key, None, length=steps)
        paths = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:]))[:dataset_size], paths
        )

        flatten_dataset = jited_get_datasets(target_q_params, q_params, paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(target_q_params, q_params, key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(None, None, 0))(
                target_q_params, q_params, keys
            )
            return datasets

        return pmap_get_datasets
    else:
        return get_datasets


def create_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key, shuffle_parallel))
    targets = jax.vmap(puzzle.solve_config_to_state_transform, in_axes=(0, 0))(
        solve_configs, jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, key, move_cost_ = carry
        neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost_), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])  # [action]
        idx = jax.vmap(lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1))(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, idx
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, idx)  # [batch]
        move_cost = move_cost_ + cost
        return (state, next_state, key, move_cost), (state, move_cost_)

    _, (moves, move_costs) = jax.lax.scan(
        _scan, (targets, targets, key, jnp.zeros(shuffle_parallel)), None, length=shuffle_length
    )  # [batch_size, shuffle_length, ...]
    moves = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), moves
    )  # [shuffle_length, batch_size, ...]
    move_costs = jnp.swapaxes(move_costs, 0, 1)  # [shuffle_length, batch_size]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [batch_size, shuffle_length, ...]
    solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), solve_configs
    )  # [batch_size * shuffle_length, ...]
    moves = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), moves
    )  # [batch_size * shuffle_length, ...]
    move_costs = jnp.reshape(move_costs, (-1))  # [batch_size * shuffle_length]
    return solve_configs, targets, moves, move_costs


def create_hindsight_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, key, move_cost_ = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost_), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])  # [action]
        idx = jax.vmap(lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1))(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, idx
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, idx)  # [batch]
        move_cost = move_cost_ + cost
        return (state, next_state, key, move_cost), (state, move_cost_)

    _, (moves, move_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, key, jnp.zeros(shuffle_parallel)),
        None,
        length=shuffle_length + 1,
    )  # [shuffle_length, batch_size, ...]
    targets = moves[-1, ...]
    moves = moves[:-1, ...]  # [shuffle_length, batch_size, ...]
    solve_configs = puzzle.batched_hindsight_transform(solve_configs, targets)  # [batch_size, ...]
    move_costs = move_costs[-1, ...] - move_costs[:-1, ...]  # [shuffle_length, batch_size]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length, batch_size, ...]

    solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), solve_configs
    )  # [batch_size * shuffle_length, ...]

    moves = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), moves
    )  # [batch_size * shuffle_length, ...]

    move_costs = jnp.reshape(move_costs, (-1))  # [batch_size * shuffle_length]
    return solve_configs, targets, moves, move_costs


def create_hindsight_target_triangular_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, key, move_cost_ = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost_), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])  # [action]
        idx = jax.vmap(lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1))(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, idx
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, idx)  # [batch]
        move_cost = move_cost_ + cost
        return (state, next_state, key, move_cost), (state, move_cost_)

    _, (moves, move_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, key, jnp.zeros(shuffle_parallel)),
        None,
        length=shuffle_length + 1,
    )  # [shuffle_length, batch_size, ...]
    solve_configs = jax.vmap(puzzle.batched_hindsight_transform)(moves)
    move_costs = move_costs[jnp.newaxis, ...] - move_costs[:, jnp.newaxis, ...]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length + 1, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )
    targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length + 1, 1) + (x.ndim - 1) * (1,)),
        moves,
    )
    moves = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length + 1) + (x.ndim - 1) * (1,)),
        moves,
    )

    # Create an explicit upper triangular mask
    upper_tri_mask = jnp.expand_dims(
        jnp.triu(jnp.ones((shuffle_length + 1, shuffle_length + 1)), k=1), axis=-1
    )
    # Combine with positive cost condition
    valid_indices = (move_costs > 0) & (upper_tri_mask > 0)

    idxs = jnp.where(
        valid_indices, size=(shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel)
    )
    solve_configs = solve_configs[idxs[0], idxs[1], idxs[2], ...]
    moves = moves[idxs[0], idxs[1], idxs[2], ...]
    move_costs = move_costs[idxs[0], idxs[1], idxs[2]]
    return solve_configs, targets, moves, move_costs
