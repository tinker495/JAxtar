import math
from functools import partial
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.test_util
import optax

from puzzle.puzzle_base import Puzzle


def qlearning_builder(
    minibatch_size: int,
    q_model: nn.Module,
    optimizer: optax.GradientTransformation,
):
    def qlearning_loss(
        q_params: jax.tree_util.PyTreeDef,
        preprocessed_solve_configs: chex.Array,
        preprocessed_states: chex.Array,
        random_neighbors: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
    ):
        q_values, variable_updates = q_model.apply(
            q_params,
            preprocessed_solve_configs,
            preprocessed_states,
            training=True,
            mutable=["batch_stats"],
            method=q_model.solve_config_distance,
        )
        q_params["batch_stats"] = variable_updates["batch_stats"]

        (cos_similarity1, cos_similarity2), variable_updates = q_model.apply(
            q_params,
            preprocessed_states,
            random_neighbors,
            training=True,
            mutable=["batch_stats"],
            method=q_model.state_similarity,
        )

        cos_similarity = (cos_similarity1 + cos_similarity2) / 2
        similarity_loss = jnp.mean(1 - cos_similarity)

        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        mse_loss = jnp.mean(jnp.square(diff))
        # loss = jnp.mean(hubberloss(diff, delta=0.1) / 0.1 * weights)
        loss = mse_loss + similarity_loss * 0.01
        return loss, (q_params, mse_loss, similarity_loss, diff, q_values_at_actions)

    def qlearning(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array, chex.Array],
        q_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        (
            preprocessed_solve_configs,
            preprocessed_states,
            random_neighbors,
            target_q,
            actions,
        ) = dataset
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

        batched_preprocessed_solve_configs = jnp.take(
            preprocessed_solve_configs, batch_indexs, axis=0
        )
        batched_preprocessed_states = jnp.take(preprocessed_states, batch_indexs, axis=0)
        batched_random_neighbors = jnp.take(random_neighbors, batch_indexs, axis=0)
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            (
                preprocessed_solve_configs,
                preprocessed_states,
                random_neighbors,
                target_q,
                actions,
            ) = batched_dataset
            (
                loss,
                (q_params, mse_loss, similarity_loss, diff, q_values_at_actions),
            ), grads = jax.value_and_grad(qlearning_loss, has_aux=True)(
                q_params,
                preprocessed_solve_configs,
                preprocessed_states,
                random_neighbors,
                actions,
                target_q,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude))
            return (q_params, opt_state), (
                loss,
                mse_loss,
                similarity_loss,
                diff,
                q_values_at_actions,
                grad_magnitude_mean,
            )

        (q_params, opt_state), (
            losses,
            mse_losses,
            similarity_losses,
            diffs,
            q_values_at_actions,
            grad_magnitude_means,
        ) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (
                batched_preprocessed_solve_configs,
                batched_preprocessed_states,
                batched_random_neighbors,
                batched_target_q,
                batched_actions,
            ),
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        mean_mse_loss = jnp.mean(mse_losses)
        mean_similarity_loss = jnp.mean(similarity_losses)
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude_means))
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(q_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.array(weights_magnitude))
        return (
            q_params,
            opt_state,
            loss,
            mean_abs_diff,
            mean_mse_loss,
            mean_similarity_loss,
            diffs,
            q_values_at_actions,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    return jax.jit(qlearning)


def boltzmann_action_selection(q_values: chex.Array, temperature: float = 1 / 3.0) -> chex.Array:
    q_values = -q_values / temperature
    q_values = jnp.exp(q_values)
    probs = q_values / jnp.sum(q_values, axis=1, keepdims=True)
    return probs


def _get_datasets(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    q_model: nn.Module,
    minibatch_size: int,
    target_q_params: jax.tree_util.PyTreeDef,
    q_params: jax.tree_util.PyTreeDef,
    shuffled_path: tuple[Puzzle.SolveConfig, Puzzle.State, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs, shuffled_path, move_costs = shuffled_path

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
        preprocessed_solve_configs = jax.vmap(solve_config_preproc_fn)(
            solve_configs
        )  # [batch_size, ...]
        preprocessed_states = jax.vmap(state_preproc_fn)(shuffled_path)  # [batch_size, ...]
        solved = puzzle.batched_is_solved(
            solve_configs, shuffled_path, multi_solve_config=True
        )  # [batch_size]

        q_values, _ = q_model.apply(
            q_params,
            preprocessed_solve_configs,
            preprocessed_states,
            training=False,
            mutable=["batch_stats"],
            method=q_model.solve_config_distance,
        )
        probs = boltzmann_action_selection(q_values)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )
        random_neighbor_idx = jax.random.randint(
            key, (minibatch_size,), 0, q_values.shape[1]
        )  # [batch_size]

        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, shuffled_path, filleds=jnp.ones_like(move_costs), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        random_neighbors = jax.tree_util.tree_map(
            lambda x: x[random_neighbor_idx, jnp.arange(batch_size), :],
            neighbors,
        )  # [batch_size, ...]
        selected_costs = jnp.take_along_axis(cost, actions[jnp.newaxis, :], axis=0).squeeze(0)
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )
        preproc_neighbors = jax.vmap(state_preproc_fn)(selected_neighbors)
        preproc_random_neighbors = jax.vmap(state_preproc_fn)(random_neighbors)

        q, _ = q_model.apply(
            target_q_params,
            preprocessed_solve_configs,
            preproc_neighbors,
            training=False,
            mutable=["batch_stats"],
            method=q_model.solve_config_distance,
        )  # [minibatch_size, action_shape]
        target_q = jnp.maximum(jnp.min(q, axis=1), 0.0) + selected_costs
        solved = jnp.logical_or(selected_neighbors_solved, solved)
        target_q = jnp.where(solved, 0.0, target_q)
        # if the puzzle is already solved, the all q is 0
        return key, (
            preprocessed_solve_configs,
            preprocessed_states,
            preproc_random_neighbors,
            target_q,
            actions,
        )

    _, (
        preprocessed_solve_configs,
        preprocessed_states,
        random_neighbors,
        target_q,
        actions,
    ) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_shuffled_path, minibatched_move_costs),
    )

    preprocessed_solve_configs = preprocessed_solve_configs.reshape(
        (-1, *preprocessed_solve_configs.shape[2:])
    )
    preprocessed_states = preprocessed_states.reshape((-1, *preprocessed_states.shape[2:]))
    random_neighbors = random_neighbors.reshape((-1, *random_neighbors.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))

    return preprocessed_solve_configs, preprocessed_states, random_neighbors, target_q, actions


def get_qlearning_dataset_builder(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    q_model: nn.Module,
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

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            solve_config_preproc_fn,
            state_preproc_fn,
            q_model,
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
        old_state, state, key, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
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
        move_cost = move_cost + cost
        return (state, next_state, key, move_cost), (next_state, move_cost)

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
    return solve_configs, moves, move_costs


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
        old_state, state, key, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
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
        move_cost = move_cost + cost
        return (state, next_state, key, move_cost), (next_state, move_cost)

    _, (moves, move_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, key, jnp.zeros(shuffle_parallel)),
        None,
        length=shuffle_length + 1,
    )  # [shuffle_length, batch_size, ...]
    solve_configs = puzzle.batched_hindsight_transform(moves[-1, ...])  # [batch_size, ...]
    moves = moves[:-1, ...]  # [shuffle_length, batch_size, ...]
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
    return solve_configs, moves, move_costs


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
        old_state, state, key, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
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
        move_cost = move_cost + cost
        return (state, next_state, key, move_cost), (next_state, move_cost)

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
    return solve_configs, moves, move_costs
