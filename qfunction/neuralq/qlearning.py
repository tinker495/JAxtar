import math
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.puzzle_base import Puzzle


def cosine_similarity_loss(a: chex.Array, b: chex.Array) -> chex.Array:
    a_norm = jnp.linalg.norm(a, axis=1)
    b_norm = jnp.linalg.norm(b, axis=1)
    return 1.0 - jnp.sum(a * b, axis=1) / (a_norm * b_norm)


def qlearning_builder(
    minibatch_size: int,
    q_train_info_fn: Callable,
    optimizer: optax.GradientTransformation,
):
    def qlearning_loss(
        q_params: jax.tree_util.PyTreeDef,
        solve_configs: chex.Array,
        shuffled_paths: chex.Array,
        preproc_neighbors: chex.Array,
        target_qs: chex.Array,
        actions: chex.Array,
        weights: chex.Array,
    ):
        (q_values, state_predict, next_state_project), variable_updates = q_train_info_fn(
            q_params, solve_configs, shuffled_paths, preproc_neighbors
        )
        new_params = {"params": q_params["params"], "batch_stats": variable_updates["batch_stats"]}
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        se = jnp.square(diff)
        similarity_loss = cosine_similarity_loss(
            state_predict, jax.lax.stop_gradient(next_state_project)
        )
        loss = jnp.mean(se * weights + similarity_loss)
        return loss, (new_params, jnp.mean(se), jnp.mean(similarity_loss), diff)

    def qlearning(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array, chex.Array],
        q_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        solve_configs, shuffled_paths, preproc_neighbors, target_qs, actions, weights = dataset
        data_size = target_qs.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_solve_configs = jnp.take(solve_configs, batch_indexs, axis=0)
        batched_shuffled_paths = jnp.take(shuffled_paths, batch_indexs, axis=0)
        batched_preproc_neighbors = jnp.take(preproc_neighbors, batch_indexs, axis=0)
        batched_target_qs = jnp.take(target_qs, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        batched_weights = jnp.take(weights, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            (
                solve_configs,
                shuffled_paths,
                preproc_neighbors,
                target_qs,
                actions,
                weights,
            ) = batched_dataset
            (loss, (q_params, mse_loss, similarity_loss, diff)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                q_params,
                solve_configs,
                shuffled_paths,
                preproc_neighbors,
                target_qs,
                actions,
                weights,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            return (q_params, opt_state), (loss, mse_loss, similarity_loss, diff)

        (q_params, opt_state), (losses, mse_losses, similarity_losses, diffs) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (
                batched_solve_configs,
                batched_shuffled_paths,
                batched_preproc_neighbors,
                batched_target_qs,
                batched_actions,
                batched_weights,
            ),
        )
        loss = jnp.mean(losses)
        mse_loss = jnp.mean(mse_losses)
        similarity_loss = jnp.mean(similarity_losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        return q_params, opt_state, loss, mse_loss, similarity_loss, mean_abs_diff, diffs

    return jax.jit(qlearning)


def boltzmann_action_selection(q_values: chex.Array, temperature: float = 3.0) -> chex.Array:
    q_values = -q_values / temperature
    q_values = jnp.exp(q_values)
    probs = q_values / jnp.sum(q_values, axis=1, keepdims=True)
    return probs


def _get_datasets(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    q_fn: Callable,
    minibatch_size: int,
    weights_lambda: float,
    use_kde: bool,
    kde_bandwidth: float,
    target_q_params: jax.tree_util.PyTreeDef,
    q_params: jax.tree_util.PyTreeDef,
    infos: tuple[Puzzle.SolveConfig, Puzzle.State, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs, shuffled_paths, move_costs = infos
    preprocessed_solve_configs = jax.vmap(solve_config_preproc_fn)(solve_configs)
    preprocessed_shuffled_paths = jax.vmap(state_preproc_fn)(shuffled_paths)

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_shuffled_paths = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), shuffled_paths
    )
    minibatched_preprocessed_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), preprocessed_solve_configs
    )
    minibatched_preprocessed_shuffled_paths = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), preprocessed_shuffled_paths
    )

    def get_minibatched_datasets(_, vals):
        solve_config, shuffled_path, preprocessed_solve_config, preprocessed_shuffled_path = vals
        solved = puzzle.batched_is_solved(
            solve_config, shuffled_path, multi_solve_config=True
        )  # [batch_size]

        q_values, _ = q_fn(q_params, preprocessed_solve_config, preprocessed_shuffled_path)
        probs = boltzmann_action_selection(q_values)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(key, q_values.shape[0]), probs
        )

        neighbors, cost = puzzle.batched_get_neighbours(
            solve_config,
            shuffled_path,
            filleds=jnp.ones((minibatch_size,)),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(minibatch_size), :],
            neighbors,
        )
        selected_costs = jnp.take_along_axis(cost, actions[jnp.newaxis, :], axis=0).squeeze()
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_config, selected_neighbors, multi_solve_config=True
        )

        preproc_neighbors = jax.vmap(state_preproc_fn)(selected_neighbors)

        q, _ = q_fn(
            target_q_params, preprocessed_solve_config, preproc_neighbors
        )  # [minibatch_size, action_shape]
        target_q = jnp.maximum(jnp.min(q, axis=1), 0.0) + selected_costs
        solved = jnp.logical_or(selected_neighbors_solved, solved)
        target_q = jnp.where(solved, 0.0, target_q)
        # if the puzzle is already solved, the all q is 0

        return None, (preproc_neighbors, target_q, actions)

    _, (preproc_neighbors, target_q, actions) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (
            minibatched_solve_configs,
            minibatched_shuffled_paths,
            minibatched_preprocessed_solve_configs,
            minibatched_preprocessed_shuffled_paths,
        ),
    )

    preproc_neighbors = preproc_neighbors.reshape((-1, *preproc_neighbors.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    move_costs = move_costs.reshape((-1, *move_costs.shape[2:]))

    # less cost, means more confident
    weights = (weights_lambda + 1.0) / (move_costs + weights_lambda)

    if use_kde:
        # Alternative method using a simplified KDE-based approach
        # Compute pairwise distances between target_q values
        # If target_q is multi-dimensional, you might need to flatten it first
        target_q_flat = target_q.reshape(target_q.shape[0], -1)

        # Use a simple density estimation
        # For each point, compute its "density" based on its distance to other points

        @jax.vmap
        def compute_density(q_value):
            # Calculate distances to all other points
            distances = jnp.sum((target_q_flat - q_value) ** 2, axis=1) ** 0.5
            # Apply Gaussian kernel
            density = jnp.mean(jnp.exp(-0.5 * (distances / kde_bandwidth) ** 2))
            return density

        densities = compute_density(target_q_flat)

        # Inverse weighting - higher density means lower weight
        epsilon = 1e-8  # Avoid division by zero
        distribution_weights = 1.0 / (densities + epsilon)

        # Combine with existing weights
        weights = weights * distribution_weights

    weights = weights / jnp.mean(weights)  # normalize weights to have mean 1.0
    return (
        preprocessed_solve_configs,
        preprocessed_shuffled_paths,
        preproc_neighbors,
        target_q,
        actions,
        weights,
    )


def get_qlearning_dataset_builder(
    puzzle: Puzzle,
    solve_config_preproc_fn: Callable,
    state_preproc_fn: Callable,
    q_fn: Callable,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_target: bool = False,
    weights_ratio: float = 100.0,
    use_kde: bool = True,
    kde_bandwidth: float = 2.0,
):
    weights_lambda = shuffle_length / max(weights_ratio, 1e-5)

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
            q_fn,
            dataset_minibatch_size,
            weights_lambda,
            use_kde,
            kde_bandwidth,
        )
    )

    def get_datasets(
        target_q_params: jax.tree_util.PyTreeDef,
        q_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        paths = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            paths.append(jited_create_shuffled_path(subkey))
        paths = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *paths)
        paths = jax.tree_util.tree_map(lambda x: x[:dataset_size], paths)

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
