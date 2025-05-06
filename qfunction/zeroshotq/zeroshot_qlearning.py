import math
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.puzzle_base import Puzzle
from qfunction.zeroshotq.zeroshotq_base import GoalProjector, ZeroshotQModelBase


def zeroshot_qlearning_builder(
    minibatch_size: int,
    goal_model: GoalProjector,
    zeroshotq_model: ZeroshotQModelBase,
    optimizer_q: optax.GradientTransformation,
    optimizer_goal: optax.GradientTransformation,
    lambda_reg: float,
    polyak_alpha: float,
    n_devices: int = 1,
):
    def zeroshotq_loss(
        q_params: jax.tree_util.PyTreeDef,
        target_q_params: jax.tree_util.PyTreeDef,
        goal_params: jax.tree_util.PyTreeDef,
        target_goal_params: jax.tree_util.PyTreeDef,
        solve_configs_i: chex.Array,
        states_i: chex.Array,
        actions_i: chex.Array,
        costs_i: chex.Array,
        states_j: chex.Array,
        actions_j: chex.Array,
        key: chex.PRNGKey,
    ):
        batch_size = states_i.shape[0]
        # Apply goal model to get representations and updates
        (solve_config_z, update_goal_1), (b_j, update_goal_2), (b_i, update_goal_3) = jax.vmap(
            lambda gp, s_conf, s_i, a_i, s_j, a_j: (
                goal_model.apply(
                    gp,
                    s_conf,
                    training=True,
                    method=goal_model.solve_config_projection,
                    mutable=["batch_stats"],
                ),
                goal_model.apply(
                    gp,
                    s_j,
                    a_j,
                    training=True,
                    method=goal_model.get_b,
                    mutable=["batch_stats"],
                ),
                goal_model.apply(
                    gp,
                    s_i,
                    a_i,
                    training=True,
                    method=goal_model.get_b,
                    mutable=["batch_stats"],
                ),
            ),
            in_axes=(None, 0, 0, 0, 0, 0),
            out_axes=0,
        )(
            goal_params,
            solve_configs_i,
            states_i,
            actions_i,
            states_j,
            actions_j,
        )  # Combine batch stats updates

        # Aggregate batch stats updates (simple mean for now, might need adjustment)
        goal_params["batch_stats"] = jax.tree_util.tree_map(
            lambda x, y, z: (x + y + z) / 3.0,
            update_goal_1["batch_stats"],
            update_goal_2["batch_stats"],
            update_goal_3["batch_stats"],
        )

        # Apply Q model to get forward projection and updates
        f_a_i, update_q = zeroshotq_model.apply(
            q_params,
            states_i,
            solve_config_z,
            training=True,
            method=zeroshotq_model.forward_projection,
            mutable=["batch_stats"],
        )  # [batch_size, action_size, latent_dim]
        q_params["batch_stats"] = update_q["batch_stats"]

        # Calculate Q-values (distances)
        q_ij = zeroshotq_model.apply(
            q_params,
            f_a_i,
            b_j,
            method=zeroshotq_model.distance,
        )  # [batch_size, action_size]
        # q_ij = jnp.squeeze(q_ij, axis=2) # [batch_size, action_size]
        # Squeeze might not be needed depending on distance implementation
        q_i = jnp.take_along_axis(q_ij, actions_i[:, jnp.newaxis], axis=1)  # [batch_size, 1]
        q_i = jnp.squeeze(q_i, axis=1)  # [batch_size]

        # Target network calculations
        target_solve_config_z, _ = goal_model.apply(
            target_goal_params,
            solve_configs_i,
            training=False,
            method=goal_model.solve_config_projection,
            mutable=["batch_stats"],
        )  # [batch_size, latent_dim]

        target_b_j, _ = goal_model.apply(
            target_goal_params,
            states_j,
            actions_j,
            training=False,
            method=goal_model.get_b,
            mutable=["batch_stats"],
        )  # [batch_size, latent_dim]

        target_f_a_i, _ = zeroshotq_model.apply(
            target_q_params,
            states_i,
            target_solve_config_z,  # Use target solve config projection
            training=False,
            method=zeroshotq_model.forward_projection,
            mutable=["batch_stats"],
        )  # [batch_size, action_size, latent_dim]

        target_q_ij = zeroshotq_model.apply(
            target_q_params,
            target_f_a_i,
            target_b_j,
            method=zeroshotq_model.distance,
        )  # [batch_size, action_size]
        # target_q_ij = jnp.squeeze(target_q_ij, axis=2) # [batch_size, action_size] # Squeeze might not be needed

        target_pi_ij = boltzmann_action_selection(target_q_ij)  # [batch_size, action_size]
        v_ij = jnp.sum(target_pi_ij * target_q_ij, axis=1)  # [batch_size]
        target_q_i = v_ij + costs_i  # [batch_size]
        target_q_i = jax.lax.stop_gradient(target_q_i)  # Stop gradient for target

        # --- MSE Loss ---
        mse_loss = jnp.mean(jnp.square(q_i - target_q_i))

        # --- Orthonormality Regularization ---
        # L_reg(ω) = (1/b^2) Σ_{i,j∈I^2, i!=j} ( Bω(si, ai)^T stop_gradient(Bω(s'j, a'j)) )^2
        #             - (1/b) Σ_{i∈I} ( Bω(si, ai)^T stop_gradient(Bω(si, ai)) )
        # Note: The pseudocode sums over i,j in I^2. The term below implements this.
        # If you need sampling like in the pseudocode (sampling s_i,a_i and s'_j,a'_j),
        # the dataset creation needs modification. This assumes i and j index the same minibatch.
        term1 = jnp.einsum("bi,bj->ij", b_i, jax.lax.stop_gradient(b_j))  # (batch_size, batch_size)
        term1 = jnp.square(term1)
        # Exclude diagonal elements i=j for the first summation
        term1 = term1 * (1.0 - jnp.eye(batch_size))
        reg_loss_term1 = jnp.sum(term1) / (
            batch_size * (batch_size - 1) if batch_size > 1 else 1
        )  # Normalize by b*(b-1) or 1 if b=1

        term2 = jnp.einsum("bi,bi->b", b_i, jax.lax.stop_gradient(b_i))  # (batch_size,)
        reg_loss_term2 = jnp.mean(term2)  # Average over batch

        reg_loss = reg_loss_term1 - reg_loss_term2

        # Combine losses
        total_loss = mse_loss + lambda_reg * reg_loss

        return total_loss, (mse_loss, reg_loss, q_params, goal_params)

    # Use value_and_grad for combined loss, getting grads for both param sets
    grad_fn = jax.value_and_grad(
        zeroshotq_loss, argnums=(0, 2), has_aux=True
    )  # Grad w.r.t q_params (0) and goal_params (2)

    def qlearning(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: jax.tree_util.PyTreeDef,
        target_q_params: jax.tree_util.PyTreeDef,
        goal_params: jax.tree_util.PyTreeDef,
        target_goal_params: jax.tree_util.PyTreeDef,
        opt_state_q: optax.OptState,
        opt_state_goal: optax.OptState,
    ):
        """
        Q-learning with orthonormality regularization and goal parameter updates.
        """
        solve_configs = dataset["solve_configs"]
        states = dataset["states"]
        actions = dataset["actions"]
        costs = dataset["cost"]
        data_size = actions.shape[0]
        n_minibatches = math.ceil(data_size / minibatch_size)  # Corrected variable name

        key_i, key_j, key = jax.random.split(key, 3)

        # Ensure batch indices cover the entire dataset potentially multiple times if needed
        total_indices_needed = n_minibatches * minibatch_size
        permutations_needed = math.ceil(total_indices_needed / data_size)

        batch_indexs_i = jnp.array([])
        for k in range(permutations_needed):
            key_i, subkey_i = jax.random.split(key_i)
            batch_indexs_i = jnp.concatenate(
                [batch_indexs_i, jax.random.permutation(subkey_i, jnp.arange(data_size))]
            )
        batch_indexs_i = batch_indexs_i[:total_indices_needed].astype(jnp.int32)
        batch_indexs_i = jnp.reshape(batch_indexs_i, (n_minibatches, minibatch_size))

        batch_indexs_j = jnp.array([])
        for k in range(permutations_needed):
            key_j, subkey_j = jax.random.split(key_j)
            batch_indexs_j = jnp.concatenate(
                [batch_indexs_j, jax.random.permutation(subkey_j, jnp.arange(data_size))]
            )
        batch_indexs_j = batch_indexs_j[:total_indices_needed].astype(jnp.int32)
        batch_indexs_j = jnp.reshape(batch_indexs_j, (n_minibatches, minibatch_size))

        batched_solve_configs_i = jnp.take(solve_configs, batch_indexs_i, axis=0)
        batched_states_i = jnp.take(states, batch_indexs_i, axis=0)
        batched_actions_i = jnp.take(actions, batch_indexs_i, axis=0)
        batched_costs_i = jnp.take(costs, batch_indexs_i, axis=0)

        batched_states_j = jnp.take(states, batch_indexs_j, axis=0)
        batched_actions_j = jnp.take(actions, batch_indexs_j, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, goal_params, opt_state_q, opt_state_goal, key = carry
            key, subkey = jax.random.split(key)
            (
                solve_configs_i,
                states_i,
                states_j,
                actions_i,
                actions_j,
                costs_i,
            ) = batched_dataset

            # Calculate loss and gradients for both q_params and goal_params
            (total_loss, (mse_loss, reg_loss, q_params_updated, goal_params_updated)), (
                q_grads,
                goal_grads,
            ) = grad_fn(
                q_params,
                target_q_params,
                goal_params,
                target_goal_params,
                solve_configs_i,
                states_i,
                actions_i,
                costs_i,
                states_j,
                actions_j,
                subkey,
            )

            # Update batch stats from aux output
            q_params = q_params_updated
            goal_params = goal_params_updated

            if n_devices > 1:
                q_grads = jax.lax.psum(q_grads, axis_name="devices")
                goal_grads = jax.lax.psum(goal_grads, axis_name="devices")

            # Update Q parameters
            updates_q, opt_state_q = optimizer_q.update(q_grads, opt_state_q, params=q_params)
            q_params = optax.apply_updates(q_params, updates_q)

            # Update Goal parameters
            updates_goal, opt_state_goal = optimizer_goal.update(
                goal_grads, opt_state_goal, params=goal_params
            )
            goal_params = optax.apply_updates(goal_params, updates_goal)

            # Calculate gradient magnitude mean for monitoring
            q_grad_leaves = (
                jax.tree_util.tree_leaves(q_grads["params"]) if "params" in q_grads else []
            )
            goal_grad_leaves = (
                jax.tree_util.tree_leaves(goal_grads["params"]) if "params" in goal_grads else []
            )

            q_grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), q_grad_leaves
            )
            goal_grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), goal_grad_leaves
            )

            all_grads_flat = jnp.concatenate(q_grad_magnitude + goal_grad_magnitude)
            grad_magnitude_mean = jnp.mean(all_grads_flat) if all_grads_flat.size > 0 else 0.0

            return (q_params, goal_params, opt_state_q, opt_state_goal, key), (
                total_loss,
                mse_loss,
                reg_loss,
                grad_magnitude_mean,
            )

        # Scan over minibatches
        (q_params, goal_params, opt_state_q, opt_state_goal, key), (
            total_losses,
            mse_losses,
            reg_losses,
            grad_magnitude_means,
        ) = jax.lax.scan(
            train_loop,
            (q_params, goal_params, opt_state_q, opt_state_goal, key),
            (
                batched_solve_configs_i,
                batched_states_i,
                batched_states_j,
                batched_actions_i,
                batched_actions_j,
                batched_costs_i,
            ),
        )
        # --- Polyak Averaging ---
        target_q_params = jax.tree_util.tree_map(
            lambda target, online: target * polyak_alpha + online * (1 - polyak_alpha),
            target_q_params,
            q_params,
        )
        target_goal_params = jax.tree_util.tree_map(
            lambda target, online: target * polyak_alpha + online * (1 - polyak_alpha),
            target_goal_params,
            goal_params,
        )

        # Calculate final metrics
        total_loss_mean = jnp.mean(total_losses)
        mse_loss_mean = jnp.mean(mse_losses)
        reg_loss_mean = jnp.mean(reg_losses)
        grad_magnitude_mean = jnp.mean(grad_magnitude_means)

        # Calculate weights magnitude means for monitoring
        q_weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))),
            jax.tree_util.tree_leaves(q_params["params"]) if "params" in q_params else [],
        )
        goal_weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))),
            jax.tree_util.tree_leaves(goal_params["params"]) if "params" in goal_params else [],
        )
        all_weights_flat = jnp.concatenate(q_weights_magnitude + goal_weights_magnitude)
        weights_magnitude_mean = jnp.mean(all_weights_flat) if all_weights_flat.size > 0 else 0.0

        return (
            q_params,
            target_q_params,
            goal_params,
            target_goal_params,
            opt_state_q,
            opt_state_goal,
            total_loss_mean,
            mse_loss_mean,
            reg_loss_mean,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    if n_devices > 1:
        # Define the pmap function correctly
        @partial(jax.pmap, axis_name="devices", in_axes=(0, 0, None, None, None, None, None, None))
        def pmapped_qlearning_step(
            key,
            dataset_shard,
            q_params,
            target_q_params,
            goal_params,
            target_goal_params,
            opt_state_q,
            opt_state_goal,
        ):
            return qlearning(
                key,
                dataset_shard,
                q_params,
                target_q_params,
                goal_params,
                target_goal_params,
                opt_state_q,
                opt_state_goal,
            )

        def pmap_qlearning(
            key,
            sharded_dataset,  # Expect pre-sharded dataset
            q_params,
            target_q_params,
            goal_params,
            target_goal_params,
            opt_state_q,
            opt_state_goal,
        ):
            keys = jax.random.split(key, n_devices)
            (
                q_params_out,
                target_q_params_out,
                goal_params_out,
                target_goal_params_out,
                opt_state_q_out,
                opt_state_goal_out,
                total_loss,
                mse_loss,
                reg_loss,
                grad_magnitude,
                weight_magnitude,
            ) = pmapped_qlearning_step(
                keys,
                sharded_dataset,
                q_params,
                target_q_params,
                goal_params,
                target_goal_params,
                opt_state_q,
                opt_state_goal,
            )
            # Take the first replica's state for non-reduced outputs
            q_params = jax.tree_util.tree_map(lambda x: x[0], q_params_out)
            target_q_params = jax.tree_util.tree_map(lambda x: x[0], target_q_params_out)
            goal_params = jax.tree_util.tree_map(lambda x: x[0], goal_params_out)
            target_goal_params = jax.tree_util.tree_map(lambda x: x[0], target_goal_params_out)
            opt_state_q = jax.tree_util.tree_map(lambda x: x[0], opt_state_q_out)
            opt_state_goal = jax.tree_util.tree_map(lambda x: x[0], opt_state_goal_out)
            # Average the metrics across devices
            total_loss = jnp.mean(total_loss)
            mse_loss = jnp.mean(mse_loss)
            reg_loss = jnp.mean(reg_loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)

            return (
                q_params,
                target_q_params,
                goal_params,
                target_goal_params,
                opt_state_q,
                opt_state_goal,
                total_loss,
                mse_loss,
                reg_loss,
                grad_magnitude,
                weight_magnitude,
            )

        return pmap_qlearning
    else:
        # Jit the single-device version
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
    goal_model: GoalProjector,
    zeroshotq_model: ZeroshotQModelBase,
    minibatch_size: int,
    goal_params: jax.tree_util.PyTreeDef,
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

        solve_config_z = goal_model.apply(
            goal_params,
            solve_config_preproc,
            method=goal_model.solve_config_projection,
        )
        f_a = zeroshotq_model.apply(
            q_params,
            state_preproc,
            solve_config_z,
            method=zeroshotq_model.forward_projection,
        )
        q_values = zeroshotq_model.apply(
            q_params,
            f_a,
            solve_config_z,
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
        # Transpose cost to shape [batch_size, action_size] before taking actions
        cost = jnp.transpose(cost, (1, 0))  # [batch_size, action_size]
        # Now select costs for chosen actions
        cost = jnp.take_along_axis(cost, actions[:, jnp.newaxis], axis=1)  # [batch_size, 1]
        cost = jnp.squeeze(cost, axis=1)  # [batch_size]

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
    goal_model: GoalProjector,
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
            goal_model,
            zeroshotq_model,
            dataset_minibatch_size,
        )
    )

    @jax.jit
    def get_datasets(
        q_params: jax.tree_util.PyTreeDef,
        goal_params: jax.tree_util.PyTreeDef,
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

        flatten_dataset = jited_get_datasets(q_params, goal_params, paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(q_params, goal_params, key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(None, None, 0))(q_params, goal_params, keys)
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
