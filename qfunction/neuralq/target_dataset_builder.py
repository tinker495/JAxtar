import math
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.annotate import MAX_GEN_DS_BATCH_SIZE
from train_util.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
    flatten_scanned_paths,
)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
) -> chex.Array:
    # Determine valid entries before sanitizing infinities
    mask = jnp.isfinite(q_values)
    q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)

    # Scale Q-values by temperature for softmax
    safe_temperature = jnp.maximum(temperature, 1e-8)
    scaled_q_values = -q_values / safe_temperature

    # Apply mask before softmax to avoid overflow
    masked_q_values = jnp.where(mask, scaled_q_values, -jnp.inf)
    probs = jax.nn.softmax(masked_q_values, axis=1)
    probs = jnp.where(mask, probs, 0.0)

    # Row-wise normalization with guard
    row_sum = jnp.sum(probs, axis=1, keepdims=True)
    probs = jnp.where(row_sum > 0.0, probs / row_sum, probs)

    # Calculate uniform probabilities
    valid_actions = jnp.sum(mask, axis=1, keepdims=True)
    uniform_valid = jnp.where(mask, 1.0 / jnp.maximum(valid_actions, 1.0), 0.0)

    action_size = q_values.shape[1]
    uniform_all = jnp.ones_like(probs) / jnp.maximum(action_size, 1)

    # Fallback if no valid actions in a row
    probs = jnp.where(valid_actions > 0, probs, uniform_all)

    # ε-greedy mixing and final guard renormalization
    probs = probs * (1.0 - epsilon) + uniform_valid * epsilon
    probs = probs / (jnp.sum(probs, axis=1, keepdims=True) + 1e-8)
    return probs


def _get_datasets_with_policy(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states = vals
        # Check if the current states are already in a solved configuration.
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        # Preprocess the states to be suitable for neural network input.
        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        # Get the Q-values Q(s,a) for all actions 'a' in the current state 's' using the online Q-network.
        q_values = q_model.apply(q_params, preproc, training=False)
        q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)
        # Get all possible neighbor states (s') and the costs c(s,a,s') to move to them.
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        cost = jnp.transpose(cost, (1, 0))
        valid_action_mask = jnp.isfinite(cost)
        has_valid_action = jnp.any(valid_action_mask, axis=1)
        # q_sum_cost = q_values + cost
        q_sum_cost = jnp.where(jnp.isfinite(cost), q_values, jnp.inf)

        # Select an action 'a' probabilistically using a Boltzmann (softmax) exploration policy.
        # Actions with lower Q-values (lower cost-to-go) are more likely to be chosen.
        # Epsilon-greedy exploration is also mixed in.
        probs = boltzmann_action_selection(q_sum_cost, temperature=temperature)
        probs = jnp.where(valid_action_mask, probs, 0.0)
        probs_sum = jnp.sum(probs, axis=1, keepdims=True)
        probs = jnp.where(probs_sum > 0.0, probs / (probs_sum + 1e-8), probs)
        uniform_all = jnp.ones_like(probs) / jnp.maximum(probs.shape[1], 1)
        probs = jnp.where(has_valid_action[:, jnp.newaxis], probs, uniform_all)

        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )
        batch_size = actions.shape[0]
        # Determine the next state (s') by applying the selected action 'a'.
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        # Get all possible actions (a') and their costs c(s',a',s'') from the next state (s').
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        neighbor_cost = jnp.transpose(neighbor_cost, (1, 0))  # [batch_size, action_size]
        neighbor_valid_mask = jnp.isfinite(neighbor_cost)
        has_valid_neighbor = jnp.any(neighbor_valid_mask, axis=1)
        # Check if the next state (s') is a solved state.
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )
        selected_neighbors_solved = jnp.logical_or(
            selected_neighbors_solved, jnp.logical_not(has_valid_action)
        )

        # Preprocess the next states (s') for neural network input.
        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        # --- Target Q-Value Calculation (Bellman Optimality Equation) ---
        # Use the target Q-network (with frozen parameters `target_q_params`)
        # to get the Q-values for the next state, Q_target(s', a').
        # Using a separate target network stabilizes training.
        q = q_model.apply(
            target_q_params, preproc_neighbors, training=False
        )  # [minibatch_size, action_shape]
        q = jnp.nan_to_num(q, posinf=1e6, neginf=-1e6)
        # Invalidate actions that are not reachable from the next state.
        valid_neighbor_cost = jnp.where(neighbor_valid_mask, neighbor_cost, jnp.inf)

        # Modified for Q(s,a) = c(s,a) + min_a'(Q(s',a'))
        q_next = jnp.where(neighbor_valid_mask, q, jnp.inf)

        if use_double_dqn:
            q_online = q_model.apply(q_params, preproc_neighbors, training=False)
            q_online = jnp.nan_to_num(q_online, posinf=1e6, neginf=-1e6)
            q_online = jnp.where(jnp.isfinite(valid_neighbor_cost), q_online, jnp.inf)
            best_actions = jnp.argmin(q_online, axis=1)
            min_next_q = jnp.take_along_axis(q_next, best_actions[:, jnp.newaxis], axis=1).squeeze(
                1
            )
        else:
            min_next_q = jnp.min(q_next, axis=1)

        # Ensure non-negative future cost
        min_next_q = jnp.maximum(min_next_q, 0.0)
        min_next_q = jnp.where(has_valid_neighbor, min_next_q, 0.0)

        # Base case: If the next state (s') is the solution, the future cost is 0.
        # Q(s,a) = c(s,a) + (0 if solved else min Q(s',a'))
        selected_cost = jnp.take_along_axis(cost, actions[:, jnp.newaxis], axis=1).squeeze(1)
        selected_cost = jnp.where(has_valid_action, selected_cost, 0.0)
        target_q = selected_cost + jnp.where(selected_neighbors_solved, 0.0, min_next_q)
        # If the current state (s) was already solved, its Q-value should also be 0.
        target_q = jnp.where(solved, 0.0, target_q)
        target_q = jnp.where(has_valid_action, target_q, 0.0)

        # if the puzzle is already solved, the all q is 0
        return key, (
            solve_configs,
            states,
            target_q,
            actions,
        )

    _, (solve_configs, states, target_q, actions,) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_q = target_q.reshape((-1, 1))
    actions = actions.reshape((-1, 1))
    cost = move_costs.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": actions,
        "cost": cost,
    }


def _compute_diffusion_q(
    solve_configs: chex.Array,
    states: chex.Array,
    trajectory_actions: chex.Array,
    move_costs: chex.Array,
    action_costs: chex.Array,
    parent_indices: chex.Array,
    SolveConfigsAndStatesAndActions: Xtructurable,
    SolveConfigsAndStates: Xtructurable,
    k_max: int,
):
    raw_move_costs = move_costs.reshape((-1,))
    solve_configs_and_states_and_actions = SolveConfigsAndStatesAndActions(
        solveconfigs=solve_configs,
        states=states,
        actions=trajectory_actions,
    )

    # 1. Find unique state-action pairs and their minimal move_costs
    # We use move_costs as the initial estimate for Q(s,a) / Cost(s,a)
    # For diffusion distance, Q(s,a) approximates the cost to go.
    target_q = raw_move_costs

    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=target_q,
        return_index=True,
        return_inverse=True,
    )
    num_unique_state_actions = unique_uint32eds_idx.shape[0]
    target_q = target_q[unique_uint32eds_idx][inverse_indices][:, jnp.newaxis]

    # 2. Find state-wise minimal cost (Global Best Value Table)
    # This helps sharing optimal cost across different actions for the same state
    solve_configs_and_states = SolveConfigsAndStates(
        solveconfigs=solve_configs,
        states=states,
    )

    _, unique_state_idx, inverse_state_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=target_q.reshape(-1),
        return_index=True,
        return_inverse=True,
    )
    num_unique_states = unique_state_idx.shape[0]

    def _collapse_duplicate_state_actions(q_vec: chex.Array) -> chex.Array:
        """Collapse duplicate (solve_config, state, action) rows to minimal Q."""
        group_min = (
            jnp.full((num_unique_state_actions, 1), jnp.inf, dtype=q_vec.dtype)
            .at[inverse_indices]
            .min(q_vec)
        )
        return group_min[inverse_indices]

    def _state_min_over_actions(q_vec: chex.Array) -> chex.Array:
        """Compute per-state min_a Q(s,a) from the (possibly duplicated) dataset rows."""
        group_min_state = (
            jnp.full((num_unique_states, 1), jnp.inf, dtype=q_vec.dtype)
            .at[inverse_state_indices]
            .min(q_vec)
        )
        return group_min_state

    # Propagate the improved Q values backwards along the trajectory
    dataset_size = target_q.shape[0]

    # Pad dataset with infinity to handle invalid parent pointers
    padded_q = jnp.pad(target_q, ((0, 1), (0, 0)), constant_values=jnp.inf)

    # Map -1 or out-of-bounds indices to the padded infinity value
    safe_parent_indices = jnp.where(
        (parent_indices < 0) | (parent_indices >= dataset_size), dataset_size, parent_indices
    )
    # Parent indices refer to dataset rows; map those rows to their parent-state group id.
    inverse_state_indices_padded = jnp.pad(
        inverse_state_indices, (0, 1), constant_values=num_unique_states
    )
    idx = jnp.arange(dataset_size, dtype=safe_parent_indices.dtype)
    valid_parent = safe_parent_indices[:dataset_size] != dataset_size
    parent_is_behind = (safe_parent_indices[:dataset_size] < idx) & valid_parent
    behind_ratio = jnp.sum(parent_is_behind).astype(jnp.float32) / jnp.maximum(
        jnp.sum(valid_parent).astype(jnp.float32), 1.0
    )
    default_use_parent_indexed_costs = behind_ratio > 0.5
    padded_action_costs = jnp.pad(action_costs, ((0, 1), (0, 0)), constant_values=0.0)
    padded_move_costs = jnp.pad(raw_move_costs, (0, 1), constant_values=0.0)

    parent_move_costs = padded_move_costs[safe_parent_indices][:, jnp.newaxis]
    parent_aligned_costs = padded_action_costs[safe_parent_indices]
    child_aligned_costs = action_costs
    err_child = jnp.abs(raw_move_costs[:, jnp.newaxis] - (parent_move_costs + child_aligned_costs))
    err_parent = jnp.abs(
        raw_move_costs[:, jnp.newaxis] - (parent_move_costs + parent_aligned_costs)
    )
    valid_parent_f = valid_parent.astype(raw_move_costs.dtype)[:, jnp.newaxis]
    denom = jnp.maximum(jnp.sum(valid_parent_f), 1.0)
    mean_err_child = jnp.sum(err_child * valid_parent_f) / denom
    mean_err_parent = jnp.sum(err_parent * valid_parent_f) / denom
    min_mean_err = jnp.minimum(mean_err_child, mean_err_parent)
    use_error_based = min_mean_err < 1e-3
    use_parent_indexed_costs = jax.lax.select(
        use_error_based,
        mean_err_parent < mean_err_child,
        default_use_parent_indexed_costs,
    )

    edge_costs = jax.lax.cond(
        use_parent_indexed_costs,
        lambda: padded_action_costs[safe_parent_indices],
        lambda: action_costs,
    )

    def body_fun(i, q):
        # q is padded [N+1, 1]
        current_q = _collapse_duplicate_state_actions(q[:dataset_size])
        state_min_cost = _state_min_over_actions(current_q)
        padded_state_min_cost = jnp.pad(state_min_cost, ((0, 1), (0, 0)), constant_values=jnp.inf)
        parent_state_group = inverse_state_indices_padded[safe_parent_indices]
        collapsed_padded_q = jnp.pad(current_q, ((0, 1), (0, 0)), constant_values=jnp.inf)

        # Gather Q from parents (neighbors closer to goal)
        # Combine global optimal info (s' best) and trajectory info
        q_parents_optimal = padded_state_min_cost[parent_state_group]  # [N, 1]
        q_parents_prop = collapsed_padded_q[safe_parent_indices]  # [N, 1]

        q_parents = jnp.minimum(q_parents_optimal, q_parents_prop)

        # Bellman update: Q(s, a) <= c(s, a, s') + Q(s', a')
        new_q = edge_costs + q_parents

        improved_q = jnp.minimum(current_q, new_q)
        return q.at[:dataset_size].set(improved_q)

    # Iterate k_max times to propagate along the longest possible path
    final_padded_q = jax.lax.fori_loop(0, k_max, body_fun, padded_q)

    return _collapse_duplicate_state_actions(final_padded_q[:dataset_size])


def _get_datasets_with_diffusion_distance(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStatesAndActions: Xtructurable,
    SolveConfigsAndStates: Xtructurable,
    q_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
):
    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]
    action_costs = shuffled_path["action_costs"].reshape((-1, 1))
    parent_indices = shuffled_path["parent_indices"]

    target_q = _compute_diffusion_q(
        solve_configs,
        states,
        trajectory_actions,
        move_costs,
        action_costs,
        parent_indices,
        SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates,
        k_max,
    )

    zeros = jnp.zeros_like(target_q)
    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": trajectory_actions,
        "cost": zeros,
    }


def _get_datasets_with_diffusion_distance_mixture(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStatesAndActions: Xtructurable,
    SolveConfigsAndStates: Xtructurable,
    q_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
):
    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    return_dict = _get_datasets_with_policy(
        puzzle,
        preproc_fn,
        q_model,
        minibatch_size,
        target_q_params,
        q_params,
        shuffled_path,
        key,
        temperature,
        use_double_dqn,
    )
    cost = return_dict["cost"]
    target_q = return_dict["target_q"]

    # Prepare for propagation
    action_costs = shuffled_path["action_costs"].reshape((-1, 1))
    parent_indices = shuffled_path["parent_indices"]
    solve_configs = return_dict["solveconfigs"]
    states = return_dict["states"]

    diffusion_q = _compute_diffusion_q(
        solve_configs,
        states,
        trajectory_actions,
        cost,  # Use cost (move_costs) as initial estimate
        action_costs,
        parent_indices,
        SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates,
        k_max,
    )

    target_q = jnp.concatenate(
        (target_q, diffusion_q),
        axis=1,
    )  # [dataset_size, 2]
    return_dict["target_q"] = target_q
    actions = return_dict["actions"]
    actions = jnp.concatenate(
        (actions, trajectory_actions),
        axis=1,
    )  # [dataset_size, 2]
    return_dict["actions"] = actions
    return return_dict


def get_qfunction_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: DistanceModel | DistanceHLGModel,
    dataset_size: int,
    k_max: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
    use_diffusion_distance: bool = False,
    use_diffusion_distance_mixture: bool = False,
    use_diffusion_distance_warmup: bool = False,
    diffusion_distance_warmup_steps: int = 0,
    non_backtracking_steps: int = 3,
):
    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    # Calculate optimal nn_minibatch_size
    # It must be <= MAX_GEN_DS_BATCH_SIZE and divide dataset_size
    n_batches = math.ceil(dataset_size / MAX_GEN_DS_BATCH_SIZE)
    while dataset_size % n_batches != 0:
        n_batches += 1
    nn_minibatch_size = dataset_size // n_batches

    # Calculate optimal shuffle_parallel and steps to respect MAX_GEN_DS_BATCH_SIZE
    max_shuffle_parallel = max(1, int(MAX_GEN_DS_BATCH_SIZE / k_max))
    needed_trajectories = math.ceil(dataset_size / k_max)
    shuffle_parallel = min(needed_trajectories, max_shuffle_parallel)
    steps = math.ceil(needed_trajectories / shuffle_parallel)

    if using_hindsight_target:
        assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"

        if using_triangular_sampling:
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                False,
                non_backtracking_steps=non_backtracking_steps,
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                False,
                non_backtracking_steps=non_backtracking_steps,
            )
    else:
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            k_max,
            shuffle_parallel,
            False,
            non_backtracking_steps=non_backtracking_steps,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    base_get_datasets = partial(
        _get_datasets_with_policy,
        puzzle,
        preproc_fn,
        q_model,
        nn_minibatch_size,
        temperature=temperature,
        use_double_dqn=use_double_dqn,
    )

    use_diffusion_features = use_diffusion_distance or use_diffusion_distance_mixture

    if use_diffusion_features:

        @xtructure_dataclass
        class SolveConfigsAndStatesAndActions:
            solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
            states: FieldDescriptor.scalar(dtype=puzzle.State)
            actions: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,))

        @xtructure_dataclass
        class SolveConfigsAndStates:
            solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
            states: FieldDescriptor.scalar(dtype=puzzle.State)

        if use_diffusion_distance_mixture:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance_mixture,
                puzzle,
                preproc_fn,
                SolveConfigsAndStatesAndActions,
                SolveConfigsAndStates,
                q_model,
                nn_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
                use_double_dqn=use_double_dqn,
            )
        else:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance,
                puzzle,
                preproc_fn,
                SolveConfigsAndStatesAndActions,
                SolveConfigsAndStates,
                q_model,
                nn_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
                use_double_dqn=use_double_dqn,
            )
    else:
        diffusion_get_datasets = base_get_datasets

    warmup_steps = max(int(diffusion_distance_warmup_steps), 0)
    warmup_enabled = use_diffusion_features and use_diffusion_distance_warmup and warmup_steps > 0

    def should_use_diffusion(step: int) -> bool:
        if not use_diffusion_features:
            return False
        if warmup_enabled:
            return step < warmup_steps
        return True

    def build_runner(dataset_extractor: Callable):
        @jax.jit
        def runner(
            target_q_params: Any,
            q_params: Any,
            key: chex.PRNGKey,
        ):
            def scan_fn(scan_key, _):
                scan_key, subkey = jax.random.split(scan_key)
                paths = jited_create_shuffled_path(subkey)
                return scan_key, paths

            key_inner, paths = jax.lax.scan(scan_fn, key, None, length=steps)
            paths = flatten_scanned_paths(paths, dataset_size)
            flatten_dataset = dataset_extractor(
                target_q_params,
                q_params,
                paths,
                key_inner,
            )
            return flatten_dataset

        return runner

    default_runner = build_runner(base_get_datasets)
    diffusion_runner = build_runner(diffusion_get_datasets)

    if n_devices > 1:
        pmap_default_runner = jax.pmap(default_runner, in_axes=(None, None, 0))
        pmap_diffusion_runner = jax.pmap(diffusion_runner, in_axes=(None, None, 0))

        def get_datasets(target_q_params, q_params, key, step: int):
            keys = jax.random.split(key, n_devices)
            runner = pmap_diffusion_runner if should_use_diffusion(step) else pmap_default_runner
            return runner(target_q_params, q_params, keys)

        return get_datasets

    def single_device_get_datasets(target_q_params, q_params, key, step: int):
        runner = diffusion_runner if should_use_diffusion(step) else default_runner
        return runner(target_q_params, q_params, key)

    return single_device_get_datasets
