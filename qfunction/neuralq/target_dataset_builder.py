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
    calculate_dataset_params,
    compute_diffusion_targets,
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
    wrap_dataset_runner,
)
from train_util.util import boltzmann_action_selection


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
    path_actions = shuffled_path["actions"]
    trajectory_indices = shuffled_path["trajectory_indices"]
    step_indices = shuffled_path["step_indices"]

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
    path_actions = path_actions.reshape((-1,))
    trajectory_indices = trajectory_indices.reshape((-1,))
    step_indices = step_indices.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": actions,
        "cost": cost,
        "path_actions": path_actions,
        "trajectory_indices": trajectory_indices,
        "step_indices": step_indices,
    }


def _compute_diffusion_q(
    solve_configs: chex.Array,
    states: chex.Array,
    is_solved: chex.Array,
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

    # 1. Unique state-action pairs
    _, unique_state_action_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=raw_move_costs,
        return_index=True,
        return_inverse=True,
    )
    num_unique_state_actions = unique_state_action_idx.shape[0]

    # 2. Unique states (for state-min-over-actions propagation)
    solve_configs_and_states = SolveConfigsAndStates(
        solveconfigs=solve_configs,
        states=states,
    )
    _, unique_state_idx, inverse_state_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=raw_move_costs,
        return_index=True,
        return_inverse=True,
    )
    num_unique_states = unique_state_idx.shape[0]

    # 3. Compute diffusion using common utility
    return compute_diffusion_targets(
        initial_values=raw_move_costs[:, jnp.newaxis],
        is_solved=is_solved,
        parent_indices=parent_indices,
        action_costs=action_costs,
        raw_move_costs=raw_move_costs,
        k_max=k_max,
        inverse_indices=inverse_indices,
        num_unique=num_unique_state_actions,
        inverse_state_indices=inverse_state_indices,
        num_unique_states=num_unique_states,
    )


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
    trajectory_indices = shuffled_path["trajectory_indices"].reshape((-1,))
    step_indices = shuffled_path["step_indices"].reshape((-1,))

    # Flatten strictly for compute_diffusion logic inside
    solve_configs_flat = solve_configs.reshape((-1,))
    states_flat = states.reshape((-1,))

    is_solved = puzzle.batched_is_solved(
        solve_configs_flat, states_flat, multi_solve_config=True
    ).reshape((-1,))

    target_q = _compute_diffusion_q(
        solve_configs_flat,
        states_flat,
        is_solved,
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
        "solveconfigs": solve_configs_flat,
        "states": states_flat,
        "target_q": target_q,
        "actions": trajectory_actions,
        "cost": zeros,
        "path_actions": trajectory_actions,
        "trajectory_indices": trajectory_indices,
        "step_indices": step_indices,
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
    path_actions = shuffled_path["actions"].reshape((-1,))
    trajectory_indices = shuffled_path["trajectory_indices"].reshape((-1,))
    step_indices = shuffled_path["step_indices"].reshape((-1,))
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

    # Already flattened in return_dict of _get_datasets_with_policy?
    # Yes, solve_configs.reshape((-1,)) at end of _get_datasets_with_policy

    is_solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True).reshape(
        (-1,)
    )

    diffusion_q = _compute_diffusion_q(
        solve_configs,
        states,
        is_solved,
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
    return_dict["path_actions"] = path_actions
    return_dict["trajectory_indices"] = trajectory_indices
    return_dict["step_indices"] = step_indices
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

    # Calculate optimal parameters for dataset generation
    nn_minibatch_size, shuffle_parallel, steps = calculate_dataset_params(
        dataset_size, k_max, MAX_GEN_DS_BATCH_SIZE
    )

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

    return wrap_dataset_runner(
        dataset_size=dataset_size,
        steps=steps,
        jited_create_shuffled_path=jited_create_shuffled_path,
        base_get_datasets=base_get_datasets,
        diffusion_get_datasets=diffusion_get_datasets,
        should_use_diffusion_fn=should_use_diffusion,
        n_devices=n_devices,
    )
