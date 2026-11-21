from functools import partial
from typing import Annotated, Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.sampling import (
    compute_diffusion_targets,
    make_diffusion_step_selector,
    prepare_shuffled_path_sampling,
    wrap_dataset_runner,
)
from train_util.trajectory_dataset_adapter import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from train_util.util import boltzmann_action_selection


def _get_datasets_with_policy(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStatesAndActions: type[Xtructurable],
    SolveConfigsAndStates: type[Xtructurable],
    q_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        q_values = q_model.apply(q_params, preproc, training=False)
        q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        cost = jnp.transpose(cost, (1, 0))
        valid_action_mask = jnp.isfinite(cost)
        has_valid_action = jnp.any(valid_action_mask, axis=1)
        q_sum_cost = jnp.where(jnp.isfinite(cost), q_values, jnp.inf)

        probs = boltzmann_action_selection(q_sum_cost, temperature=temperature)
        probs = jnp.where(valid_action_mask, probs, 0.0)
        probs_sum = jnp.sum(probs, axis=1, keepdims=True)
        probs = jnp.where(probs_sum > 0.0, probs / (probs_sum + 1e-8), probs)
        uniform_all = jnp.ones_like(probs) / jnp.maximum(probs.shape[1], 1)
        probs = jnp.where(has_valid_action[:, jnp.newaxis], probs, uniform_all)

        idxs = jnp.arange(q_values.shape[1])
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )
        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        neighbor_cost = jnp.transpose(neighbor_cost, (1, 0))
        neighbor_valid_mask = jnp.isfinite(neighbor_cost)
        has_valid_neighbor = jnp.any(neighbor_valid_mask, axis=1)
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )
        selected_neighbors_solved = jnp.logical_or(
            selected_neighbors_solved, jnp.logical_not(has_valid_action)
        )

        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        # Evaluate next-state Q-values with frozen target_q_params to stabilize Bellman bootstrapping.
        q = q_model.apply(target_q_params, preproc_neighbors, training=False)
        q = jnp.nan_to_num(q, posinf=1e6, neginf=-1e6)
        valid_neighbor_cost = jnp.where(neighbor_valid_mask, neighbor_cost, jnp.inf)

        # Q(s, a) includes c(s, a), so next-state values do not add neighbor_cost.
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

        min_next_q = jnp.maximum(min_next_q, 0.0)
        min_next_q = jnp.where(has_valid_neighbor, min_next_q, 0.0)

        selected_cost = jnp.take_along_axis(cost, actions[:, jnp.newaxis], axis=1).squeeze(1)
        selected_cost = jnp.where(has_valid_action, selected_cost, 0.0)
        target_q = selected_cost + jnp.where(selected_neighbors_solved, 0.0, min_next_q)
        target_q = jnp.where(solved, 0.0, target_q)
        target_q = jnp.where(has_valid_action, target_q, 0.0)

        return key, (
            solve_configs,
            states,
            target_q,
            actions,
            solved,
        )

    _, (solve_configs, states, target_q, actions, solved) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_q = target_q.reshape((-1, 1))
    actions = actions.reshape((-1, 1))
    is_solved = solved.reshape((-1,))

    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    diffusion_q = _compute_diffusion_q(
        solve_configs,
        states,
        is_solved,
        trajectory_actions,
        shuffled_path["move_costs"],
        shuffled_path["action_costs"].reshape((-1, 1)),
        shuffled_path["parent_indices"],
        SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates,
        k_max,
    )
    # Diffusion Q values describe the trajectory action; cap only rows where the
    # sampled action matches it, so targets of unrelated actions stay untouched.
    same_action = actions == trajectory_actions
    target_q = jnp.where(same_action, jnp.minimum(target_q, diffusion_q), target_q)

    return {
        "solve_config": solve_configs,
        "state": states,
        "distance": target_q,
        "action": actions,
    }


def _compute_diffusion_q(
    solve_configs: chex.Array,
    states: chex.Array,
    is_solved: chex.Array,
    trajectory_actions: chex.Array,
    move_costs: chex.Array,
    action_costs: chex.Array,
    parent_indices: chex.Array,
    SolveConfigsAndStatesAndActions: type[Xtructurable],
    SolveConfigsAndStates: type[Xtructurable],
    k_max: int,
):
    raw_move_costs = move_costs.reshape((-1,))
    solve_configs_and_states_and_actions = SolveConfigsAndStatesAndActions(
        solveconfigs=solve_configs,
        states=states,
        actions=trajectory_actions,
    )

    _, unique_state_action_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=raw_move_costs,
        return_index=True,
        return_inverse=True,
    )
    num_unique_state_actions = unique_state_action_idx.shape[0]

    # Group by state so Bellman propagation takes the min over all actions at the parent state.
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
    SolveConfigsAndStatesAndActions: type[Xtructurable],
    SolveConfigsAndStates: type[Xtructurable],
    q_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    temperature: float = 1.0 / 3.0,
    use_double_dqn: bool = False,
):
    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]
    action_costs = shuffled_path["action_costs"].reshape((-1, 1))
    parent_indices = shuffled_path["parent_indices"]

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

    return {
        "solve_config": solve_configs_flat,
        "state": states_flat,
        "distance": target_q,
        "action": trajectory_actions,
    }


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
    label: str = "td",
    diffusion_warmup_steps: int = 0,
    non_backtracking_steps: int = 3,
):
    if label not in ("td", "diffusion", "warmup_td"):
        raise ValueError(f"Unknown training label: {label!r}")
    if label == "warmup_td" and diffusion_warmup_steps <= 0:
        raise ValueError("label='warmup_td' requires diffusion_warmup_steps > 0")
    (nn_minibatch_size, _, steps, jited_create_shuffled_path,) = prepare_shuffled_path_sampling(
        puzzle=puzzle,
        dataset_size=dataset_size,
        k_max=k_max,
        max_batch_size=262144,
        using_hindsight_target=using_hindsight_target,
        using_triangular_sampling=using_triangular_sampling,
        include_action_costs=False,
        non_backtracking_steps=non_backtracking_steps,
        create_hindsight_target_shuffled_path=create_hindsight_target_shuffled_path,
        create_hindsight_target_triangular_shuffled_path=(
            create_hindsight_target_triangular_shuffled_path
        ),
        create_target_shuffled_path=create_target_shuffled_path,
        fixed_target_error="Fixed target is not supported for hindsight target",
    )

    class SolveConfigsAndStatesAndActions(Xtructurable):
        solveconfigs: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.SolveConfig)]
        states: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.State)]
        actions: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,))]

    class SolveConfigsAndStates(Xtructurable):
        solveconfigs: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.SolveConfig)]
        states: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.State)]

    base_get_datasets = partial(
        _get_datasets_with_policy,
        puzzle,
        preproc_fn,
        SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates,
        q_model,
        nn_minibatch_size,
        k_max=k_max,
        temperature=temperature,
        use_double_dqn=use_double_dqn,
    )
    diffusion_get_datasets = partial(
        _get_datasets_with_diffusion_distance,
        puzzle,
        preproc_fn,
        SolveConfigsAndStatesAndActions,
        SolveConfigsAndStates,
        q_model,
        nn_minibatch_size,
        k_max=k_max,
        temperature=temperature,
        use_double_dqn=use_double_dqn,
    )

    return wrap_dataset_runner(
        dataset_size=dataset_size,
        steps=steps,
        jited_create_shuffled_path=jited_create_shuffled_path,
        base_get_datasets=base_get_datasets,
        diffusion_get_datasets=diffusion_get_datasets,
        should_use_diffusion_fn=make_diffusion_step_selector(
            label=label,
            diffusion_warmup_steps=diffusion_warmup_steps,
        ),
        n_devices=n_devices,
    )
