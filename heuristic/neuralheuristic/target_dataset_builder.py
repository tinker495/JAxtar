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


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size))

    def get_minibatched_datasets(_, vals):
        solve_configs, states, move_costs = vals
        solved = puzzle.batched_is_solved(
            solve_configs, states, multi_solve_config=True
        )  # [batch_size]
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        preproc_neighbors = jax.vmap(jax.vmap(preproc_fn, in_axes=(0, 0)), in_axes=(None, 0))(
            solve_configs, neighbors
        )
        # preproc_neighbors: [action_size, batch_size, ...]

        def compute_heur_for_action(carry, preproc_neighbor):
            heur = heuristic_model.apply(target_heuristic_params, preproc_neighbor, training=False)
            return None, heur.squeeze()

        _, heur = jax.lax.scan(
            compute_heur_for_action,
            None,
            preproc_neighbors,
        )  # [action_size, batch_size]
        backup = heur + cost  # [action_size, batch_size]
        target_heuristic = jnp.min(jnp.maximum(backup, cost), axis=0)
        target_heuristic = jnp.where(
            solved, 0.0, target_heuristic
        )  # if the puzzle is already solved, the heuristic is 0
        target_heuristic = jnp.minimum(target_heuristic, move_costs)
        # heuristic cannot be greater than the move cost.

        return None, (
            solve_configs,
            states,
            target_heuristic,
            move_costs,
        )

    _, (solve_configs, states, target_heuristic, cost,) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_heuristic = target_heuristic.reshape((-1,))
    cost = cost.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_heuristic": target_heuristic,
        "cost": cost,
    }


def _compute_diffusion_distance(
    solve_configs: chex.Array,
    states: chex.Array,
    move_costs: chex.Array,
    action_costs: chex.Array,
    parent_indices: chex.Array,
    SolveConfigsAndStates: Xtructurable,
    k_max: int,
):
    raw_move_costs = move_costs.reshape((-1,))
    action_costs = action_costs.reshape((-1,))
    solve_configs_and_states = SolveConfigsAndStates(solveconfigs=solve_configs, states=states)

    # 1. Unique states
    _, unique_state_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=raw_move_costs,
        return_index=True,
        return_inverse=True,
    )
    num_unique_states = unique_state_idx.shape[0]

    # 2. Compute diffusion using common utility
    return compute_diffusion_targets(
        initial_values=raw_move_costs,
        parent_indices=parent_indices,
        action_costs=action_costs,
        raw_move_costs=raw_move_costs,
        k_max=k_max,
        inverse_indices=inverse_indices,
        num_unique=num_unique_states,
    )


def _get_datasets_with_diffusion_distance(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStates: Xtructurable,
    heuristic_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
):

    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]
    action_costs = shuffled_path["action_costs"]

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    action_costs = action_costs.reshape((-1,))
    parent_indices = shuffled_path["parent_indices"]

    target_heuristic = _compute_diffusion_distance(
        solve_configs,
        states,
        move_costs,
        action_costs,
        parent_indices,
        SolveConfigsAndStates,
        k_max,
    )

    cost = move_costs.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_heuristic": target_heuristic,
        "cost": cost,
    }


def _get_datasets_with_diffusion_distance_mixture(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStates: Xtructurable,
    heuristic_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
):
    return_dict = _get_datasets(
        puzzle,
        preproc_fn,
        heuristic_model,
        minibatch_size,
        target_heuristic_params,
        heuristic_params,
        shuffled_path,
        key,
        temperature,
    )
    target_heuristic = return_dict["target_heuristic"]

    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]
    action_costs = shuffled_path["action_costs"]
    parent_indices = shuffled_path["parent_indices"]

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    action_costs = action_costs.reshape((-1,))

    diffusion_heuristic = _compute_diffusion_distance(
        solve_configs,
        states,
        move_costs,
        action_costs,
        parent_indices,
        SolveConfigsAndStates,
        k_max,
    )

    # Mixture: target_heuristic = max(target_heuristic, diffusion_heuristic * 0.8 - 2.0)
    target_heuristic = jnp.maximum(target_heuristic, diffusion_heuristic * 0.8 - 2.0)
    return_dict["target_heuristic"] = target_heuristic

    return return_dict


def get_heuristic_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: DistanceModel | DistanceHLGModel,
    dataset_size: int,
    k_max: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
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
        if using_triangular_sampling:
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                True,
                non_backtracking_steps=non_backtracking_steps,
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                True,
                non_backtracking_steps=non_backtracking_steps,
            )
    else:
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            k_max,
            shuffle_parallel,
            True,
            non_backtracking_steps=non_backtracking_steps,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    base_get_datasets = partial(
        _get_datasets,
        puzzle,
        preproc_fn,
        heuristic_model,
        nn_minibatch_size,
        temperature=temperature,
    )

    use_diffusion_features = use_diffusion_distance or use_diffusion_distance_mixture

    if use_diffusion_features:

        @xtructure_dataclass
        class SolveConfigsAndStates:
            solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
            states: FieldDescriptor.scalar(dtype=puzzle.State)

        if use_diffusion_distance_mixture:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance_mixture,
                puzzle,
                preproc_fn,
                SolveConfigsAndStates,
                heuristic_model,
                nn_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
            )
        else:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance,
                puzzle,
                preproc_fn,
                SolveConfigsAndStates,
                heuristic_model,
                nn_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
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
