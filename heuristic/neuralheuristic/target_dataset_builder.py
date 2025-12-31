import math
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from neural_util.basemodel import BaseModel
from train_util.annotate import MAX_GEN_DS_BATCH_SIZE
from train_util.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
    flatten_scanned_paths,
)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: BaseModel,
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
    target_heuristic = raw_move_costs

    # Find unique states and broadcast the minimal target_heuristic to all duplicates
    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=target_heuristic,
        return_index=True,
        return_inverse=True,
    )
    num_unique_states = unique_uint32eds_idx.shape[0]

    def _collapse_duplicate_states(h_vec: chex.Array) -> chex.Array:
        """Collapse duplicate (solve_config, state) rows to their minimal heuristic.

        Important: duplicate states can appear in different trajectories. During diffusion,
        one copy may get improved earlier than another; re-collapsing ensures all parents
        always see the shortest value for a repeated child state.
        """
        group_min = (
            jnp.full((num_unique_states,), jnp.inf, dtype=h_vec.dtype)
            .at[inverse_indices]
            .min(h_vec)
        )
        return group_min[inverse_indices]

    target_heuristic = _collapse_duplicate_states(target_heuristic)

    # Propagate the improved heuristic values backwards along the trajectory
    dataset_size = target_heuristic.shape[0]

    # Pad dataset with infinity to handle invalid parent pointers
    padded_heuristic = jnp.pad(target_heuristic, (0, 1), constant_values=jnp.inf)

    # Map -1 or out-of-bounds indices to the padded infinity value
    safe_parent_indices = jnp.where(
        (parent_indices < 0) | (parent_indices >= dataset_size), dataset_size, parent_indices
    )
    idx = jnp.arange(dataset_size, dtype=safe_parent_indices.dtype)
    valid_parent = safe_parent_indices[:dataset_size] != dataset_size
    # Heuristic for whether `action_costs` are aligned to the child row or the parent row:
    # - Hindsight trajectory sampling uses parent=i+P (parent index ahead), and action_costs are aligned to the child.
    # - Inverse-trajectory sampling uses parent=i-P (parent index behind), and action_costs are aligned to the parent.
    parent_is_behind = (safe_parent_indices[:dataset_size] < idx) & valid_parent
    behind_ratio = jnp.sum(parent_is_behind).astype(jnp.float32) / jnp.maximum(
        jnp.sum(valid_parent).astype(jnp.float32), 1.0
    )
    default_use_parent_indexed_costs = behind_ratio > 0.5

    padded_action_costs = jnp.pad(action_costs, (0, 1), constant_values=0.0)
    padded_move_costs = jnp.pad(raw_move_costs, (0, 1), constant_values=0.0)

    # If `move_costs` are true path costs (as produced by `create_*_shuffled_path`), we can
    # disambiguate parent- vs child-aligned `action_costs` by checking which alignment satisfies:
    #   move_cost(i) ≈ move_cost(parent(i)) + edge_cost(i)
    parent_move_costs = padded_move_costs[safe_parent_indices]
    parent_aligned_costs = padded_action_costs[safe_parent_indices]
    child_aligned_costs = action_costs
    err_child = jnp.abs(raw_move_costs - (parent_move_costs + child_aligned_costs))
    err_parent = jnp.abs(raw_move_costs - (parent_move_costs + parent_aligned_costs))
    valid_parent_f = valid_parent.astype(raw_move_costs.dtype)
    denom = jnp.maximum(jnp.sum(valid_parent_f), 1.0)
    mean_err_child = jnp.sum(err_child * valid_parent_f) / denom
    mean_err_parent = jnp.sum(err_parent * valid_parent_f) / denom
    min_mean_err = jnp.minimum(mean_err_child, mean_err_parent)
    # Only trust this check when errors are near-zero; otherwise fall back to the index-direction heuristic.
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

    def body_fun(i, h):
        # h is padded [N+1]
        current_h = _collapse_duplicate_states(h[:dataset_size])
        collapsed_padded_h = jnp.pad(current_h, (0, 1), constant_values=jnp.inf)

        # Gather heuristic from parents (neighbors closer to goal)
        h_parents = collapsed_padded_h[safe_parent_indices]  # [N]

        # Bellman update: h(s) <= c(s, s') + h(s')
        # action_costs is c(s, s') where s' is the parent/next state in path
        new_h = edge_costs + h_parents

        improved_h = jnp.minimum(current_h, new_h)
        return h.at[:dataset_size].set(improved_h)

    # Iterate k_max times to propagate along the longest possible path
    final_padded_h = jax.lax.fori_loop(0, k_max, body_fun, padded_heuristic)

    target_heuristic = _collapse_duplicate_states(final_padded_h[:dataset_size])
    return target_heuristic


def _get_datasets_with_diffusion_distance(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStates: Xtructurable,
    heuristic_model: BaseModel,
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
    heuristic_model: BaseModel,
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
    heuristic_model: BaseModel,
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

    def build_runner(dataset_extractor: Callable):
        @jax.jit
        def runner(
            target_heuristic_params: Any,
            heuristic_params: Any,
            key: chex.PRNGKey,
        ):
            def scan_fn(scan_key, _):
                scan_key, subkey = jax.random.split(scan_key)
                paths = jited_create_shuffled_path(subkey)
                return scan_key, paths

            key_inner, paths = jax.lax.scan(scan_fn, key, None, length=steps)
            paths = flatten_scanned_paths(paths, dataset_size)
            flatten_dataset = dataset_extractor(
                target_heuristic_params,
                heuristic_params,
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

        def get_datasets(target_heuristic_params, heuristic_params, key, step: int):
            keys = jax.random.split(key, n_devices)
            runner = pmap_diffusion_runner if should_use_diffusion(step) else pmap_default_runner
            return runner(target_heuristic_params, heuristic_params, keys)

        return get_datasets

    def single_device_get_datasets(target_heuristic_params, heuristic_params, key, step: int):
        runner = diffusion_runner if should_use_diffusion(step) else default_runner
        return runner(target_heuristic_params, heuristic_params, key)

    return single_device_get_datasets
