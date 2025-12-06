import math
from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from heuristic.neuralheuristic.neuralheuristic_base import HeuristicBase
from train_util.annotate import MAX_GEN_DS_BATCH_SIZE
from train_util.losses import loss_from_diff
from train_util.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
)


def davi_builder(
    minibatch_size: int,
    heuristic_model: HeuristicBase,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: Optional[dict[str, Any]] = None,
    replay_ratio: int = 1,
    td_error_clip: Optional[float] = None,
):
    def davi_loss(
        heuristic_params: Any,
        solveconfigs: chex.Array,
        states: chex.Array,
        target_heuristic: chex.Array,
        weights: chex.Array,
    ):
        # Preprocess during training
        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        current_heuristic, variable_updates = apply_with_conditional_batch_stats(
            heuristic_model.apply, heuristic_params, preproc, training=True, n_devices=n_devices
        )
        new_params = build_new_params_from_updates(heuristic_params, variable_updates)
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        per_sample = loss_from_diff(diff, loss=loss_type, loss_args=loss_args)
        loss_value = jnp.mean(per_sample * weights)
        return loss_value, new_params

    def davi(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_heuristic = dataset["target_heuristic"]
        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            solveconfigs, states, target_heuristic, weights = batched_dataset
            (loss, heuristic_params), grads = jax.value_and_grad(davi_loss, has_aux=True)(
                heuristic_params,
                solveconfigs,
                states,
                target_heuristic,
                weights,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state), loss

        # Repeat training loop for replay_ratio iterations with reshuffling
        def replay_loop(carry, replay_key):
            heuristic_params, opt_state = carry

            key_perm_replay, key_fill_replay = jax.random.split(replay_key)
            batch_indexs_replay = jnp.concatenate(
                [
                    jax.random.permutation(key_perm_replay, jnp.arange(data_size)),
                    jax.random.randint(
                        key_fill_replay,
                        (batch_size * minibatch_size - data_size,),
                        0,
                        data_size,
                    ),
                ],
                axis=0,
            )
            loss_weights_replay = loss_weights

            batch_indexs_replay = jnp.reshape(batch_indexs_replay, (batch_size, minibatch_size))

            # Create new batches with reshuffled indices
            batched_solveconfigs_replay = xnp.take(solveconfigs, batch_indexs_replay, axis=0)
            batched_states_replay = xnp.take(states, batch_indexs_replay, axis=0)
            batched_target_heuristic_replay = jnp.take(
                target_heuristic, batch_indexs_replay, axis=0
            )
            batched_weights_replay = jnp.take(loss_weights_replay, batch_indexs_replay, axis=0)
            # Normalize weights per batch to prevent scale drift
            batched_weights_replay = batched_weights_replay / (
                jnp.mean(batched_weights_replay, axis=1, keepdims=True) + 1e-8
            )

            (heuristic_params, opt_state), losses = jax.lax.scan(
                train_loop,
                (heuristic_params, opt_state),
                (
                    batched_solveconfigs_replay,
                    batched_states_replay,
                    batched_target_heuristic_replay,
                    batched_weights_replay,
                ),
            )
            return (heuristic_params, opt_state), losses

        # Generate keys for replay iterations
        replay_keys = jax.random.split(key, replay_ratio)
        (heuristic_params, opt_state), losses = jax.lax.scan(
            replay_loop,
            (heuristic_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        return (
            heuristic_params,
            opt_state,
            loss,
        )

    if n_devices > 1:

        def pmap_davi(key, dataset, heuristic_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (heuristic_params, opt_state, loss,) = jax.pmap(
                davi, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, heuristic_params, opt_state)
            heuristic_params = jax.tree_util.tree_map(lambda xs: xs[0], heuristic_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            return (
                heuristic_params,
                opt_state,
                loss,
            )

        return pmap_davi
    else:
        return jax.jit(davi)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: HeuristicBase,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
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
    solve_configs_and_states = SolveConfigsAndStates(solveconfigs=solve_configs, states=states)
    target_heuristic = move_costs.reshape((-1,))

    # Find unique states and broadcast the minimal target_heuristic to all duplicates
    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=target_heuristic,
        return_index=True,
        return_inverse=True,
    )
    target_heuristic = target_heuristic[unique_uint32eds_idx][inverse_indices]

    # Propagate the improved heuristic values backwards along the trajectory
    dataset_size = target_heuristic.shape[0]

    # Pad dataset with infinity to handle invalid parent pointers
    padded_heuristic = jnp.pad(target_heuristic, (0, 1), constant_values=jnp.inf)

    # Map -1 or out-of-bounds indices to the padded infinity value
    safe_parent_indices = jnp.where(
        (parent_indices < 0) | (parent_indices >= dataset_size), dataset_size, parent_indices
    )

    def body_fun(i, h):
        # h is padded [N+1]
        current_h = h[:dataset_size]

        # Gather heuristic from parents (neighbors closer to goal)
        h_parents = h[safe_parent_indices]  # [N]

        # Bellman update: h(s) <= c(s, s') + h(s')
        # action_costs is c(s, s') where s' is the parent/next state in path
        new_h = action_costs + h_parents

        improved_h = jnp.minimum(current_h, new_h)
        return h.at[:dataset_size].set(improved_h)

    # Iterate k_max times to propagate along the longest possible path
    final_padded_h = jax.lax.fori_loop(0, k_max, body_fun, padded_heuristic)

    target_heuristic = final_padded_h[:dataset_size]
    return target_heuristic


def _get_datasets_with_diffusion_distance(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStates: Xtructurable,
    heuristic_model: HeuristicBase,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
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
    heuristic_model: HeuristicBase,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
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
        td_error_clip,
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
    heuristic_model: HeuristicBase,
    dataset_size: int,
    k_max: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
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
        td_error_clip=td_error_clip,
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
                td_error_clip=td_error_clip,
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
                td_error_clip=td_error_clip,
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
            for k, v in paths.items():
                paths[k] = v.flatten()[:dataset_size]
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
