import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from xtructure.core.type_utils import is_xtructure_dataclass_instance


def minibatch_datasets(
    *train_datasets: dict[str, chex.Array],
    data_size: int = None,
    batch_size: int = None,
    minibatch_size: int = None,
    key: chex.PRNGKey = jax.random.PRNGKey(0),
):
    key_perm_replay, key_fill_replay = jax.random.split(key)
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

    batch_indexs_replay = jnp.reshape(batch_indexs_replay, (batch_size, minibatch_size))

    batched_train_datasets = []
    for train_dataset in train_datasets:
        if is_xtructure_dataclass_instance(train_dataset):
            batched_train_dataset = xnp.take(train_dataset, batch_indexs_replay, axis=0)
        else:
            batched_train_dataset = jnp.take(train_dataset, batch_indexs_replay, axis=0)
        batched_train_datasets.append(batched_train_dataset)
    return tuple(batched_train_datasets)


def calculate_dataset_params(dataset_size: int, k_max: int, max_batch_size: int):
    """Calculate optimal minibatch size and shuffle parameters for dataset generation."""
    # Calculate optimal nn_minibatch_size
    # It must be <= max_batch_size and divide dataset_size
    n_batches = math.ceil(dataset_size / max_batch_size)
    while dataset_size % n_batches != 0:
        n_batches += 1
    nn_minibatch_size = dataset_size // n_batches

    # Calculate optimal shuffle_parallel and steps to respect max_batch_size
    max_shuffle_parallel = max(1, int(max_batch_size / k_max))
    needed_trajectories = math.ceil(dataset_size / k_max)
    shuffle_parallel = min(needed_trajectories, max_shuffle_parallel)
    steps = math.ceil(needed_trajectories / shuffle_parallel)

    return nn_minibatch_size, shuffle_parallel, steps


def compute_diffusion_targets(
    initial_values: chex.Array,  # [N] or [N, D]
    is_solved: chex.Array,  # [N]
    parent_indices: chex.Array,  # [N]
    action_costs: chex.Array,  # [N] or [N, D]
    raw_move_costs: chex.Array,  # [N]
    k_max: int,
    inverse_indices: chex.Array,  # [N] mapping row to unique group
    num_unique: int,
    inverse_state_indices: chex.Array = None,  # [N] mapping row to unique state group
    num_unique_states: int = None,
):
    """Compute diffusion targets using Bellman propagation along trajectories.

    This unifies the logic for state-based (heuristic) and state-action-based (Q) diffusion.
    """
    dataset_size = initial_values.shape[0]

    # Ensure values are at least 2D [N, D] for consistent handling
    is_1d = initial_values.ndim == 1
    if is_1d:
        initial_values = initial_values[:, jnp.newaxis]
        action_costs = action_costs[:, jnp.newaxis]

    # Always ensure is_solved matches initial_values dimensions for broadcasting
    if is_solved.ndim == 1:
        is_solved = is_solved[:, jnp.newaxis]

    # 1. Edge cost alignment logic
    idx = jnp.arange(dataset_size, dtype=parent_indices.dtype)
    valid_parent = (parent_indices >= 0) & (parent_indices < dataset_size)
    safe_parent_indices = jnp.where(valid_parent, parent_indices, dataset_size)

    parent_is_behind = (safe_parent_indices < idx) & valid_parent
    behind_ratio = jnp.sum(parent_is_behind).astype(jnp.float32) / jnp.maximum(
        jnp.sum(valid_parent).astype(jnp.float32), 1.0
    )
    default_use_parent_indexed_costs = behind_ratio > 0.5

    padded_action_costs = jnp.pad(action_costs, ((0, 1), (0, 0)), constant_values=0.0)
    padded_move_costs = jnp.pad(raw_move_costs, (0, 1), constant_values=0.0)

    parent_move_costs = padded_move_costs[safe_parent_indices]
    if parent_move_costs.ndim == 1:
        parent_move_costs = parent_move_costs[:, jnp.newaxis]

    parent_aligned_costs = padded_action_costs[safe_parent_indices]
    child_aligned_costs = action_costs

    # Error checks for cost alignment
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

    def _collapse(vals, inv_idx, n_unique):
        return jnp.full((n_unique, vals.shape[1]), jnp.inf, dtype=vals.dtype).at[inv_idx].min(vals)

    def body_fun(_, v):
        # v is padded [N+1, D]
        unique_v = _collapse(v[:dataset_size], inverse_indices, num_unique)
        current_v = unique_v[inverse_indices]

        if inverse_state_indices is not None and num_unique_states is not None:
            # Q-learning: combine global state-min info and trajectory info
            state_min_v = _collapse(current_v, inverse_state_indices, num_unique_states)
            padded_state_min_v = jnp.pad(state_min_v, ((0, 1), (0, 0)), constant_values=jnp.inf)

            inverse_state_indices_padded = jnp.pad(
                inverse_state_indices, (0, 1), constant_values=num_unique_states
            )
            parent_state_group = inverse_state_indices_padded[safe_parent_indices]
            v_parents_optimal = padded_state_min_v[parent_state_group]

            collapsed_padded_v = jnp.pad(current_v, ((0, 1), (0, 0)), constant_values=jnp.inf)
            v_parents_prop = collapsed_padded_v[safe_parent_indices]
            v_parents = jnp.minimum(v_parents_optimal, v_parents_prop)
        else:
            # Heuristic: simple propagation
            collapsed_padded_v = jnp.pad(current_v, ((0, 1), (0, 0)), constant_values=jnp.inf)
            v_parents = collapsed_padded_v[safe_parent_indices]

        new_v = edge_costs + v_parents
        improved_v = jnp.minimum(current_v, new_v)

        # [Goal State Anchoring]
        improved_v = jnp.where(is_solved, 0.0, improved_v)

        # [Utilizing State Identity]
        # Consolidate improvements across identical states immediately
        unique_improved = _collapse(improved_v, inverse_indices, num_unique)
        improved_v = unique_improved[inverse_indices]

        return v.at[:dataset_size].set(improved_v)

    padded_v = jnp.pad(initial_values, ((0, 1), (0, 0)), constant_values=jnp.inf)
    final_padded_v = jax.lax.fori_loop(0, k_max, body_fun, padded_v)

    result = _collapse(final_padded_v[:dataset_size], inverse_indices, num_unique)[inverse_indices]

    if is_1d:
        result = result.squeeze(1)
    return result


def wrap_dataset_runner(
    dataset_size: int,
    steps: int,
    jited_create_shuffled_path: Callable,
    base_get_datasets: Callable,
    diffusion_get_datasets: Callable,
    should_use_diffusion_fn: Callable[[int], bool],
    n_devices: int = 1,
):
    """Wrap dataset extraction into a runner that handles pmap and diffusion warmup.

    The returned function accepts a TrainStateExtended object and extracts:
    - target_params from state.target_params (with batch_stats if present)
    - eval_params from state.params (with batch_stats if present)
    """
    from train_util.optimizer import get_eval_params
    from train_util.train_state import TrainStateExtended

    def _extract_params_from_state(state: TrainStateExtended) -> tuple[dict, dict]:
        """Extract target_params and eval_params from TrainStateExtended."""
        target_params = {"params": state.target_params}
        eval_params = {"params": get_eval_params(state.opt_state, state.params)}

        if state.batch_stats is not None:
            target_params["batch_stats"] = eval_params["batch_stats"] = state.batch_stats
        return target_params, eval_params

    def build_runner(dataset_extractor: Callable):
        @jax.jit
        def runner(target_params: Any, params: Any, key: chex.PRNGKey):
            def scan_fn(scan_key, _):
                scan_key, subkey = jax.random.split(scan_key)
                paths = jited_create_shuffled_path(subkey)
                return scan_key, paths

            key_inner, paths = jax.lax.scan(scan_fn, key, None, length=steps)
            paths = flatten_scanned_paths(paths, dataset_size)
            flatten_dataset = dataset_extractor(target_params, params, paths, key_inner)
            return flatten_dataset

        return runner

    default_runner = build_runner(base_get_datasets)
    diffusion_runner = build_runner(diffusion_get_datasets)

    if n_devices > 1:
        pmap_default_runner = jax.pmap(default_runner, in_axes=(None, None, 0))
        pmap_diffusion_runner = jax.pmap(diffusion_runner, in_axes=(None, None, 0))

        def get_datasets(state: TrainStateExtended, key: chex.PRNGKey, step: int):
            target_params, eval_params = _extract_params_from_state(state)
            keys = jax.random.split(key, n_devices)
            runner = pmap_diffusion_runner if should_use_diffusion_fn(step) else pmap_default_runner
            return runner(target_params, eval_params, keys)

        return get_datasets

    def single_device_get_datasets(state: TrainStateExtended, key: chex.PRNGKey, step: int):
        target_params, eval_params = _extract_params_from_state(state)
        runner = diffusion_runner if should_use_diffusion_fn(step) else default_runner
        return runner(target_params, eval_params, key)

    return single_device_get_datasets


def _offset_parent_indices_for_scan(parent_indices: chex.Array) -> chex.Array:
    """Offset local `parent_indices` when multiple trajectory chunks are concatenated.

    The various `create_*_shuffled_path` functions return `parent_indices` that are local to a
    single chunk: indices are in `[0, chunk_size)` (or `-1` for invalid), where
    `chunk_size == k_max * shuffle_parallel`.

    In dataset builders, we often generate multiple chunks via `jax.lax.scan`, producing an array
    shaped `[steps, chunk_size]`. When these are flattened, each chunk must have its parent indices
    shifted by `step_idx * chunk_size` so pointers remain within the correct chunk.
    """
    if parent_indices.ndim != 2:
        return parent_indices
    steps, chunk_size = parent_indices.shape
    offsets = (jnp.arange(steps, dtype=parent_indices.dtype) * chunk_size)[:, jnp.newaxis]
    return jnp.where(parent_indices >= 0, parent_indices + offsets, parent_indices)


def flatten_scanned_paths(paths: dict[str, chex.Array], dataset_size: int) -> dict[str, chex.Array]:
    """Flatten `jax.lax.scan` outputs of shuffled paths into a dataset batch.

    Handles `parent_indices` specially: when multiple chunks are concatenated, parent pointers are
    offset so they continue to reference the correct row after flattening.
    """
    dataset_size = int(dataset_size)
    out = dict(paths)
    if "parent_indices" in out:
        out["parent_indices"] = _offset_parent_indices_for_scan(out["parent_indices"])
    for k, v in out.items():
        out[k] = v.flatten()[:dataset_size]
    return out
