import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure.core.type_utils import is_xtructure_dataclass_instance


def minibatch_datasets(
    *train_datasets: dict[str, chex.Array],
    data_size: int = None,
    batch_size: int = None,
    minibatch_size: int = None,
    sample_path_length: int = 1,
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
    from train_util.train_state import TrainStateExtended
    from train_util.optimizer import get_eval_params

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


def _masked_action_sample_uniform(mask: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Sample an action uniformly among True entries in `mask`.

    Args:
        mask: bool array shaped [action, batch]
        key: PRNGKey

    Returns:
        actions: int array shaped [batch]
    """
    # `jax.random.categorical` samples along the last axis, so transpose to [batch, action].
    mask_bt = mask.T
    # Use a large negative value instead of -inf for better numerical/device compatibility.
    logits = jnp.where(
        mask_bt, jnp.array(0.0, dtype=jnp.float32), jnp.array(-1.0e9, dtype=jnp.float32)
    )
    keys = jax.random.split(key, logits.shape[0])
    actions = jax.vmap(lambda k, lg: jax.random.categorical(k, lg, axis=-1))(keys, logits)
    return actions.astype(jnp.int32)


def _gather_by_action(neighbor_states, actions: chex.Array):
    """Gather `neighbor_states[action, batch, ...]` using `actions[batch]` for each batch."""
    batch_idx = jnp.arange(actions.shape[0], dtype=jnp.int32)

    def _gather(leaf: chex.Array) -> chex.Array:
        return leaf[actions, batch_idx, ...]

    return jax.tree_util.tree_map(_gather, neighbor_states)


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


def _leafwise_equal(candidate_leaf: chex.Array, reference_leaf: chex.Array) -> chex.Array:
    expanded_ref = reference_leaf[jnp.newaxis, ...]
    eq = jnp.equal(candidate_leaf, expanded_ref)
    if eq.ndim <= 2:
        return eq
    axes = tuple(range(2, eq.ndim))
    return jnp.all(eq, axis=axes)


def _states_equal(candidate_states, reference_state) -> chex.Array:
    equality_tree = jax.tree_util.tree_map(_leafwise_equal, candidate_states, reference_state)
    leaves, _ = jax.tree_util.tree_flatten(equality_tree)
    if not leaves:
        raise ValueError("State comparison received an empty tree")
    result = leaves[0]
    for leaf in leaves[1:]:
        result = jnp.logical_and(result, leaf)
    return result


def _match_history(candidate_states, history_states) -> chex.Array:
    def _compare(prev_state):
        return _states_equal(candidate_states, prev_state)

    matches = jax.vmap(_compare)(history_states)
    return jnp.any(matches, axis=0)


def _initialize_history(state, history_len: int):
    if history_len <= 0:
        return None

    def _repeat(leaf):
        expanded = leaf[jnp.newaxis, ...]
        return jnp.repeat(expanded, repeats=history_len, axis=0)

    return jax.tree_util.tree_map(_repeat, state)


def _roll_history(history_states, new_state):
    if history_states is None:
        return None
    tail = history_states[1:, ...]
    new_entry = new_state[jnp.newaxis, ...]
    return xnp.concatenate([tail, new_entry], axis=0)


def get_random_inverse_trajectory(
    puzzle: Puzzle,
    k_max: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    key_inits, key_targets, key_scan = jax.random.split(key, 3)
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key_inits, shuffle_parallel))
    target_states = jax.vmap(puzzle.solve_config_to_state_transform, in_axes=(0, 0))(
        solve_configs, jax.random.split(key_targets, shuffle_parallel)
    )

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    history_states = _initialize_history(target_states, int(non_backtracking_steps))
    use_history = history_states is not None

    step_keys = jax.random.split(key_scan, k_max)

    def _scan(carry, step_key):
        history, state, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        action_mask = jnp.isfinite(cost)  # bool [action, batch]
        history_block = (
            _match_history(neighbor_states, history) if use_history else jnp.zeros_like(action_mask)
        )
        same_block = _states_equal(neighbor_states, state)
        backtracking_mask = (~history_block) & (~same_block)
        masked = action_mask & backtracking_mask
        no_valid_backtracking = jnp.sum(masked, axis=0) == 0  # [batch]
        final_mask = jnp.where(no_valid_backtracking[jnp.newaxis, :], action_mask, masked)
        inv_actions = _masked_action_sample_uniform(final_mask, step_key)  # [batch]
        next_state = _gather_by_action(neighbor_states, inv_actions)
        batch_idx = jnp.arange(inv_actions.shape[0], dtype=jnp.int32)
        step_cost = cost[inv_actions, batch_idx]  # [batch]
        next_history = _roll_history(history, state) if use_history else history
        return (
            (next_history, next_state, move_cost + step_cost),  # carry
            (state, move_cost, inv_actions, step_cost),  # return
        )

    (_, last_state, last_move_cost), (
        states,
        move_costs,
        inv_actions,
        action_costs,
    ) = jax.lax.scan(
        _scan,
        (history_states, target_states, jnp.zeros(shuffle_parallel)),
        step_keys,
        length=k_max,
    )  # [k_max, batch_size, ...]

    states = xnp.concatenate(
        [states, last_state[jnp.newaxis, ...]], axis=0
    )  # [k_max + 1, shuffle_parallel, ...]
    move_costs = jnp.concatenate(
        [move_costs, last_move_cost[jnp.newaxis, ...]], axis=0
    )  # [k_max + 1, shuffle_parallel]
    move_costs_tm1 = jnp.concatenate(
        [jnp.zeros_like(move_costs[:1, ...]), move_costs[:-1, ...]], axis=0
    )

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "move_costs_tm1": move_costs_tm1,
        "actions": inv_actions,
        "action_costs": action_costs,
    }


def get_random_trajectory(
    puzzle: Puzzle,
    k_max: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    key_inits, key_scan = jax.random.split(key, 2)
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key_inits, shuffle_parallel)
    )

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    history_states = _initialize_history(initial_states, int(non_backtracking_steps))
    use_history = history_states is not None

    step_keys = jax.random.split(key_scan, k_max)

    def _scan(carry, step_key):
        history, state, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        action_mask = jnp.isfinite(cost)  # bool [action, batch]
        history_block = (
            _match_history(neighbor_states, history) if use_history else jnp.zeros_like(action_mask)
        )
        same_block = _states_equal(neighbor_states, state)
        backtracking_mask = (~history_block) & (~same_block)
        masked = action_mask & backtracking_mask
        no_valid_backtracking = jnp.sum(masked, axis=0) == 0  # [batch]
        final_mask = jnp.where(no_valid_backtracking[jnp.newaxis, :], action_mask, masked)
        actions = _masked_action_sample_uniform(final_mask, step_key)  # [batch]
        next_state = _gather_by_action(neighbor_states, actions)
        batch_idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
        step_cost = cost[actions, batch_idx]  # [batch]
        next_history = _roll_history(history, state) if use_history else history
        return (
            (next_history, next_state, move_cost + step_cost),  # carry
            (state, move_cost, actions, step_cost),  # return
        )

    (_, last_state, last_move_cost), (states, move_costs, actions, action_costs) = jax.lax.scan(
        _scan,
        (history_states, initial_states, jnp.zeros(shuffle_parallel)),
        step_keys,
        length=k_max,
    )  # [k_max, shuffle_parallel, ...]

    states = xnp.concatenate(
        [states, last_state[jnp.newaxis, ...]], axis=0
    )  # [k_max + 1, shuffle_parallel, ...]
    move_costs = jnp.concatenate(
        [move_costs, last_move_cost[jnp.newaxis, ...]], axis=0
    )  # [k_max + 1, shuffle_parallel]
    move_costs_tm1 = jnp.concatenate(
        [jnp.zeros_like(move_costs[:1, ...]), move_costs[:-1, ...]], axis=0
    )

    return {
        "solve_configs": solve_configs,  # [shuffle_parallel, ...]
        "states": states,  # [k_max + 1, shuffle_parallel, ...]
        "move_costs": move_costs,  # [k_max + 1, shuffle_parallel]
        "move_costs_tm1": move_costs_tm1,  # [k_max + 1, shuffle_parallel]
        "actions": actions,  # [k_max, shuffle_parallel]
        "action_costs": action_costs,  # [k_max, shuffle_parallel]
    }


def create_target_shuffled_path(
    puzzle: Puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    inverse_trajectory = get_random_inverse_trajectory(
        puzzle,
        k_max,
        shuffle_parallel,
        key,
        non_backtracking_steps=non_backtracking_steps,
    )

    solve_configs = inverse_trajectory["solve_configs"]
    if include_solved_states:
        states = inverse_trajectory["states"][:-1, ...]  # [k_max, shuffle_parallel, ...]
        move_costs = inverse_trajectory["move_costs"][:-1, ...]  # [k_max, shuffle_parallel]
        move_costs_tm1 = inverse_trajectory["move_costs_tm1"][:-1, ...]  # [k_max, shuffle_parallel]
    else:
        states = inverse_trajectory["states"][1:, ...]  # [k_max, shuffle_parallel, ...]
        move_costs = inverse_trajectory["move_costs"][1:, ...]  # [k_max, shuffle_parallel]
        move_costs_tm1 = inverse_trajectory["move_costs_tm1"][1:, ...]  # [k_max, shuffle_parallel]
    inv_actions = inverse_trajectory["actions"]
    action_costs = inverse_trajectory["action_costs"]

    # Transpose to [shuffle_parallel, k_max, ...] to keep trajectories contiguous
    states = states.transpose((1, 0))
    move_costs = move_costs.transpose((1, 0))
    move_costs_tm1 = move_costs_tm1.transpose((1, 0))
    inv_actions = inv_actions.transpose((1, 0))
    action_costs = action_costs.transpose((1, 0))

    # Tile solve_configs: [shuffle_parallel, ...] -> [shuffle_parallel, k_max, ...]
    solve_configs = xnp.repeat(solve_configs[:, jnp.newaxis], k_max, axis=1)

    # Create indices
    # trajectory_indices: [shuffle_parallel, k_max] -> value p for p in 0..P-1
    trajectory_indices = jnp.broadcast_to(
        jnp.arange(shuffle_parallel, dtype=jnp.int32)[:, jnp.newaxis],
        (shuffle_parallel, k_max),
    )

    # step_indices: [shuffle_parallel, k_max] -> value k for k in 0..K-1
    # For inverse trajectory, step 0 is closest to solved state (distance 0 or 1).
    step_indices = jnp.broadcast_to(
        jnp.arange(k_max, dtype=jnp.int32)[jnp.newaxis, :],
        (shuffle_parallel, k_max),
    )

    # Create parent indices
    # Flattened order is (traj 0, step 0), (traj 0, step 1), ...
    # Parent of step t is step t-1 (closer to solved state in inverse generation).
    # Index I corresponds to (p, t). Parent I-1 corresponds to (p, t-1).
    indices = jnp.arange(k_max * shuffle_parallel, dtype=jnp.int32)
    parent_indices = indices - 1
    # Mask the first element of each trajectory (step 0 has no parent in this batch)
    # reshaped: [shuffle_parallel, k_max]
    parent_indices = parent_indices.reshape(shuffle_parallel, k_max)
    parent_indices = parent_indices.at[:, 0].set(-1)

    return {
        "solve_configs": solve_configs.flatten(),
        "states": states.flatten(),
        "move_costs": move_costs.flatten(),
        "move_costs_tm1": move_costs_tm1.flatten(),
        "actions": inv_actions.flatten(),
        "action_costs": action_costs.flatten(),
        "parent_indices": parent_indices.flatten(),
        "trajectory_indices": trajectory_indices.flatten(),
        "step_indices": step_indices.flatten(),
    }


def create_hindsight_target_shuffled_path(
    puzzle: Puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    key_traj, key_append = jax.random.split(key, 2)
    trajectory = get_random_trajectory(
        puzzle,
        k_max,
        shuffle_parallel,
        key_traj,
        non_backtracking_steps=non_backtracking_steps,
    )

    original_solve_configs = trajectory["solve_configs"]  # [shuffle_parallel, ...]
    states = trajectory["states"]  # [k_max + 1, shuffle_parallel, ...]
    move_costs = trajectory["move_costs"]  # [k_max + 1, shuffle_parallel]
    move_costs_tm1 = trajectory["move_costs_tm1"]  # [k_max + 1, shuffle_parallel]
    actions = trajectory["actions"]  # [k_max, shuffle_parallel]
    action_costs = trajectory["action_costs"]  # [k_max, shuffle_parallel]

    targets = states[-1, ...]  # [shuffle_parallel, ...]
    if include_solved_states:
        states = states[1:, ...]  # [k_max, shuffle_parallel, ...] this is include the last state
    else:
        states = states[:-1, ...]  # [k_max, shuffle_parallel, ...] this is exclude the last state

    solve_configs = puzzle.batched_hindsight_transform(
        original_solve_configs, targets
    )  # [shuffle_parallel, ...]

    if include_solved_states:
        move_costs = move_costs[-1, ...] - move_costs[1:, ...]  # [k_max, shuffle_parallel]
        move_costs_tm1 = move_costs[-1, ...] - move_costs_tm1[1:, ...]  # [k_max, shuffle_parallel]
        actions = jnp.concatenate(
            [
                actions[1:],
                jax.random.randint(
                    key_append, (1, shuffle_parallel), minval=0, maxval=puzzle.action_size
                ),
            ]
        )  # [k_max, shuffle_parallel]
        action_costs = jnp.concatenate(
            [action_costs[1:], jnp.zeros((1, shuffle_parallel))]
        )  # [k_max, shuffle_parallel]
    else:
        move_costs = move_costs[-1, ...] - move_costs[:-1, ...]  # [k_max, shuffle_parallel]
        move_costs_tm1 = move_costs[-1, ...] - move_costs_tm1[:-1, ...]  # [k_max, shuffle_parallel]
        move_costs_tm1 = move_costs_tm1.at[0, ...].set(0.0)

    # Reverse along time axis (axis 0) to order from Target (close) to Start (far)
    states = states[::-1, ...]
    move_costs = move_costs[::-1, ...]
    move_costs_tm1 = move_costs_tm1[::-1, ...]
    actions = actions[::-1, ...]
    action_costs = action_costs[::-1, ...]

    # Transpose to [shuffle_parallel, k_max, ...]
    states = states.transpose((1, 0))
    move_costs = move_costs.transpose((1, 0))
    move_costs_tm1 = move_costs_tm1.transpose((1, 0))
    actions = actions.transpose((1, 0))
    action_costs = action_costs.transpose((1, 0))

    # Tile solve_configs: [shuffle_parallel, ...] -> [shuffle_parallel, k_max, ...]
    solve_configs = xnp.repeat(solve_configs[:, jnp.newaxis], k_max, axis=1)

    # Create indices
    trajectory_indices = jnp.broadcast_to(
        jnp.arange(shuffle_parallel, dtype=jnp.int32)[:, jnp.newaxis],
        (shuffle_parallel, k_max),
    )
    step_indices = jnp.broadcast_to(
        jnp.arange(k_max, dtype=jnp.int32)[jnp.newaxis, :],
        (shuffle_parallel, k_max),
    )

    # Parent indices:
    # After reversal, index 0 is Target (or closest to it). Index 1 is next step away.
    # Parent of s_i is s_{i-1} (the one closer to target in this reversed view? No.)
    # Original path: s_start -> ... -> s_target.
    # Reversed: s_target -> ... -> s_start.
    # In learning (Bellman), value at s_{t} depends on s_{t+1} (closer to target).
    # Here, s_0 (Target) has value. s_1 (Target-1) depends on s_0.
    # So parent of s_1 is s_0.
    # Parent of index t is t-1.
    indices = jnp.arange(k_max * shuffle_parallel, dtype=jnp.int32)
    parent_indices = indices - 1
    parent_indices = parent_indices.reshape(shuffle_parallel, k_max)
    parent_indices = parent_indices.at[:, 0].set(-1)

    return {
        "solve_configs": solve_configs.flatten(),
        "states": states.flatten(),
        "move_costs": move_costs.flatten(),
        "move_costs_tm1": move_costs_tm1.flatten(),
        "actions": actions.flatten(),
        "action_costs": action_costs.flatten(),
        "parent_indices": parent_indices.flatten(),
        "trajectory_indices": trajectory_indices.flatten(),
        "step_indices": step_indices.flatten(),
    }


def create_hindsight_target_triangular_shuffled_path(
    puzzle: Puzzle,
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    key, subkey = jax.random.split(key)
    trajectory = get_random_trajectory(
        puzzle,
        k_max,
        shuffle_parallel,
        subkey,
        non_backtracking_steps=non_backtracking_steps,
    )

    original_solve_configs = trajectory["solve_configs"]  # [P, ...]
    states = trajectory["states"]  # [L+1, P, ...]
    move_costs = trajectory["move_costs"]  # [L+1, P]
    move_costs_tm1 = trajectory["move_costs_tm1"]  # [L+1, P]
    actions = trajectory["actions"]  # [L, P]
    action_costs = trajectory["action_costs"]  # [L, P]

    # Uniformly sample path length `k` and then a valid starting point `i`.
    # This ensures that the distribution of path lengths `k` is uniform.
    key, key_k, key_i = jax.random.split(key, 3)

    # 1. Sample path lengths `k` uniformly from [0, L] or [1, L]
    minval = 0 if include_solved_states else 1
    k = jax.random.randint(
        key_k,
        shape=(k_max, shuffle_parallel),
        minval=minval,
        maxval=k_max + 1,
    )  # [L, P] -> [k_max, shuffle_parallel]

    # 2. For each `k`, sample start_index `i` uniformly from [0, L-k]
    random_floats = jax.random.uniform(key_i, shape=(k_max, shuffle_parallel))  # [L, P]
    max_start_idx = k_max - k
    start_indices = (random_floats * (max_start_idx + 1)).astype(jnp.int32)  # [L, P]

    # 3. The target_index `j` is simply i + k
    target_indices = start_indices + k  # [L, P]

    parallel_indices = jnp.tile(jnp.arange(shuffle_parallel)[None, :], (k_max, 1))  # [L, P]

    # Gather data using the sampled indices
    start_states = states[start_indices, parallel_indices]
    target_states = states[target_indices, parallel_indices]

    start_move_costs = move_costs[start_indices, parallel_indices]
    target_move_costs = move_costs[target_indices, parallel_indices]
    start_move_costs_tm1 = move_costs_tm1[start_indices, parallel_indices]
    final_move_costs = target_move_costs - start_move_costs
    final_move_costs_tm1 = target_move_costs - start_move_costs_tm1
    final_move_costs_tm1 = jnp.where(start_indices == 0, 0.0, final_move_costs_tm1)

    # Handle boundary condition: clamp start_indices to prevent out-of-bounds access
    # actions array has size [L], so valid indices are [0, L-1]
    clamped_start_indices = jnp.clip(start_indices, 0, k_max - 1)
    final_actions = actions[clamped_start_indices, parallel_indices]
    final_action_costs = action_costs[clamped_start_indices, parallel_indices]

    # For cases where start_state == target_state (k=0), set action cost to 0
    # This represents reaching the goal state (no further action needed)
    is_goal_state = (k == 0) & include_solved_states
    final_action_costs = jnp.where(is_goal_state, 0.0, final_action_costs)

    # Apply hindsight transform
    tiled_solve_configs = xnp.repeat(original_solve_configs[jnp.newaxis, ...], k_max, axis=0)
    flat_tiled_sc = tiled_solve_configs.flatten()
    flat_target_states = target_states.flatten()
    final_solve_configs = puzzle.batched_hindsight_transform(
        flat_tiled_sc, flat_target_states
    ).reshape((k_max, shuffle_parallel, -1))

    # Transpose everything to [shuffle_parallel, k_max, ...]
    # k (distance) is used for sorting
    # current shape: [k_max, shuffle_parallel]
    k_transposed = k.transpose((1, 0))  # [P, k_max]

    # Sort indices based on k (ascending: small distance first -> close to solved)
    sort_indices = jnp.argsort(k_transposed, axis=1)  # [P, k_max]

    def _sort_and_transpose(arr):
        # arr is [k_max, shuffle_parallel, ...]
        # Transpose first: [shuffle_parallel, k_max, ...]
        arr_t = jnp.swapaxes(arr, 0, 1)
        # Apply sort
        # Expand sort_indices to match rank of arr_t for take_along_axis
        indices = sort_indices
        while indices.ndim < arr_t.ndim:
            indices = indices[..., jnp.newaxis]
        return jnp.take_along_axis(arr_t, indices, axis=1)

    final_solve_configs = _sort_and_transpose(final_solve_configs)
    final_start_states = _sort_and_transpose(start_states)
    final_move_costs = _sort_and_transpose(final_move_costs)
    final_move_costs_tm1 = _sort_and_transpose(final_move_costs_tm1)
    final_actions = _sort_and_transpose(final_actions)
    final_action_costs = _sort_and_transpose(final_action_costs)

    # step_indices is the sorted k (distance)
    step_indices = jnp.take_along_axis(k_transposed, sort_indices, axis=1)

    # trajectory_indices
    trajectory_indices = jnp.broadcast_to(
        jnp.arange(shuffle_parallel, dtype=jnp.int32)[:, jnp.newaxis],
        (shuffle_parallel, k_max),
    )

    # Triangular sampling produces independent samples, no valid parent structure
    parent_indices = jnp.full((shuffle_parallel, k_max), -1, dtype=jnp.int32)

    return {
        "solve_configs": final_solve_configs.flatten(),
        "states": final_start_states.flatten(),
        "move_costs": final_move_costs.flatten(),
        "move_costs_tm1": final_move_costs_tm1.flatten(),
        "actions": final_actions.flatten(),
        "action_costs": final_action_costs.flatten(),
        "parent_indices": parent_indices.flatten(),
        "trajectory_indices": trajectory_indices.flatten(),
        "step_indices": step_indices.flatten(),
    }
