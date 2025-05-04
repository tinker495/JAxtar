from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from heuristic.neuralheuristic.neuralheuristic_base import (
    NeuralHeuristicBase as NeuralHeuristic,
)
from heuristic.neuralheuristic.replay import BUFFER_STATE_TYPE, BUFFER_TYPE
from JAxtar.astar import astar_builder
from JAxtar.search_base import (
    HASH_POINT_DTYPE,
    HASH_TABLE_IDX_DTYPE,
    Current,
    Parent,
    SearchResult,
)
from puzzle.puzzle_base import Puzzle


def regression_replay_trainer_builder(
    buffer: BUFFER_TYPE,
    train_steps: int,
    heuristic_model: NeuralHeuristic,
    optimizer: optax.GradientTransformation,
) -> Callable:
    def regression_loss(
        heuristic_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        target_heuristic: chex.Array,
    ):
        current_heuristic, variable_updates = heuristic_model(
            heuristic_params, states, training=True, mutable=["batch_stats"]
        )
        heuristic_params["batch_stats"] = variable_updates["batch_stats"]
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        loss = jnp.mean(jnp.square(diff))
        return loss, (heuristic_params, diff)

    def regression(
        key: chex.PRNGKey,
        buffer_state: BUFFER_STATE_TYPE,
        heuristic_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """

        def train_loop(carry, _):
            heuristic_params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            sample = buffer.sample(buffer_state, subkey)
            states, target_heuristic = (
                sample.experience.first["obs"],
                sample.experience.first["distance"],
            )
            (loss, (heuristic_params, diff)), grads = jax.value_and_grad(
                regression_loss, has_aux=True
            )(
                heuristic_params,
                states,
                target_heuristic,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude))
            return (heuristic_params, opt_state, key), (
                loss,
                diff,
                target_heuristic,
                grad_magnitude_mean,
            )

        (heuristic_params, opt_state, key), (
            losses,
            diffs,
            target_heuristics,
            grad_magnitude_means,
        ) = jax.lax.scan(
            train_loop,
            (heuristic_params, opt_state, key),
            None,
            length=train_steps,
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(jnp.array(grad_magnitude_means))
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.mean(jnp.abs(x)), jax.tree_util.tree_leaves(heuristic_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.array(weights_magnitude))
        sampled_target_heuristics = jnp.reshape(target_heuristics, (-1,))
        return (
            heuristic_params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_target_heuristics,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    return jax.jit(regression)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_top_k_branchs_paths(
    search_result: SearchResult, top_k: int = 1000, max_depth: int = 100
) -> tuple[Current, Parent, chex.Array]:
    """
    Get all branch paths from the solved state.
    All closed states are pseudo-optimal (they are optimal when the heuristic is admissible).
    This allows us to collect ground truth heuristic values from these states.
    If the heuristic is not admissible, the optimality of these paths cannot be guaranteed.
    All closed states are generally close to optimal paths, even if the heuristic is not perfectly admissible.
    """
    closed_masks = jnp.isfinite(search_result.cost)  # [size_table, n_table]
    closed_masks = closed_masks.at[-1, :].set(False)  # mask the last row as a dummy node
    no_parented_masks = (
        jnp.ones_like(closed_masks, dtype=jnp.bool_)
        .at[search_result.parent.index, search_result.parent.table_index]
        .set(False)
    )
    leaf_mask = jnp.logical_and(closed_masks, no_parented_masks)
    masked_cost = jnp.where(leaf_mask, search_result.cost, 0)  # [size_table, n_table]
    flattened_cost = jnp.reshape(masked_cost, (-1,))  # [size_table * n_table]
    flattened_idxs = jnp.stack(
        jnp.unravel_index(jnp.arange(search_result.cost.size), search_result.cost.shape), axis=1
    ).astype(jnp.uint32)
    flattend_sort_indices = jnp.argsort(flattened_cost, descending=True)
    sorted_idxs = flattened_idxs[flattend_sort_indices]
    sorted_cost = flattened_cost[flattend_sort_indices]
    sorted_mask = leaf_mask[sorted_idxs[:, 0], sorted_idxs[:, 1]]
    sorted_leaf_nodes = Current(
        index=sorted_idxs[:, 0].astype(HASH_POINT_DTYPE),
        table_index=sorted_idxs[:, 1].astype(HASH_TABLE_IDX_DTYPE),
        cost=sorted_cost,
    )

    paths = []
    top_k_leaf_nodes = sorted_leaf_nodes[:top_k]
    top_k_mask = sorted_mask[:top_k]
    paths, path_masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
        search_result, top_k_leaf_nodes, top_k_mask, max_depth
    )
    return top_k_leaf_nodes, paths, path_masks


def get_one_solved_branch_samples(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    astar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, jax.tree_util.PyTreeDef], SearchResult],
    max_depth: int,
    sample_ratio: float,
    use_topk_branch: bool,
    heuristic_params: jax.tree_util.PyTreeDef,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = astar_fn(solve_config, initial_state, heuristic_params)
    batch_size = filled.shape[0]

    if use_topk_branch:
        leafs, paths, masks = get_top_k_branchs_paths(
            search_result, top_k=batch_size, max_depth=max_depth - 1
        )
    else:
        paths, masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
            search_result, leafs, filled, max_depth - 1
        )

    # leafs: [topk_branch_size, ...]
    # paths: [topk_branch_size, max_depth - 1, ...]
    # masks: [topk_branch_size, max_depth - 1]
    masks = jnp.concatenate((jnp.ones(masks.shape[0], dtype=bool)[:, jnp.newaxis], masks), axis=1)

    leaf_states = search_result.get_state(leafs)
    leaf_costs = search_result.get_cost(leafs)
    # leaf_states: [topk_branch_size, ...], leaf_costs: [topk_branch_size]
    leaf_solve_configs = puzzle.batched_hindsight_transform(leaf_states)  # states -> solve_configs
    # leaf_solve_configs: [topk_branch_size, ...]

    max_leaf_costs = jnp.max(leaf_costs)
    leaf_mask = leaf_costs > max_leaf_costs * (1.0 - sample_ratio)  # batch_size
    masks = jnp.where(leaf_mask[:, jnp.newaxis], masks, False)  # [topk_branch_size, max_depth]

    path_states = search_result.get_state(paths)
    path_states = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=1), leaf_states[:, jnp.newaxis], path_states
    )
    path_costs = search_result.get_cost(paths)
    path_costs = jnp.concatenate((leaf_costs[:, jnp.newaxis], path_costs), axis=1)
    # path_states: [topk_branch_size, max_depth, ...], path_costs: [topk_branch_size, max_depth]

    raw_costs = leaf_costs[:, jnp.newaxis] - path_costs
    raw_costs = jnp.where(masks, raw_costs, leaf_costs[:, jnp.newaxis])
    # raw_costs: [topk_branch_size, max_depth] , [[1, 2, 3, 4, 5, ...], [1, 2, 3, 4, 5, ...], ...]
    # example: [1, 2, 3, 4, 5, 6, 7, ... , 20, 21, 22, 22, 22, 22, 22, ...]

    incr_costs = raw_costs[:, 1:] - raw_costs[:, :-1]
    incr_costs = jnp.concatenate((raw_costs[:, 0, jnp.newaxis], incr_costs), axis=1)
    # incr_costs: [topk_branch_size, max_depth] , [[1, 1, 1, 1, 1, ...], [1, 1, 1, 1, 1, ...], ...]

    is_solved = jax.vmap(jax.vmap(puzzle.is_solved, in_axes=(None, 0)))(
        leaf_solve_configs, path_states
    )
    # is_solved: [topk_branch_size, max_depth]
    incr_costs = jnp.where(is_solved, 0, incr_costs)
    # incr_costs: [topk_branch_size, max_depth] , [[0, 0, 1, 1, 1, ...], [0, 0, 0, 1, 1, ...], ...]

    true_costs = jnp.cumsum(incr_costs, axis=1)
    # true_costs: [topk_branch_size, max_depth] , [[0, 0, 1, 2, 3, ...], [0, 0, 0, 1, 2, ...], ...]
    # This represents the cumulative cost from each state to the leaf node
    shifted_is_solved = jnp.concatenate(
        (is_solved[:, 1:], jnp.zeros((is_solved.shape[0], 1), dtype=jnp.bool_)), axis=1
    )
    masks = jnp.logical_and(masks, ~shifted_is_solved)
    # masks: [topk_branch_size, max_depth] ,
    # [[False, False, True, True, True, ...], [False, False, False, True, True, ...], ...]

    preprocessed_data = jax.vmap(jax.vmap(heuristic.pre_process, in_axes=(None, 0)))(
        leaf_solve_configs, path_states
    )
    # preprocessed_data: [topk_branch_size, max_depth, ...]
    flattened_preprocessed_data = jnp.reshape(
        preprocessed_data, (batch_size * max_depth, *preprocessed_data.shape[2:])
    )
    flattened_true_costs = jnp.reshape(true_costs, (batch_size * max_depth,))
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return flattened_preprocessed_data, flattened_true_costs, flattened_masks, search_result.solved


def wbsdai_dataset_builder(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    buffer: BUFFER_TYPE,
    add_batch_size: int = 8192,
    search_batch_size: int = 8192,
    max_nodes: int = int(2e6),
    cost_weight: float = 1.0 - 1e-3,
    max_depth: int = 100,
    sample_ratio: float = 0.3,
    use_topk_branch: bool = False,
) -> Callable:
    """
    wbsdai_builder is a function that returns a partial function of wbsdai.
    """

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        search_batch_size,
        max_nodes,
        cost_weight,
        use_heuristic_params=True,
        export_last_pops=True,
    )

    jitted_get_one_solved_branch_samples = jax.jit(
        partial(
            get_one_solved_branch_samples,
            puzzle,
            heuristic,
            astar_fn,
            max_depth,
            sample_ratio,
            use_topk_branch,
        )
    )

    def get_wbsdai_dataset(
        heuristic_params: jax.tree_util.PyTreeDef,
        buffer_state: BUFFER_STATE_TYPE,
        key: chex.PRNGKey,
    ):
        run = True
        search_count = 0
        solved_count = 0
        while run:
            preprocessed_datas = []
            true_costs = []
            data_len = 0
            while data_len < add_batch_size:
                key, subkey = jax.random.split(key)
                preprocessed_data, true_cost, masks, solved = jitted_get_one_solved_branch_samples(
                    heuristic_params, subkey
                )
                preprocessed_data = preprocessed_data[masks].astype(jnp.float32)
                true_cost = true_cost[masks].astype(jnp.float32)
                size = jnp.sum(masks)
                preprocessed_datas.append(preprocessed_data)
                true_costs.append(true_cost)
                data_len += size
                search_count += 1
                solved_count += solved
            preprocessed_data = jnp.concatenate(preprocessed_datas, axis=0)
            true_costs = jnp.concatenate(true_costs, axis=0)
            split_len = data_len // add_batch_size

            for i in range(split_len):
                timestep = {
                    "obs": preprocessed_data[i * add_batch_size : (i + 1) * add_batch_size],
                    "distance": true_costs[i * add_batch_size : (i + 1) * add_batch_size],
                }
                buffer_state = buffer.add(buffer_state, timestep)

            run = not buffer.can_sample(
                buffer_state
            )  # get datas until the buffer is enough to sample

        return buffer_state, search_count, solved_count, key

    return get_wbsdai_dataset
