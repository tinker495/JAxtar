from functools import partial
from typing import Annotated, Any, Callable

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

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


def _infer_neighbor_heuristics(
    model: DistanceModel | DistanceHLGModel,
    inputs: jax.Array,
    solve_configs: Xtructurable,
    neighbors: Xtructurable,
    SolveConfigsAndStates: type[Xtructurable],
    infer: Callable,
) -> tuple[None, jax.Array]:
    if not model.supports_row_independent_inference:
        return jax.lax.scan(infer, None, inputs)

    actions, rows = inputs.shape[:2]
    total = actions * rows
    pairs = SolveConfigsAndStates(
        solveconfigs=jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (actions, *leaf.shape)).reshape(
                (total, *leaf.shape[1:])
            ),
            solve_configs,
        ),
        states=neighbors.reshape((-1,)),
    )
    mask, indices, inverse = xnp.unique_mask(pairs, return_index=True, return_inverse=True)
    count = jnp.sum(mask, dtype=jnp.int32)
    keys = pairs.uint32ed
    representatives = indices[jnp.clip(inverse, 0, total - 1)]
    safe_representatives = jnp.clip(representatives, 0, total - 1)
    flat_inputs = inputs.reshape((total, *inputs.shape[2:]))
    comparison_inputs = flat_inputs.reshape((total, -1))
    # Keep sub-byte rows byte-aligned only in the comparison copy.
    comparison_inputs = jnp.pad(comparison_inputs, ((0, 0), (0, (-comparison_inputs.shape[1]) % 8)))
    input_bytes = comparison_inputs.view(jnp.uint8)
    # Custom preprocessing can give equal keys different input rows.
    mapping_valid = (
        (count >= 1)
        & (count <= total)
        & jnp.all(
            (inverse >= 0) & (inverse < count) & (representatives >= 0) & (representatives < total)
        )
        & jnp.all(keys == keys[safe_representatives])
        & jnp.all(input_bytes == input_bytes[safe_representatives])
    )
    identity = jnp.arange(total, dtype=jnp.int32)
    count = jnp.where(mapping_valid, count, total)
    indices = jnp.where(mapping_valid, indices, identity)
    inverse = jnp.where(mapping_valid, inverse, identity)
    prediction_spec = jax.eval_shape(infer, None, inputs[0])[1]

    def condition(carry):
        index, _ = carry
        return index * rows < count

    def body(carry):
        index, result = carry
        selected = jax.lax.dynamic_slice(indices, (index * rows,), (rows,))
        _, prediction = infer(None, flat_inputs[jnp.clip(selected, 0, total - 1)])
        result = jax.lax.dynamic_update_slice(result, prediction.reshape((rows,)), (index * rows,))
        return index + 1, result

    _, result = jax.lax.while_loop(
        condition, body, (jnp.int32(0), jnp.zeros((total,), dtype=prediction_spec.dtype))
    )
    return None, result[inverse].reshape((actions, *prediction_spec.shape))


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStates: type[Xtructurable],
    heuristic_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    temperature: float = 1.0 / 3.0,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))

    def get_minibatched_datasets(_, vals):
        solve_configs, states = vals
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

        _, heur = _infer_neighbor_heuristics(
            heuristic_model,
            preproc_neighbors,
            solve_configs,
            neighbors,
            SolveConfigsAndStates,
            compute_heur_for_action,
        )  # [action_size, batch_size]
        backup = heur + cost  # [action_size, batch_size]
        target_heuristic = jnp.min(jnp.maximum(backup, cost), axis=0)
        target_heuristic = jnp.where(
            solved, 0.0, target_heuristic
        )  # if the puzzle is already solved, the heuristic is 0

        return None, (
            solve_configs,
            states,
            target_heuristic,
            solved,
        )

    _, (solve_configs, states, target_heuristic, solved) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_states),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_heuristic = target_heuristic.reshape((-1,))
    is_solved = solved.reshape((-1,))

    # Cap bootstrap targets with deduplicated trajectory (diffusion) distances to
    # suppress overestimation; strictly tighter than the old min(target, move_costs).
    diffusion_heuristic = _compute_diffusion_distance(
        solve_configs,
        states,
        is_solved,
        shuffled_path["move_costs"],
        shuffled_path["action_costs"].reshape((-1,)),
        shuffled_path["parent_indices"],
        SolveConfigsAndStates,
        k_max,
    )
    target_heuristic = jnp.minimum(target_heuristic, diffusion_heuristic)

    return {
        "solve_config": solve_configs,
        "state": states,
        "distance": target_heuristic,
    }


def _compute_diffusion_distance(
    solve_configs: chex.Array,
    states: chex.Array,
    is_solved: chex.Array,
    move_costs: chex.Array,
    action_costs: chex.Array,
    parent_indices: chex.Array,
    SolveConfigsAndStates: type[Xtructurable],
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
        is_solved=is_solved,
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
    SolveConfigsAndStates: type[Xtructurable],
    heuristic_model: DistanceModel | DistanceHLGModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
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

    is_solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True).reshape(
        (-1,)
    )

    target_heuristic = _compute_diffusion_distance(
        solve_configs,
        states,
        is_solved,
        move_costs,
        action_costs,
        parent_indices,
        SolveConfigsAndStates,
        k_max,
    )

    return {
        "solve_config": solve_configs,
        "state": states,
        "distance": target_heuristic,
    }


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
        include_action_costs=True,
        non_backtracking_steps=non_backtracking_steps,
        create_hindsight_target_shuffled_path=create_hindsight_target_shuffled_path,
        create_hindsight_target_triangular_shuffled_path=(
            create_hindsight_target_triangular_shuffled_path
        ),
        create_target_shuffled_path=create_target_shuffled_path,
    )

    @xtructure_dataclass
    class SolveConfigsAndStates:
        solveconfigs: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.SolveConfig)]
        states: Annotated[Xtructurable, FieldDescriptor.scalar(dtype=puzzle.State)]

    base_get_datasets = partial(
        _get_datasets,
        puzzle,
        preproc_fn,
        SolveConfigsAndStates,
        heuristic_model,
        nn_minibatch_size,
        k_max=k_max,
        temperature=temperature,
    )
    diffusion_get_datasets = partial(
        _get_datasets_with_diffusion_distance,
        puzzle,
        preproc_fn,
        SolveConfigsAndStates,
        heuristic_model,
        nn_minibatch_size,
        k_max=k_max,
        temperature=temperature,
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
