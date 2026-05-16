import jax.numpy as jnp
from xtructure import FieldDescriptor, xtructure_dataclass

from heuristic.neuralheuristic.target_dataset_builder import _compute_diffusion_distance


@xtructure_dataclass
class DummyState:
    val: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass
class DummySolveConfig:
    target: FieldDescriptor.scalar(dtype=jnp.int32)


def _linear_chain_diffusion_inputs(k_max: int, shuffle_parallel: int):
    step_indices = jnp.broadcast_to(
        jnp.arange(k_max, dtype=jnp.int32)[jnp.newaxis, :],
        (shuffle_parallel, k_max),
    )
    flat_size = k_max * shuffle_parallel
    flat_indices = jnp.arange(flat_size, dtype=jnp.int32).reshape(shuffle_parallel, k_max)
    parent_indices = (flat_indices - 1).at[:, 0].set(-1).reshape(-1)

    move_costs = step_indices.astype(jnp.float32).reshape(-1)
    return {
        "solve_configs": DummySolveConfig(target=jnp.zeros((flat_size,), dtype=jnp.int32)),
        "states": DummyState(val=step_indices.reshape(-1)),
        "move_costs": move_costs,
        "action_costs": jnp.ones_like(move_costs),
        "parent_indices": parent_indices,
        "is_solved": step_indices.reshape(-1) == 0,
    }


def analyze_diffusion_drop(
    k_max: int = 10,
    shuffle_parallel: int = 10000,
    print_summary: bool = True,
):
    path_data = _linear_chain_diffusion_inputs(k_max, shuffle_parallel)

    @xtructure_dataclass
    class SolveConfigsAndStates:
        solveconfigs: FieldDescriptor.scalar(dtype=DummySolveConfig)
        states: FieldDescriptor.scalar(dtype=DummyState)

    target_heuristic = _compute_diffusion_distance(
        path_data["solve_configs"],
        path_data["states"],
        path_data["is_solved"],
        path_data["move_costs"],
        path_data["action_costs"],
        path_data["parent_indices"],
        SolveConfigsAndStates,
        k_max=k_max,
    )

    h_init = path_data["move_costs"].reshape(shuffle_parallel, k_max).T
    h_final = target_heuristic.reshape(shuffle_parallel, k_max).T

    if print_summary:
        print("Step | Mean Init H | Mean Final H | Mean Drop")
        print("-----|-------------|--------------|----------")
        for k in range(k_max):
            mean_init = jnp.mean(h_init[k])
            mean_final = jnp.mean(h_final[k])
            drop = mean_init - mean_final
            print(f"{k: 4d} | {mean_init: 11.4f} | {mean_final: 12.4f} | {drop: 9.4f}")

    return {
        "target_heuristic": target_heuristic,
        "move_costs": path_data["move_costs"],
        "h_init": h_init,
        "h_final": h_final,
    }


def test_diffusion_distance_preserves_linear_chain_cost_bounds():
    result = analyze_diffusion_drop(k_max=4, shuffle_parallel=3, print_summary=False)

    target_heuristic = result["target_heuristic"]
    move_costs = result["move_costs"]

    assert target_heuristic.shape == move_costs.shape
    assert bool(jnp.all(jnp.isfinite(target_heuristic)))
    assert bool(jnp.all(target_heuristic <= move_costs))
    assert bool(jnp.all(target_heuristic[::4] == 0.0))


if __name__ == "__main__":
    analyze_diffusion_drop()
