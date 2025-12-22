import jax
import jax.numpy as jnp
from xtructure import FieldDescriptor, xtructure_dataclass

from heuristic.neuralheuristic.heuristic_train import _compute_diffusion_distance
from train_util.sampling import create_target_shuffled_path


class DummyPuzzle:
    def __init__(self):
        self.action_size = 4
        self.fixed_target = False

        @xtructure_dataclass
        class State:
            val: FieldDescriptor.scalar(dtype=jnp.int32)

        @xtructure_dataclass
        class SolveConfig:
            target: FieldDescriptor.scalar(dtype=jnp.int32)

        self.State = State
        self.SolveConfig = SolveConfig

    def get_inits(self, key):
        return self.SolveConfig(target=jnp.int32(0)), self.State(val=jnp.int32(0))

    def solve_config_to_state_transform(self, solve_config, state):
        return state

    def batched_get_neighbours(self, solve_configs, states, filleds, multi_solve_config=True):
        # Explicit type conversion to handle tracers correctly if needed, though JAX usually handles it.
        # The issue 'Var' and 'int' suggests states.val is a Tracer that doesn't support + int?
        # That's weird for JAX. It usually means we need jnp.add.

        val = states.val

        n_vals = [val + 1, jnp.maximum(val - 1, 0), val, val + 2]
        neighbor_vals = jnp.stack(n_vals)
        neighbor_costs = jnp.ones_like(neighbor_vals, dtype=jnp.float32)

        return self.State(val=neighbor_vals), neighbor_costs

    def batched_get_inverse_neighbours(
        self, solve_configs, states, filleds, multi_solve_config=True
    ):
        return self.batched_get_neighbours(solve_configs, states, filleds, multi_solve_config)

    def batched_hindsight_transform(self, solve_configs, targets):
        return solve_configs


def analyze_diffusion_drop():
    puzzle = DummyPuzzle()
    k_max = 10
    shuffle_parallel = 10000
    key = jax.random.PRNGKey(42)

    path_data = create_target_shuffled_path(
        puzzle,
        k_max,
        shuffle_parallel,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=3,
    )

    solve_configs = path_data["solve_configs"]
    states = path_data["states"]
    move_costs = path_data["move_costs"]
    action_costs = path_data["action_costs"]
    parent_indices = path_data["parent_indices"]

    @xtructure_dataclass
    class SolveConfigsAndStates:
        solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
        states: FieldDescriptor.scalar(dtype=puzzle.State)

    # Use JIT to handle JAX tracing correctly
    @jax.jit
    def run_diffusion(sc, st, mc, ac, pi):
        return _compute_diffusion_distance(sc, st, mc, ac, pi, SolveConfigsAndStates, k_max=k_max)

    target_heuristic = run_diffusion(
        solve_configs, states, move_costs, action_costs, parent_indices
    )

    # Reshape back to (k_max, parallel)
    # The sampling output 'move_costs' is [k_max, parallel] flattened.
    # Note: create_target_shuffled_path uses tile(..., (k_max, 1)) then flatten.
    # So reshaped indices [k, p] correspond to step k of path p.

    h_init = move_costs.reshape(k_max, shuffle_parallel)
    h_final = target_heuristic.reshape(k_max, shuffle_parallel)

    print("Step | Mean Init H | Mean Final H | Mean Drop")
    print("-----|-------------|--------------|----------")
    for k in range(k_max):
        mean_init = jnp.mean(h_init[k])
        mean_final = jnp.mean(h_final[k])
        drop = mean_init - mean_final
        print(f"{k:4d} | {mean_init:11.4f} | {mean_final:12.4f} | {drop:9.4f}")


if __name__ == "__main__":
    analyze_diffusion_drop()
