import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from heuristic.tsp_heuristic import TSPHeuristic
from puzzle.tsp import TSP


def old_baseline(solve_config: TSP.SolveConfig, state: TSP.State):
    visited_mask = state.unpacking().mask
    inv_mask = 1 - visited_mask
    dmat = solve_config.distance_matrix
    masked = dmat * inv_mask[None, :] * inv_mask[:, None]
    return jnp.mean(jnp.sum(masked, axis=1))


def run_benchmark(num_instances: int = 100, size: int = 10):
    h = []
    h_old = []

    puzzle = TSP(size=size)
    heuristic = TSPHeuristic(puzzle)

    key = jax.random.PRNGKey(42)
    for i in tqdm(range(num_instances), desc="Benchmark"):
        key, subkey = jax.random.split(key)
        solve_config = puzzle.get_solve_config(key=subkey)
        state = puzzle.get_initial_state(solve_config, key=subkey)

        h.append(float(heuristic.distance(solve_config, state)))
        h_old.append(float(old_baseline(solve_config, state)))

    h = jnp.array(h)
    h_old = jnp.array(h_old)

    improvement = jnp.mean(h - h_old)
    ratio = jnp.mean(h / jnp.maximum(h_old, 1e-8))

    print("Average baseline heuristic:", float(jnp.mean(h_old)))
    print("Average new heuristic:", float(jnp.mean(h)))
    print("Average absolute improvement:", float(improvement))
    print("Average ratio new / old:", float(ratio))


if __name__ == "__main__":
    start = time.time()
    run_benchmark()
    print("Elapsed time: {:.2f}s".format(time.time() - start))