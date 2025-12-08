import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.stars.search_base import Current, SearchResult


def vmapping_init_target(
    puzzle: Puzzle, vmap_size: int, start_state_seeds: list[int]
) -> tuple[Puzzle.SolveConfig, Puzzle.State]:
    start_state_seed = start_state_seeds[0]
    solve_configs, states = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
    solve_configs = xnp.tile(solve_configs[jnp.newaxis, ...], (vmap_size, 1))
    states = xnp.tile(states[jnp.newaxis, ...], (vmap_size, 1))

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_solve_configs, new_states = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
            states = states.at[i + 1].set(new_states)
            solve_configs = solve_configs.at[i + 1].set(new_solve_configs)
    return solve_configs, states


def vmapping_search(
    puzzle: Puzzle,
    star_fn: callable,
    vmap_size: int,
    show_compile_time: bool = False,
):
    """
    Vmap the search function over the batch dimension.
    """

    empty_states = puzzle.State.default((vmap_size,))
    empty_solve_configs = puzzle.SolveConfig.default((vmap_size,))
    vmapped_star = jax.jit(jax.vmap(star_fn, in_axes=(0, 0)))
    if show_compile_time:
        print("initializing vmapped jit")
        start = time.time()
    vmapped_star(empty_solve_configs, empty_states)
    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")
    return vmapped_star


def vmapping_get_state(search_result: SearchResult, idx: Current):
    return jax.vmap(SearchResult.get_state, in_axes=(0, 0))(search_result, idx)
