import time

import jax
import xtructure.numpy as xnp
from puxle import Puzzle

_VMAPPED_STAR_CACHE: dict[tuple[int, int], callable] = {}


def vmapping_init_target(
    puzzle: Puzzle, vmap_size: int, start_state_seeds: list[int]
) -> tuple[Puzzle.SolveConfig, Puzzle.State]:
    init_rows = [
        puzzle.get_inits(jax.random.PRNGKey(seed)) for seed in start_state_seeds[:vmap_size]
    ]
    init_rows.extend([init_rows[0]] * (vmap_size - len(init_rows)))
    solve_configs, states = zip(*init_rows)
    return xnp.stack(solve_configs, axis=0), xnp.stack(states, axis=0)


def vmapping_search(
    puzzle: Puzzle,
    star_fn: callable,
    vmap_size: int,
    show_compile_time: bool = False,
):
    """
    Vmap the search function over the batch dimension.
    """

    cache_key = (id(star_fn), vmap_size)
    cached = _VMAPPED_STAR_CACHE.get(cache_key)
    if cached is not None:
        return cached

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
    _VMAPPED_STAR_CACHE[cache_key] = vmapped_star
    return vmapped_star
