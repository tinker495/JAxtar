import time
from typing import Callable

import jax
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_with_example, jit_vmapped_search, warmup_with_example

_VMAPPED_STAR_CACHE: dict[tuple[int, int], tuple[Callable, Callable]] = {}


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
    star_fn: Callable,
    vmap_size: int,
    show_compile_time: bool = False,
):
    """
    Vmap the search function over the batch dimension.
    """

    cache_key = (id(star_fn), vmap_size)
    if cache_key in _VMAPPED_STAR_CACHE:
        return _VMAPPED_STAR_CACHE[cache_key][1]

    empty_states = puzzle.State.default((vmap_size,))
    empty_solve_configs = puzzle.SolveConfig.default((vmap_size,))
    vmapped_star = jit_vmapped_search(star_fn)
    start = None
    if show_compile_time:
        print("initializing vmapped jit")
        start = time.time()
    compile_with_example(vmapped_star, empty_solve_configs, empty_states)
    if start is not None:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")
    warmup_with_example(vmapped_star, empty_solve_configs, empty_states)
    # The batched JIT can close over the body alone. Keep the scalar's cache identity alive.
    _VMAPPED_STAR_CACHE[cache_key] = (star_fn, vmapped_star)
    return vmapped_star
