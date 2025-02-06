import time

import jax
import jax.numpy as jnp

from JAxtar.hash import HashTable
from puzzle.puzzle_base import Puzzle


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def vmapping_search(
    puzzle: Puzzle,
    search_result_build: callable,
    star_fn: callable,
    vmap_size: int,
    batch_size: int,
    show_compile_time: bool = False,
):
    """
    Vmap the search function over the batch dimension.
    """

    inital_search_result = search_result_build()
    empty_states = puzzle.State.default()[jnp.newaxis, ...]
    empty_states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), empty_states
    )
    empty_states_batched, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(empty_states)
    vmapped_star = jax.jit(jax.vmap(star_fn, in_axes=(None, 0, 0, 0)))
    if show_compile_time:
        print("initializing vmapped jit")
        start = time.time()
    vmapped_star(inital_search_result, empty_states_batched, filled, empty_states)
    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")
    return vmapped_star
