import time
from itertools import islice

import jax
import jax.numpy as jnp

from JAxtar.search_base import HashTableIdx_HeapValue, SearchResult
from JAxtar.util import set_tree
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


def vmapping_init_target(puzzle: Puzzle, vmap_size: int, start_state_seeds: list[int]):
    start_state_seed = start_state_seeds[0]
    states, targets = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), states[jnp.newaxis, ...]
    )
    targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), targets[jnp.newaxis, ...]
    )

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_state, new_target = puzzle.get_init_target_state_pair(
                jax.random.PRNGKey(start_state_seed)
            )
            states = set_tree(
                states,
                new_state,
                i + 1,
            )
            targets = set_tree(
                targets,
                new_target,
                i + 1,
            )
    return states, targets


def vmapping_search(
    puzzle: Puzzle,
    search_result_build: callable,
    star_fn: callable,
    vmap_size: int,
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
    vmapped_star = jax.jit(jax.vmap(star_fn, in_axes=(None, 0, 0)))
    if show_compile_time:
        print("initializing vmapped jit")
        start = time.time()
    vmapped_star(inital_search_result, empty_states, empty_states)
    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")
    return vmapped_star


def vmapping_get_state(search_result: SearchResult, idx: HashTableIdx_HeapValue):
    return jax.vmap(SearchResult.get_state, in_axes=(None, 0))(search_result, idx)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
