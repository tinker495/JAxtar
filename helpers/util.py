import json
import time
from collections.abc import MutableMapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle
from pydantic import BaseModel

from JAxtar.search_base import Current, SearchResult


def convert_to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        # Recursively process the dict representation
        return convert_to_serializable_dict(obj.dict())
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable_dict(i) for i in obj]
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_hashable(val):
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return val


def display_value(val):
    # Convert tuples back to lists for display, and pretty-print JSON strings
    if isinstance(val, tuple):
        return str(list(val))
    try:
        loaded = json.loads(val)
        if isinstance(loaded, dict) or isinstance(loaded, list):
            return json.dumps(loaded, indent=2)
    except Exception:
        pass
    return str(val)


def flatten_array(array: chex.Array, dims: int) -> chex.Array:
    """
    Reshape the array to the given shape.
    """
    return jnp.reshape(array, (-1,) + array.shape[dims:])


def flatten_tree(tree: chex.Array, dims: int) -> chex.Array:
    """
    Reshape the index of the tree to the given shape.
    """
    return jax.tree_util.tree_map(lambda t: flatten_array(t, dims), tree)


def vmapping_init_target(puzzle: Puzzle, vmap_size: int, start_state_seeds: list[int]):
    start_state_seed = start_state_seeds[0]
    solve_configs, states = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), states[jnp.newaxis, ...]
    )
    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])),
        solve_configs[jnp.newaxis, ...],
    )

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_solve_config, new_state = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
            states = states.at[i + 1].set(new_state)
            solve_configs = solve_configs.at[i + 1].set(new_solve_config)
    return states, solve_configs


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
