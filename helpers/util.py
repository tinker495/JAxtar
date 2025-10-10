import json
import time
from collections.abc import MutableMapping
from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
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
