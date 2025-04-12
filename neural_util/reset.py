from typing import Any

import jax
from jax import numpy as jnp

PyTree = Any


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree = None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t),
        target,
        keys_tree,
    )


def build_soft_reset(params: PyTree, tau: float):
    def _is_batch_stat(path, value):
        # Check if 'batch_stats' is part of any dictionary key in the path
        is_batch_stat = any(
            isinstance(entry, jax.tree_util.DictKey) and "batch_stats" in entry.key
            for entry in path
        )
        return tau * (1 - is_batch_stat)  # 0 if batch_stat, tau if not batch_stat

    tau = jax.tree_util.tree_map_with_path(_is_batch_stat, params)

    @jax.jit
    def soft_reset(
        params: PyTree,
        key: jax.random.PRNGKey,
    ) -> PyTree:
        new_tensors = tree_random_normal_like(key, params)
        soft_reseted = jax.tree_map(
            lambda new, old, tau: tau * new + (1.0 - tau) * old, new_tensors, params, tau
        )
        # name dense is hardreset
        return soft_reseted

    return soft_reset
