from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


@jax.jit
def soft_update(target_params: PyTree, params: PyTree, tau: float) -> PyTree:
    return jax.tree.map(lambda t, n: t * tau + n * (1 - tau), target_params, params)


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree = None, treedef=None):
    if treedef is None:
        treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t),
        target,
        keys_tree,
    )


@jax.jit
def scaled_by_reset(
    tensors: PyTree,
    key: jax.random.PRNGKey,
    tau: float,
):
    new_tensors = tree_random_normal_like(key, tensors)
    soft_reseted = jax.tree_util.tree_map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, tensors
    )
    return soft_reseted
