import chex
import jax
import jax.numpy as jnp


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


def unflatten_array(array: chex.Array, shape: tuple) -> chex.Array:
    """
    Unflatten the array to the given shape.
    """
    return jnp.reshape(array, shape + array.shape[1:])


def unflatten_tree(tree: chex.Array, shape: tuple) -> chex.Array:
    """
    Unflatten the tree to the given shape.
    """
    return jax.tree_util.tree_map(lambda t: unflatten_array(t, shape), tree)
