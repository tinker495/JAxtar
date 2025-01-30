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


def set_array(array: chex.Array, insert_value: chex.Array, *indexs) -> chex.Array:
    """
    Set the value of the array at the given indexs to the insert_value.
    """
    return array.at[indexs].set(insert_value)


def set_tree(tree: chex.Array, insert_value: chex.Array, *indexs) -> chex.Array:
    """
    Set the value of the tree at the given indexs to the insert_value.
    """
    return jax.tree_util.tree_map(lambda t, i: set_array(t, i, *indexs), tree, insert_value)


def set_array_as_condition(
    array: chex.Array, condition: chex.Array, insert_value: chex.Array, *indexs
) -> chex.Array:
    """
    Set the value of the array at the given indexs to the insert_value if the condition is true,
    otherwise keep the original value.
    """
    return array.at[indexs].set(
        jnp.where(
            jnp.reshape(condition, (-1,) + (insert_value.ndim - 1) * (1,)),
            insert_value,
            array[indexs],
        )
    )


def set_tree_as_condition(
    tree: chex.Array, condition: chex.Array, insert_tree: chex.Array, *indexs
) -> chex.Array:
    """
    Set the value of the tree at the given indexs to the insert_tree if the condition is true,
    otherwise keep the original value.
    """
    return jax.tree_util.tree_map(
        lambda t, i: set_array_as_condition(t, condition, i, *indexs), tree, insert_tree
    )
