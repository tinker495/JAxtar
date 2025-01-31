import chex
import jax
import jax.numpy as jnp


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
