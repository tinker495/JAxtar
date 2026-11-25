import chex
import jax
import jax.numpy as jnp


def batched_state_equal(lhs: object, rhs: object) -> chex.Array:
    """Per-row equality of two equally-shaped batched states -> ``(batch,)`` bool.

    Compares the raw leaves instead of ``lhs == rhs``: ``Xtructurable.__eq__`` reduces
    the whole comparison to a single scalar, so the previous implementation returned a
    0-d array. Every caller then broadcast that scalar over its mask, making the
    non-backtracking filters in both ``id_stars`` and ``beamsearch`` all-or-nothing --
    in practice always "nothing", so they blocked no node while still paying for the
    ancestor trail/trace bookkeeping.
    """
    lhs_leaves = jax.tree_util.tree_leaves(lhs)
    rhs_leaves = jax.tree_util.tree_leaves(rhs)
    if not lhs_leaves:
        raise ValueError("State comparison received an empty tree")
    if len(lhs_leaves) != len(rhs_leaves):
        raise ValueError("State comparison received mismatched trees")

    result = None
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        equal = lhs_leaf == rhs_leaf
        if equal.ndim > 1:
            # Leading axis is the batch; every other axis belongs to the state itself.
            equal = jnp.all(equal, axis=tuple(range(1, equal.ndim)))
        result = equal if result is None else jnp.logical_and(result, equal)
    assert result is not None
    return result


def stable_partition_three(mask2: chex.Array, mask1: chex.Array) -> chex.Array:
    """
    Compute a stable 3-way partition inverse permutation for flattened arrays.

    - Category 2 (mask2): first block
    - Category 1 (mask1 & ~mask2): second block
    - Category 0 (else): last block

    Returns indices suitable for gathering flattened arrays to achieve the
    [2..., 1..., 0...] ordering while preserving relative order within each class.
    Implementation uses prefix-sum (cumsum) + scatter for O(N) complexity,
    replacing the previous O(N log N) sort-based approach.
    """

    flat2 = mask2.reshape(-1)
    flat1 = jnp.logical_and(mask1.reshape(-1), jnp.logical_not(flat2))
    indices = jnp.arange(flat2.size, dtype=jnp.int32)
    if flat2.size == 0:
        return indices

    prefix2 = jnp.cumsum(flat2)
    prefix1 = jnp.cumsum(flat1)
    count2 = prefix2[-1]
    count1 = prefix1[-1]

    # At a category-0 element both prefixes count only preceding elements.
    pos0 = indices - prefix2 - prefix1
    dest_idx = jnp.where(
        flat2, prefix2 - 1, jnp.where(flat1, count2 + prefix1 - 1, count2 + count1 + pos0)
    )
    return jnp.zeros_like(indices).at[dest_idx].set(indices)
