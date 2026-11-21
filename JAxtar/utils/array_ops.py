import chex
import jax
import jax.numpy as jnp


def batched_state_equal(lhs: object, rhs: object) -> chex.Array:
    """Per-row equality of two equally-shaped batched states -> ``(batch,)`` bool.

    Compares raw pytree leaves instead of ``lhs == rhs`` because ``Xtructurable.__eq__``
    reduces across the batch to a single scalar, which would collapse per-row
    non-backtracking masks.
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
    [2..., 1..., 0...] ordering while preserving relative order within each class,
    using prefix-sum (cumsum) + scatter for O(N) complexity.
    """

    flat2 = mask2.reshape(-1)
    flat1 = jnp.logical_and(mask1.reshape(-1), jnp.logical_not(flat2))
    flat0 = jnp.logical_not(jnp.logical_or(flat2, flat1))

    count2 = jnp.sum(flat2)
    count1 = jnp.sum(flat1)

    pos2 = jnp.cumsum(flat2) - 1
    pos1 = jnp.cumsum(flat1) - 1
    pos0 = jnp.cumsum(flat0) - 1

    offset1 = count2
    offset0 = count2 + count1

    dest_idx = jnp.where(flat2, pos2, jnp.where(flat1, offset1 + pos1, offset0 + pos0))

    # Invert permutation: we want invperm such that arr[invperm] is sorted.
    # dest_idx tells us where each element goes: sorted_arr[dest_idx[i]] = arr[i]
    # So invperm[dest_idx[i]] = i

    n = flat2.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)

    invperm = jnp.zeros(n, dtype=jnp.int32).at[dest_idx].set(indices)

    return invperm
