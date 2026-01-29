import chex
import jax.numpy as jnp


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

    # Flatten masks
    flat2 = mask2.reshape(-1)
    # Ensure category 1 excludes category 2
    flat1 = jnp.logical_and(mask1.reshape(-1), jnp.logical_not(flat2))
    # Category 0 is whatever is left
    flat0 = jnp.logical_not(jnp.logical_or(flat2, flat1))

    # Calculate counts and offsets
    count2 = jnp.sum(flat2)
    count1 = jnp.sum(flat1)

    # Calculate position within each category (stable order)
    # cumsum gives 1-based index, so subtract 1 for 0-based
    pos2 = jnp.cumsum(flat2) - 1
    pos1 = jnp.cumsum(flat1) - 1
    pos0 = jnp.cumsum(flat0) - 1

    # Calculate destination indices
    # If in cat2: dest = pos2
    # If in cat1: dest = count2 + pos1
    # If in cat0: dest = count2 + count1 + pos0

    offset1 = count2
    offset0 = count2 + count1

    dest_idx = jnp.where(flat2, pos2, jnp.where(flat1, offset1 + pos1, offset0 + pos0))

    # Invert permutation: we want invperm such that arr[invperm] is sorted.
    # dest_idx tells us where each element goes: sorted_arr[dest_idx[i]] = arr[i]
    # So invperm[dest_idx[i]] = i

    n = flat2.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)

    # Use scatter to create invperm
    invperm = jnp.zeros(n, dtype=jnp.int32).at[dest_idx].set(indices)

    return invperm
