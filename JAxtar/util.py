import chex
import jax.numpy as jnp
from jax import lax


def stable_partition_three(mask2: chex.Array, mask1: chex.Array) -> chex.Array:
    """
    Compute a stable 3-way partition inverse permutation for flattened arrays.

    - Category 2 (mask2): first block
    - Category 1 (mask1 & ~mask2): second block
    - Category 0 (else): last block

    Returns indices suitable for gathering flattened arrays to achieve the
    [2..., 1..., 0...] ordering while preserving relative order within each class.
    """

    # Flatten masks
    flat2 = mask2.reshape(-1)
    # Ensure category 1 excludes category 2
    flat1 = jnp.logical_and(mask1.reshape(-1), jnp.logical_not(flat2))

    # Compute category id per element: 2, 1, or 0
    cat = jnp.where(flat2, 2, jnp.where(flat1, 1, 0)).astype(jnp.int32)

    n = cat.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)

    # Stable sort by key = -cat so that 2-block comes first, then 1, then 0.
    # The stable flag preserves original order within equal keys (intra-class stability).
    _, invperm = lax.sort_key_val(-cat, indices, dimension=0, is_stable=True)

    # Return gather indices: arr[invperm] yields [2..., 1..., 0...] with stable intra-class order
    return invperm
