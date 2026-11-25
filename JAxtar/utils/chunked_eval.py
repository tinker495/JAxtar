import jax
import jax.numpy as jnp
import xtructure.numpy as xnp

from JAxtar.annotate import KEY_DTYPE
from JAxtar.utils.array_ops import stable_partition_three


def chunked_masked_eval(value_fn, flat_states, flat_valid, action_size, batch_size):
    """Evaluate ``value_fn`` only on the masked states, chunk by chunk.

    Partition the row indices, then gather one chunk of states per iteration.
    Write each chunk's scores directly to their original positions.

    Args:
        value_fn: ``value_fn(states_slice, mask_slice) -> (batch_size,)``. Any
            per-state parameters must be closed over by the caller.
        flat_states: Flattened states of leading size ``action_size * batch_size``.
        flat_valid: Boolean mask over ``flat_states`` selecting states to evaluate.
        action_size: Number of chunks (rows) to scan over.
        batch_size: Number of states per chunk.

    Returns:
        ``flat_vals`` of shape ``(action_size * batch_size,)`` where
        ``flat_vals[i] == max(0.0, value_fn(states[i]))`` when ``flat_valid[i]``
        and ``jnp.inf`` otherwise.

    Invariants:
        - Chunk skip: valid rows are partitioned to the front, so chunk ``j``
          holds work iff ``j < ceil(n_valid / batch_size)``; a bounded
          ``jax.lax.while_loop`` evaluates exactly those chunks and all-masked
          chunks stay ``inf`` without calling ``value_fn``. (A per-chunk
          ``lax.cond`` inside a scan pays one host predicate sync for every
          chunk; the while pays one per *executed* chunk plus one.)
        - ``max(0.0, v)`` non-negativity clamp on every computed value.
        - Scatter uses the ``stable_partition_three`` inverse permutation (a full
          permutation of all indices), so results land at their original indices.
    """
    flat_size = action_size * batch_size
    sorted_idx = stable_partition_three(flat_valid, jnp.zeros_like(flat_valid, dtype=jnp.bool_))
    chunk_indices = jnp.reshape(sorted_idx, (action_size, batch_size))
    n_valid = jnp.sum(flat_valid, dtype=jnp.int32)
    n_chunks = (n_valid + batch_size - 1) // batch_size

    def _cond(carry):
        j, _ = carry
        return j < n_chunks

    def _body(carry):
        j, vals = carry
        start = j * batch_size
        indices = chunk_indices[j]
        states_j = xnp.take(flat_states, indices, axis=0)
        mask_j = start + jnp.arange(batch_size, dtype=jnp.int32) < n_valid
        v = value_fn(states_j, mask_j).astype(KEY_DTYPE)
        v = jnp.maximum(0.0, v)  # Ensure non-negative value
        v = jnp.where(mask_j, v, jnp.inf)
        return j + 1, vals.at[indices].set(v, unique_indices=True)

    init_vals = jnp.full((flat_size,), jnp.inf, dtype=KEY_DTYPE)
    _, flat_vals = jax.lax.while_loop(_cond, _body, (jnp.array(0, dtype=jnp.int32), init_vals))
    return flat_vals
