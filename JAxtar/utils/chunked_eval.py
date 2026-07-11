import jax
import jax.numpy as jnp
import xtructure.numpy as xnp

from JAxtar.annotate import KEY_DTYPE
from JAxtar.utils.array_ops import stable_partition_three


def chunked_masked_eval(value_fn, flat_states, flat_valid, action_size, batch_size):
    """Evaluate ``value_fn`` only on the masked states, chunk by chunk.

    The valid states are partitioned to the front, reshaped into
    ``(action_size, batch_size)`` chunks, evaluated with a ``jax.lax.scan`` so
    only one batch is resident at a time, and scattered back to their original
    positions. This keeps the peak memory footprint at a single chunk while
    still producing a fully populated ``(action_size * batch_size,)`` result.

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
    sorted_states = xnp.take(flat_states, sorted_idx, axis=0)
    sorted_mask = flat_valid[sorted_idx]

    chunk_states = xnp.reshape(sorted_states, (action_size, batch_size))
    chunk_mask = sorted_mask.reshape((action_size, batch_size))

    n_valid = jnp.sum(sorted_mask, dtype=jnp.int32)
    n_chunks = (n_valid + batch_size - 1) // batch_size

    def _cond(carry):
        j, _ = carry
        return j < n_chunks

    def _body(carry):
        j, vals = carry
        states_j = xnp.take(chunk_states, j, axis=0)
        mask_j = chunk_mask[j]
        v = value_fn(states_j, mask_j).astype(KEY_DTYPE)
        v = jnp.maximum(0.0, v)  # Ensure non-negative value
        v = jnp.where(mask_j, v, jnp.inf)
        return j + 1, vals.at[j].set(v)

    init_vals = jnp.full((action_size, batch_size), jnp.inf, dtype=KEY_DTYPE)
    _, chunk_vals = jax.lax.while_loop(_cond, _body, (jnp.array(0, dtype=jnp.int32), init_vals))
    sorted_vals = chunk_vals.reshape((flat_size,))
    flat_vals = jnp.full((flat_size,), jnp.inf, dtype=KEY_DTYPE)
    flat_vals = flat_vals.at[sorted_idx].set(sorted_vals)
    return flat_vals
