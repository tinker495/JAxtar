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
        - Per-chunk ``jax.lax.cond`` skip: an all-``False`` chunk yields all ``inf``
          without calling ``value_fn``.
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

    def _compute(_, inputs):
        states_slice, mask_slice = inputs

        def _calc(_):
            vals = value_fn(states_slice, mask_slice).astype(KEY_DTYPE)
            vals = jnp.maximum(0.0, vals)  # Ensure non-negative value
            return jnp.where(mask_slice, vals, jnp.inf)

        return None, jax.lax.cond(
            jnp.any(mask_slice),
            _calc,
            lambda _: jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE),
            None,
        )

    _, chunk_vals = jax.lax.scan(_compute, None, (chunk_states, chunk_mask))
    sorted_vals = chunk_vals.reshape((flat_size,))
    flat_vals = jnp.full((flat_size,), jnp.inf, dtype=KEY_DTYPE)
    flat_vals = flat_vals.at[sorted_idx].set(sorted_vals)
    return flat_vals
