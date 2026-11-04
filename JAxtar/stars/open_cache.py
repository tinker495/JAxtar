"""Exact, lossy cache of children a deferred search queued but has not expanded yet."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import BGPQ, HashTableProbe
from xtructure.core.dataclass import base_dataclass

from JAxtar.annotate import KEY_DTYPE

# A larger cache catches more duplicates, but random access over a big table cost Q* more
# than it saved (rubikscube-deepcubea Q*: 54 MB cache -1.6%, 109 MB +2.6%, 872 MB +3.1%).
# Slots round down to a power of two, so the cache stays within the budget.
OPEN_CACHE_BYTES = 64 * 2**20


@base_dataclass
class OpenCache:
    """Direct-mapped cache of the children a deferred search recently pushed into its BGPQ.

    The deferred hash table holds only expanded states, so a child still waiting in the
    queue is invisible to it and every new path to that child was queued again. A slot keeps
    one queued child's full uint32 encoding with its cost and queue key (both stored and
    compared in KEY_DTYPE); a newer child overwrites it. A hit is a child whose state is
    already queued with a cost and key no worse, so dropping it loses no better path.
    Comparing full encodings means a slot collision can only miss, never drop.

    A full BGPQ evicts entries to make room. Once an insert step could fill the queue the
    cache retires for good (`intact`); edges dropped before that point still rely on their
    twins, which a later eviction may remove just as it removes any queued entry.
    """

    encodings: jax.Array  # (slots, lanes) uint32 state encodings
    cost: jax.Array  # (slots,) cost each child was queued with; inf marks an empty slot
    key: jax.Array  # (slots,) queue key each child was queued with
    intact: jax.Array  # the queue has not been full enough to evict since the search began

    @staticmethod
    def build(statecls: Puzzle.State, max_queued: int) -> "OpenCache":
        """Largest power-of-two slot count within `OPEN_CACHE_BYTES` and `max_queued`.

        `max_queued` bounds how many children the search can ever queue (expansions times
        actions); `0` builds an empty cache for searches that never query it.
        """
        _, encodings = statecls.default((1,)).hash_with_uint32ed()
        lanes = encodings.shape[1]
        row_bytes = 4 * lanes + 2 * jnp.dtype(KEY_DTYPE).itemsize
        fit = min(max_queued, OPEN_CACHE_BYTES // row_bytes)
        slots = 1 << (fit.bit_length() - 1) if fit else 0
        return OpenCache(
            encodings=jnp.zeros((slots, lanes), dtype=jnp.uint32),
            cost=jnp.full((slots,), jnp.inf, dtype=KEY_DTYPE),
            key=jnp.full((slots,), jnp.inf, dtype=KEY_DTYPE),
            intact=jnp.array(True),
        )

    def retire_if_full(self, queue: BGPQ, incoming_entries: int) -> "OpenCache":
        """Retire the cache once inserting `incoming_entries` could make `queue` evict."""
        # BGPQ evicts a leaf when an insert finds heap_size >= branch_size - 1; one batch of
        # slack keeps the queue below that.
        return replace(
            self,
            intact=self.intact
            & (queue.size + incoming_entries <= queue.max_size - queue.batch_size),
        )

    def hits(
        self,
        probe: HashTableProbe,
        costs: jax.Array,
        keys: jax.Array | None,
        mask: jax.Array,
    ) -> jax.Array:
        """Rows of `mask` whose state is already queued with a cost and key no worse.

        Pass `keys=None` when the key is a function of state and cost (A*-d: g + h(state)),
        so a no-costlier twin is already no later in the queue.
        """
        slots = self._slots(probe)
        same_state = jnp.all(self.encodings[slots] == probe.uint32ed, axis=1)
        no_cheaper = self.cost[slots] <= costs.astype(KEY_DTYPE)
        if keys is not None:
            no_cheaper = no_cheaper & (self.key[slots] <= keys.astype(KEY_DTYPE))
        return mask & self.intact & same_state & no_cheaper

    def record(
        self, probe: HashTableProbe, costs: jax.Array, keys: jax.Array, queued: jax.Array
    ) -> "OpenCache":
        """Store the children queued this step over whatever their slots held before.

        Rows with an infinite key never enter the queue, so they are not recorded.
        """
        slots = self._slots(probe)
        queued = queued & jnp.isfinite(keys)
        # One writer per slot (the lowest row) keeps encoding, cost and key from one state.
        # Sorting the rows avoids a slot-sized owner array.
        sorted_slots, order = jax.lax.sort(
            (
                jnp.where(queued, slots, self.cost.shape[0]),
                jnp.arange(queued.shape[0], dtype=jnp.int32),
            ),
            num_keys=1,
            is_stable=True,
        )
        first = jnp.concatenate([jnp.ones((1,), jnp.bool_), sorted_slots[1:] != sorted_slots[:-1]])
        writer = jnp.zeros_like(queued).at[order].set(first & (sorted_slots < self.cost.shape[0]))
        return replace(
            self,
            encodings=xnp.update_on_condition(
                self.encodings, slots, writer, probe.uint32ed, unique_indices=True
            ),
            cost=xnp.update_on_condition(
                self.cost, slots, writer, costs.astype(KEY_DTYPE), unique_indices=True
            ),
            key=xnp.update_on_condition(
                self.key, slots, writer, keys.astype(KEY_DTYPE), unique_indices=True
            ),
        )

    def _slots(self, probe: HashTableProbe) -> jax.Array:
        if self.cost.shape[0] == 0:
            raise ValueError(
                "OpenCache has no slots; build the SearchResult with use_open_cache=True"
            )
        return probe.fingerprint % self.cost.shape[0]
