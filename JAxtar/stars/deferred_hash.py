"""Retain resolved hash slots between a deferred pop's lookup and commit."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from xtructure import HashIdx, HashTable, Xtructurable
from xtructure.core.dtype_facts import SIZE_DTYPE
from xtructure.hashtable.constants import SLOT_IDX_DTYPE
from xtructure.hashtable.hash_utils import get_new_idx_byterized_batched
from xtructure.hashtable.insert import _hashtable_parallel_insert_internal
from xtructure.hashtable.lookup import _hashtable_lookup_parallel_internal
from xtructure.hashtable.types import BucketIdx


class DeferredLookup(NamedTuple):
    """One immutable table's lookup results, permuted with the corresponding states."""

    initial_bucket: jax.Array
    step: jax.Array
    fingerprint: jax.Array
    resolved: BucketIdx
    found: jax.Array

    @classmethod
    def resolve(
        cls, table: HashTable, states: Xtructurable, active: jax.Array
    ) -> tuple[HashIdx, "DeferredLookup"]:
        probe = get_new_idx_byterized_batched(states, table._capacity, table.seed)
        start = BucketIdx(
            index=probe.index, slot_index=jnp.zeros_like(probe.index, dtype=SLOT_IDX_DTYPE)
        )
        resolved, found = _hashtable_lookup_parallel_internal(
            table, states, start, probe.step, probe.fingerprint, jnp.zeros_like(active), active
        )
        # Keep bucket and slot separate: after exhausting a probe budget, a
        # miss can return slot == bucket_size, which a flat index cannot recover.
        flat = resolved.index * SIZE_DTYPE(table.bucket_size) + resolved.slot_index
        return HashIdx(index=flat), cls(probe.index, probe.step, probe.fingerprint, resolved, found)

    def insert_unique(
        self, table: HashTable, states: Xtructurable, selected: jax.Array
    ) -> tuple[HashTable, jax.Array, HashIdx]:
        """Commit selected unique states to the same table they were resolved against."""
        idx = BucketIdx(
            index=jnp.where(selected, self.resolved.index, self.initial_bucket),
            slot_index=jnp.where(selected, self.resolved.slot_index, SLOT_IDX_DTYPE(0)),
        )
        insert_mask = selected & ~self.found
        table, placed = _hashtable_parallel_insert_internal(
            table, states, self.step, idx, insert_mask, self.fingerprint
        )
        bucket = jnp.where(insert_mask, placed.index, idx.index)
        slot = jnp.where(insert_mask, placed.slot_index, idx.slot_index)
        flat = bucket * SIZE_DTYPE(table.bucket_size) + slot
        # parallel_insert(assume_unique=True) maps every inactive lane to
        # the representative at lane zero, including when that lane is inactive.
        return table, insert_mask, HashIdx(index=jnp.where(selected, flat, flat[:1]))
