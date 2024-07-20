import chex
import jax.numpy as jnp
import jax
from functools import partial

@chex.dataclass
class BGPQ: # Batched GPU Priority Queue
    max_size: int # maximum size of the heap
    size: int # current size of the heap
    group_size: int # size of the group
    key_store: chex.Array # shape = (total_size, group_size) batched binary tree of keys
    val_store: chex.Array # shape = (total_size, group_size) batched binary tree of values
    key_buffer: chex.Array # shape = (group_size - 1,) key buffer for unbatched(not enough to fill a batch) keys
    val_buffer: chex.Array # shape = (group_size - 1,) value buffer for unbatched(not enough to fill a batch) values

    @staticmethod
    def make_heap(total_size, group_size, value_type=jnp.uint32):
        """
        Create a heap over vectors of `group_size` that
        can store up to `total_size` elements.
        value_type is the type of the values stored in the heap.
        In this repository, we only use uint32 for hash indexes values.
        """
        branch_size = total_size // group_size + 1
        max_size = branch_size * group_size
        size = jnp.zeros(1, dtype=jnp.uint32)
        key_store = jnp.full((branch_size, group_size), 1.e5, dtype=jnp.float32)
        val_store = jnp.zeros((branch_size, group_size), dtype=value_type)
        key_buffer = jnp.full((group_size - 1,), 1.e5, dtype=jnp.float32) 
        val_buffer = jnp.zeros((group_size - 1,), dtype=value_type)
        return BGPQ(max_size=max_size,
                    size=size,
                    group_size=group_size,
                    key_store=key_store,
                    val_store=val_store,
                    key_buffer=key_buffer,
                    val_buffer=val_buffer)

    @staticmethod
    def merge(ak, bk, av, bv):
        """
        Merge two sorted key tensors ak and bk as well as corresponding
        value tensors av and bv into a single sorted tensor.
        """
        n = ak.shape[-1] # size of group
        key = jnp.concatenate([ak, bk])
        val = jnp.concatenate([av, bv])
        idx = jnp.argsort(key)
        key = key[idx]
        val = val[idx]
        return (key[:n], key[n:], val[:n], val[n:])
    
    @staticmethod
    def make_batched(key, val, group_size):
        """
        Make a batched version of the key-value pair.
        """
        n = key.shape[-1]
        m = n // group_size
        key = jnp.concatenate([key, jnp.full((m * group_size - n,), 1.e5, dtype=jnp.float32)])
        val = jnp.concatenate([val, jnp.zeros((m * group_size - n,), dtype=jnp.uint32)])
        key = key[:m * group_size].reshape((m, group_size))
        val = val[:m * group_size].reshape((m, group_size))
        return key, val

    @staticmethod
    def insert_batched():
        """
        Insert a batched key-value pair into the heap.
        """
        pass

    @staticmethod
    def insert(heap: "BGPQ", insert_key, insert_val):
        """
        Insert a key-value pair into the heap.
        """
        batched_insert_key, batched_insert_val = BGPQ.make_batched(insert_key, insert_val, heap.group_size)