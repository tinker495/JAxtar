import chex
import jax.numpy as jnp
import jax
from functools import partial
from abc import ABC, abstractmethod

@chex.dataclass
class HeapValue(ABC):
    """
    This class is a dataclass that represents a heap value.
    value could be a uint32 value, but it could be more complex.
    so, we use a dataclass to represent the value.
    """
    pass
    
    @staticmethod
    @abstractmethod
    def default(_ = None) -> "HeapValue":
        """
        This function is a default dataclass for HeapValue.
        we can use this function with vmapped functions.
        vmap(HeapValue.default)(jnp.arange(1000)) -> HeapValue[1000, ...]
        """
        pass

@chex.dataclass
class HashTableHeapValue:
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: jnp.uint32 / hashtable index
    2. table_index: jnp.uint8 / cuckoo table index
    """
    index: jnp.uint32
    table_index: jnp.uint8

    @staticmethod
    def default(_ = None) -> "HashTableHeapValue":
        return HashTableHeapValue(index=jnp.zeros(1, dtype=jnp.uint32), table_index=jnp.zeros(1, dtype=jnp.uint8))

@chex.dataclass
class BGPQ: # Batched GPU Priority Queue
    """
    This class is a batched GPU priority queue.
    It is a dataclass with the following fields:
    1. max_size: int
    2. size: int
    3. group_size: int
    4. key_store: chex.Array
    5. val_store: chex.Array
    6. key_buffer: chex.Array
    7. val_buffer: chex.Array

    This class works with the following properties:
    inf value is used to pad the key_store and key_buffer.
    """
    max_size: int # maximum size of the heap
    size: int # current size of the heap
    group_size: int # size of the group
    key_store: chex.Array # shape = (total_size, group_size) batched binary tree of keys
    val_store: HeapValue # shape = (total_size, group_size, ...) batched binary tree of values
    key_buffer: chex.Array # shape = (group_size - 1,) key buffer for unbatched(not enough to fill a batch) keys
    val_buffer: HeapValue # shape = (group_size - 1, ...) value buffer for unbatched(not enough to fill a batch) values

    @staticmethod
    def make_heap(total_size, group_size, value_class=HeapValue):
        """
        Create a heap over vectors of `group_size` that
        can store up to `total_size` elements.
        value_type is the type of the values stored in the heap.
        In this repository, we only use uint32 for hash indexes values.
        """
        branch_size = total_size // group_size + 1
        max_size = branch_size * group_size
        size = jnp.zeros(1, dtype=jnp.uint32)
        key_store = jnp.full((branch_size, group_size), jnp.inf, dtype=jnp.float32) # [branch_size, group_size]
        val_store = jax.vmap(lambda _: jax.vmap(value_class.default)(jnp.arange(group_size)))(jnp.arange(branch_size)) # [branch_size, group_size, ...]
        key_buffer = jnp.full((group_size - 1,), jnp.inf, dtype=jnp.float32) # [group_size - 1]
        val_buffer = jax.vmap(value_class.default)(jnp.arange(group_size - 1)) # [group_size - 1, ...]
        return BGPQ(max_size=max_size,
                    size=size,
                    group_size=group_size,
                    key_store=key_store,
                    val_store=val_store,
                    key_buffer=key_buffer,
                    val_buffer=val_buffer)

    @staticmethod
    def merge_sort_split(ak: chex.Array, av: HeapValue, bk: chex.Array, bv: HeapValue):
        """
        Merge two sorted key tensors ak and bk as well as corresponding
        value tensors av and bv into a single sorted tensor.
        """
        n = ak.shape[-1] # size of group
        key = jnp.concatenate([ak, bk])
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]))(av, bv)
        idx = jnp.argsort(key)
        key = key[idx]
        val = jax.tree_util.tree_map(lambda x: x[idx], val)
        key1 = key[:n] # smaller keys
        key2 = key[n:] # larger keys
        val1, val2 = jax.tree_util.tree_map(lambda x: (x[:n], x[n:]), val) # smaller and larger key values
        return (key1, val1), (key2, val2)
    
    @staticmethod
    def make_batched(key: chex.Array, val: HeapValue, group_size: int):
        """
        Make a batched version of the key-value pair.
        """
        n = key.shape[0]
        m = n // group_size + 1
        key = jnp.concatenate([key, jnp.full((m * group_size - n,), jnp.inf, dtype=jnp.float32)])
        val = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), val, jax.vmap(val.default)(jnp.arange(m * group_size - n)))
        key = key[:m * group_size].reshape((m, group_size))
        val = jax.tree_util.tree_map(lambda x: x[:m * group_size].reshape((m, group_size) + x.shape[1:]), val)
        return key, val

    @staticmethod
    def insert(heap: "BGPQ", batched_key: chex.Array, batched_val: chex.Array):
        """
        Insert a key-value pair into the heap.
        """
        # first, 
        root_key, batched_key, root_val, batched_val = BGPQ.merge_sort_split(heap.key_store[0], batched_key, heap.val_store[0], batched_val)
        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = heap.val_store.at[0].set(root_val)
        
        batched_key, heap.key_buffer, batched_val, heap.val_buffer = BGPQ.merge_sort_split(batched_key, heap.key_buffer, batched_val, heap.val_buffer)