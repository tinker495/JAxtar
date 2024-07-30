import chex
import jax.numpy as jnp
import jax
from functools import partial
from abc import ABC, abstractmethod
from collections import namedtuple

def heapcalue_dataclass(cls):
    """
    This function is a decorator that creates a dataclass for HeapValue.
    """
    cls = chex.dataclass(cls)

    shape_tuple = namedtuple('shape', cls.__annotations__.keys())
    def get_shape(self) -> shape_tuple:
        return shape_tuple(*[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()])
    setattr(cls, 'shape', property(get_shape))

    type_tuple = namedtuple('dtype', cls.__annotations__.keys())
    def get_type(self) -> type_tuple:
        return type_tuple(*[jnp.dtype(getattr(self, field_name).dtype) for field_name in cls.__annotations__.keys()])
    setattr(cls, 'dtype', property(get_type))

    def getitem(self, index):
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__getitem__'):
                try:
                    new_values[field_name] = field_value[index]
                except IndexError:
                    new_values[field_name] = field_value
            else:
                new_values[field_name] = field_value
        return cls(**new_values)
    setattr(cls, '__getitem__', getitem)

    def len(self):
        return self.shape[0][0]
    setattr(cls, '__len__', len)

    return cls

@heapcalue_dataclass
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

@heapcalue_dataclass
class HashTableHeapValue(HeapValue):
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: jnp.uint32 / hashtable index
    2. table_index: jnp.uint8 / cuckoo table index
    """
    index: chex.Array
    table_index: chex.Array

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
        branch_size = jnp.where(total_size % group_size == 0, total_size // group_size, total_size // group_size + 1)
        max_size = branch_size * group_size
        size = jnp.uint32(0)
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
    def merge_sort_split(ak: chex.Array, av: HeapValue, bk: chex.Array, bv: HeapValue) -> tuple[chex.Array, HeapValue, chex.Array, HeapValue]:
        """
        Merge two sorted key tensors ak and bk as well as corresponding
        value tensors av and bv into a single sorted tensor.

        Args:
            ak: chex.Array - sorted key tensor
            av: HeapValue - sorted value tensor
            bk: chex.Array - sorted key tensor
            bv: HeapValue - sorted value tensor

        Returns:
            key1: chex.Array - merged and sorted
            val1: HeapValue - merged and sorted
            key2: chex.Array - merged and sorted
            val2: HeapValue - merged and sorted
        """
        n = ak.shape[-1] # size of group
        key = jnp.concatenate([ak, bk])
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)
        idx = jnp.argsort(key)
        key = key[idx]
        val = jax.tree_util.tree_map(lambda x: x[idx], val)
        key1 = key[:n] # smaller keys
        key2 = key[n:] # larger keys
        val1 = val[:n]
        val2 = val[n:]
        return key1, val1, key2, val2
    
    @staticmethod
    def merge_buffer(blockk: chex.Array, blockv: HeapValue, bufferk: chex.Array, bufferv: HeapValue):
        """
        Merge buffer into the key and value.

        inf key values are not active key, so it is not filled.
        if buffer overflow, heapify filled block is returned.
        if buffer not overflow, heapify filled buffer is not returned.
        only buffer is filled with active keys.

        Args:
            blockk: chex.Array
            blockv: HeapValue
            bufferk: chex.Array
            bufferv: HeapValue

        Returns:
            blockk: chex.Array - heapyfing block keys
            blockv: HeapValue - heapyfing block values
            bufferk: chex.Array - buffer keys
            bufferv: HeapValue - buffer values
            buffer_overflow: bool # if buffer overflow, return True
        """
        n = blockk.shape[0]
        key = jnp.concatenate([blockk, bufferk])
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), blockv, bufferv)
        idx = jnp.argsort(key)
        key = key[idx]
        val = jax.tree_util.tree_map(lambda x: x[idx], val)
        merged_n = key.shape[0]
        filled = jnp.isfinite(key) # inf values are not filled
        n_filled = jnp.sum(filled)
        buffer_overflow = n_filled >= n # buffer overflow
        block_idx, buffer_idx = jax.lax.cond(buffer_overflow,
                                        lambda _: (jnp.arange(0,n), jnp.arange(n, merged_n)), # if buffer overflow, block is filled with smaller keys
                                        lambda _: (jnp.arange(n - 1, merged_n), jnp.arange(0, n - 1)), # if buffer not overflow, buffer is filled with smaller keys 
                                        None)
        blockk = key[block_idx]
        blockv = val[block_idx]
        bufferk = key[buffer_idx]
        bufferv = val[buffer_idx]
        return blockk, blockv, bufferk, bufferv, buffer_overflow

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
    def _next(current, target):
        """
        Get the next index.
        travel to target as heapify.
        """
        return target.astype(jnp.uint32) >> (jax.lax.clz(current) - jax.lax.clz(target) - 1)

    @staticmethod
    def _insert_heapify(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue):
        """
        Insert a key-value pair into the heap.
        """
        size = heap.size // heap.group_size
        def _cond(var):
            _, _, _, _, n = var
            return n < size

        def insert_heapify(var):
            key_store, val_store, keys, values, n = var
            head, hvalues, keys, values = BGPQ.merge_sort_split(
                key_store[n], val_store[n], keys, values
            )
            key_store = key_store.at[n].set(head)
            val_store = jax.tree_util.tree_map(lambda x, y: x.at[n].set(y), val_store, hvalues)
            return  key_store, val_store, keys, values, BGPQ._next(n, size)

        key_store, val_store, keys, values, _ = jax.lax.while_loop(_cond, insert_heapify, (heap.key_store, heap.val_store, block_key, block_val, BGPQ._next(0, size)))
        key_store = key_store.at[size].set(keys)
        val_store = jax.tree_util.tree_map(lambda x, y: x.at[size].set(y), val_store, values)
        heap.key_store = key_store
        heap.val_store = val_store
        return heap

    @staticmethod
    def insert(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue):
        """
        Insert a key-value pair into the heap.
        """
        added_size = jnp.sum(jnp.isfinite(block_key))
        root_key = heap.key_store[0]
        root_val = heap.val_store[0]
        root_key, root_val, block_key, block_val = BGPQ.merge_sort_split(root_key, root_val, block_key, block_val)
        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = jax.tree_util.tree_map(lambda x, r: x.at[0].set(r), heap.val_store, root_val)
        
        block_key, block_val, heap.key_buffer, heap.val_buffer, buffer_overflow = BGPQ.merge_buffer(block_key, block_val, heap.key_buffer, heap.val_buffer)

        heap = jax.lax.cond(buffer_overflow,
                            lambda heap, block_key, block_val: BGPQ._insert_heapify(heap, block_key, block_val),
                            lambda heap, block_key, block_val: heap,
                            heap, block_key, block_val)
        heap.size = heap.size + added_size
        return heap
    
    @staticmethod
    def delete_mins(heap: "BGPQ"):
        """
        Delete the minimum key-value pair from the heap.
        In this function we did not clear the values, we just set the key to inf.
        """
        min_keys = heap.key_store[0]
        min_values = heap.val_store[0]
        size = jnp.uint32(heap.size // heap.group_size)

        # todo: optimize this function

        def make_empty(key_store, val_store, key_buffer, val_buffer):
            root_key, root_val, key_buffer, val_buffer = BGPQ.merge_sort_split(jnp.full_like(key_store[0], jnp.inf), val_store[0], key_buffer, val_buffer)
            val_store = jax.tree_util.tree_map(lambda x, y: x.at[0].set(y), val_store, root_val)
            return key_store.at[0].set(root_key), val_store, key_buffer, val_buffer # empty the root
        
        def delete_heapify(key_store, val_store, key_buffer, val_buffer):
            last = size - 1
            key_store = key_store.at[0].set(key_store[last]).at[last].set(jnp.inf)
            root_key, root_val, key_buffer, val_buffer = BGPQ.merge_sort_split(key_store[0], val_store[0], key_buffer, val_buffer)
            key_store = key_store.at[0].set(root_key)
            root_val = jax.tree_util.tree_map(lambda x, y: x.at[0].set(y), val_store, root_val)
            def _f(_, var):
                key_store, val_store, n = var
                c = jnp.stack(((n + 1) * 2 - 1, (n + 1) * 2))
                c_l,c_r = key_store[c[0]], key_store[c[1]]
                c_lv, c_rv = val_store[c[0]], val_store[c[1]]
                ins = jnp.where(c_l[-1] < c_r[-1], 0, 1)
                s, l = c[ins], c[1 - ins]
                small, smallv, k3, v3 = BGPQ.merge_sort_split(c_l, c_lv, c_r, c_rv)
                k1, v1, k2, v2 = BGPQ.merge_sort_split(key_store[n], val_store[n], small, smallv)
                key_store = key_store.at[l].set(k3).at[n].set(k1).at[s].set(k2)
                val_store = jax.tree_util.tree_map(lambda x, v1, v2, v3: x.at[l].set(v3).at[n].set(v1).at[s].set(v2), val_store, v1, v2, v3)
                return key_store, val_store, s
            key_store, val_store, _ = jax.lax.fori_loop(jnp.uint32(0), size, _f, (key_store, val_store, 0))
            return key_store, val_store, key_buffer, val_buffer
        
        key_store, val_store, key_buffer, val_buffer = jax.lax.cond(size > 1,
                                            delete_heapify,
                                            make_empty,
                                            heap.key_store, heap.val_store, heap.key_buffer, heap.val_buffer)
        heap.key_store = key_store
        heap.val_store = val_store
        heap.key_buffer = key_buffer
        heap.val_buffer = val_buffer
        heap.size = heap.size - jnp.sum(jnp.isfinite(min_keys))
        return heap, min_keys, min_values
