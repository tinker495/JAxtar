from collections import namedtuple

import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import (
    HASH_POINT_DTYPE,
    HASH_TABLE_IDX_DTYPE,
    KEY_DTYPE,
    SIZE_DTYPE,
)

SORT_STABLE = True


def bgpq_value_dataclass(cls):
    """
    This function is a decorator that creates a dataclass for HeapValue.
    """
    cls = chex.dataclass(cls)

    shape_tuple = namedtuple("shape", cls.__annotations__.keys())

    def get_shape(self) -> shape_tuple:
        return shape_tuple(
            *[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        return type_tuple(
            *[
                jnp.dtype(getattr(self, field_name).dtype)
                for field_name in cls.__annotations__.keys()
            ]
        )

    setattr(cls, "dtype", property(get_type))

    def getitem(self, index):
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__getitem__"):
                new_values[field_name] = field_value[index]
            else:
                new_values[field_name] = field_value
        return cls(**new_values)

    setattr(cls, "__getitem__", getitem)

    def len(self):
        return self.shape[0][0]

    setattr(cls, "__len__", len)

    # this class must have a default method for creating a default instance.
    assert hasattr(cls, "default"), "HeapValue class must have a default method."

    return cls


@bgpq_value_dataclass
class HeapValue:  # dummy heap value for type hinting
    """
    This class is a dataclass that represents a heap value.
    value could be a uint32 value, but it could be more complex.
    so, we use a dataclass to represent the value.
    """

    pass

    @staticmethod
    def default(_=None) -> "HeapValue":
        pass


@bgpq_value_dataclass
class HashTableIdx_HeapValue:
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: hashtable index
    2. table_index: cuckoo table index
    """

    index: chex.Array
    table_index: chex.Array

    @staticmethod
    def default(_=None) -> "HashTableIdx_HeapValue":
        return HashTableIdx_HeapValue(
            index=jnp.full((), jnp.inf, dtype=HASH_POINT_DTYPE),
            table_index=jnp.full((), jnp.inf, dtype=HASH_TABLE_IDX_DTYPE),
        )


@chex.dataclass
class BGPQ:  # Batched GPU Priority Queue
    """
    This class is a batched GPU priority queue.
    It is a dataclass with the following fields:
    1. max_size: int
    2. size: int
    3. batch_size: int
    4. key_store: chex.Array
    5. val_store: chex.Array
    6. key_buffer: chex.Array
    7. val_buffer: chex.Array

    This class works with the following properties:
    inf value is used to pad the key_store and key_buffer.
    """

    max_size: int  # maximum size of the heap
    size: int  # current size of the heap
    branch_size: int  # size of the branch
    batch_size: int  # size of the group
    key_store: chex.Array  # shape = (total_size, batch_size) batched binary tree of keys
    val_store: HeapValue  # shape = (total_size, batch_size, ...) batched binary tree of values
    key_buffer: chex.Array  # shape = (batch_size - 1,) key buffer for unbatched(not enough to fill a batch) keys
    val_buffer: HeapValue  # shape = (batch_size - 1, ...) value buffer for unbatched(not enough to fill a batch) values

    @staticmethod
    def build(total_size, batch_size, value_class=HeapValue):
        """
        Create a heap over vectors of `batch_size` that
        can store up to `total_size` elements.
        value_type is the type of the values stored in the heap.
        In this repository, we only use uint32 for hash indexes values.
        """
        total_size = total_size
        branch_size = jnp.where(
            total_size % batch_size == 0, total_size // batch_size, total_size // batch_size + 1
        ).astype(SIZE_DTYPE)
        max_size = branch_size * batch_size
        size = SIZE_DTYPE(0)
        key_store = jnp.full(
            (branch_size, batch_size), jnp.inf, dtype=KEY_DTYPE
        )  # [branch_size, batch_size]
        val_store = jax.vmap(lambda _: jax.vmap(value_class.default)(jnp.arange(batch_size)))(
            jnp.arange(branch_size)
        )  # [branch_size, batch_size, ...]
        key_buffer = jnp.full((batch_size - 1,), jnp.inf, dtype=KEY_DTYPE)  # [batch_size - 1]
        val_buffer = jax.vmap(value_class.default)(
            jnp.arange(batch_size - 1)
        )  # [batch_size - 1, ...]
        return BGPQ(
            max_size=max_size,
            size=size,
            branch_size=branch_size,
            batch_size=batch_size,
            key_store=key_store,
            val_store=val_store,
            key_buffer=key_buffer,
            val_buffer=val_buffer,
        )

    @staticmethod
    @jax.jit
    def merge_sort_split(
        ak: chex.Array, av: HeapValue, bk: chex.Array, bv: HeapValue
    ) -> tuple[chex.Array, HeapValue, chex.Array, HeapValue]:
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
        n = ak.shape[-1]  # size of group
        key = jnp.concatenate([ak, bk])
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)
        idx = jnp.argsort(key, stable=SORT_STABLE)

        # Sort both key and value arrays using the same index
        sorted_key = key[idx]
        sorted_val = jax.tree_util.tree_map(lambda x: x[idx], val)
        return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]

    @staticmethod
    @jax.jit
    def merge_buffer(
        blockk: chex.Array, blockv: HeapValue, bufferk: chex.Array, bufferv: HeapValue
    ):
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
        idx = jnp.argsort(key, stable=SORT_STABLE)
        key = key[idx]
        val = jax.tree_util.tree_map(lambda x: x[idx], val)
        filled = jnp.isfinite(key)  # inf values are not filled
        n_filled = jnp.sum(filled)
        buffer_overflow = n_filled >= n  # buffer overflow

        def overflowed(key, val):
            return key[:n], val[:n], key[n:], val[n:]

        def not_overflowed(key, val):
            return key[n - 1 :], val[n - 1 :], key[: n - 1], val[: n - 1]

        blockk, blockv, bufferk, bufferv = jax.lax.cond(
            buffer_overflow,
            overflowed,
            not_overflowed,
            key,
            val,
        )
        return blockk, blockv, bufferk, bufferv, buffer_overflow

    @staticmethod
    def make_batched(key: chex.Array, val: HeapValue, batch_size: int):
        """
        Make a batched version of the key-value pair.
        """
        n = key.shape[0]
        m = n // batch_size + 1
        key = jnp.concatenate([key, jnp.full((m * batch_size - n,), jnp.inf, dtype=KEY_DTYPE)])
        val = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y]),
            val,
            jax.vmap(val.default)(jnp.arange(m * batch_size - n)),
        )
        key = key[: m * batch_size].reshape((m, batch_size))
        val = jax.tree_util.tree_map(
            lambda x: x[: m * batch_size].reshape((m, batch_size) + x.shape[1:]), val
        )
        return key, val

    @staticmethod
    def _next(current, target):
        """
        Get the next index.
        travel to target as heapify.
        """
        clz_current = jax.lax.clz(current)
        clz_target = jax.lax.clz(target)
        shift_amount = clz_current - clz_target - 1
        next_index = target.astype(SIZE_DTYPE) >> shift_amount
        return next_index

    @staticmethod
    def _insert_heapify(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue):
        """
        Insert a key-value pair into the heap.
        """
        last_node = heap.size // heap.batch_size

        def _cond(var):
            _, _, _, _, n = var
            return n < last_node

        def insert_heapify(var):
            key_store, val_store, keys, values, n = var
            head, hvalues, keys, values = BGPQ.merge_sort_split(
                key_store[n], val_store[n], keys, values
            )
            key_store = key_store.at[n].set(head)
            val_store = jax.tree_util.tree_map(lambda x, y: x.at[n].set(y), val_store, hvalues)
            return key_store, val_store, keys, values, BGPQ._next(n, last_node)

        heap.key_store, heap.val_store, keys, values, _ = jax.lax.while_loop(
            _cond,
            insert_heapify,
            (heap.key_store, heap.val_store, block_key, block_val, BGPQ._next(0, last_node)),
        )

        def _size_not_full(heap):
            # if size is not full, insert the keys and values to the heap
            # but size is full, last keys and values are not inserted to the heap
            # because, this 'last' keys and values are shoud be waste of memory.
            heap.key_store = heap.key_store.at[last_node].set(keys)
            heap.val_store = jax.tree_util.tree_map(
                lambda x, y: x.at[last_node].set(y), heap.val_store, values
            )
            return heap

        added = last_node < heap.branch_size
        heap = jax.lax.cond(added, _size_not_full, lambda heap: heap, heap)
        return heap, added

    @staticmethod
    @jax.jit
    def insert(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue, added_size: int = None):
        """
        Insert a key-value pair into the heap.
        """
        if added_size is None:
            added_size = jnp.sum(jnp.isfinite(block_key))
        root_key = heap.key_store[0]
        root_val = heap.val_store[0]
        root_key, root_val, block_key, block_val = BGPQ.merge_sort_split(
            root_key, root_val, block_key, block_val
        )
        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = jax.tree_util.tree_map(
            lambda x, r: x.at[0].set(r), heap.val_store, root_val
        )

        block_key, block_val, heap.key_buffer, heap.val_buffer, buffer_overflow = BGPQ.merge_buffer(
            block_key, block_val, heap.key_buffer, heap.val_buffer
        )

        heap, added = jax.lax.cond(
            buffer_overflow,
            BGPQ._insert_heapify,
            lambda heap, block_key, block_val: (heap, True),
            heap,
            block_key,
            block_val,
        )
        heap.size = heap.size + added_size * added
        return heap

    @staticmethod
    def delete_heapify(heap: "BGPQ"):
        size = SIZE_DTYPE(heap.size // heap.batch_size)
        last = size - 1

        heap.key_store = heap.key_store.at[0].set(heap.key_store[last]).at[last].set(jnp.inf)
        root_key, root_val, heap.key_buffer, heap.val_buffer = BGPQ.merge_sort_split(
            heap.key_store[0], heap.val_store[0], heap.key_buffer, heap.val_buffer
        )
        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = jax.tree_util.tree_map(
            lambda x, y: x.at[0].set(y), heap.val_store, root_val
        )

        def _lr(n):
            left_child = n * 2 + 1
            right_child = n * 2 + 2
            return left_child, right_child

        def _cond(var):
            key_store, val_store, c, l, r = var
            max_c = key_store[c][-1]
            min_l = key_store[l][0]
            min_r = key_store[r][0]
            min_lr = jnp.minimum(min_l, min_r)
            return max_c > min_lr

        def _f(var):
            key_store, val_store, current_node, left_child, right_child = var
            max_left_child = key_store[left_child][-1]
            max_right_child = key_store[right_child][-1]

            x, y = jax.lax.cond(
                max_left_child > max_right_child,
                lambda _: (left_child, right_child),
                lambda _: (right_child, left_child),
                None,
            )
            ky, vy, kx, vx = BGPQ.merge_sort_split(
                key_store[left_child],
                val_store[left_child],
                key_store[right_child],
                val_store[right_child],
            )
            key_store = key_store.at[x].set(kx)
            val_store = jax.tree_util.tree_map(lambda val, v1: val.at[x].set(v1), val_store, vx)
            kc, vc, ky, vy = BGPQ.merge_sort_split(
                key_store[current_node], val_store[current_node], ky, vy
            )
            key_store = key_store.at[current_node].set(kc).at[y].set(ky)
            val_store = jax.tree_util.tree_map(
                lambda val, v1, v2: val.at[c].set(v1).at[y].set(v2), val_store, vc, vy
            )

            nc = y
            nl, nr = _lr(y)
            return key_store, val_store, nc, nl, nr

        c = SIZE_DTYPE(0)
        l, r = _lr(c)
        heap.key_store, heap.val_store, _, _, _ = jax.lax.while_loop(
            _cond, _f, (heap.key_store, heap.val_store, c, l, r)
        )
        return heap

    @staticmethod
    @jax.jit
    def delete_mins(heap: "BGPQ"):
        """
        Delete the minimum key-value pair from the heap.
        In this function we did not clear the values, we just set the key to inf.
        """
        min_keys = heap.key_store[0].squeeze()
        min_values = jax.tree_util.tree_map(lambda x: x.squeeze(), heap.val_store[0])
        size = SIZE_DTYPE(heap.size // heap.batch_size)

        def make_empty(heap: "BGPQ"):
            root_key, root_val, heap.key_buffer, heap.val_buffer = BGPQ.merge_sort_split(
                jnp.full_like(heap.key_store[0], jnp.inf),
                heap.val_store[0],
                heap.key_buffer,
                heap.val_buffer,
            )
            heap.key_store = heap.key_store.at[0].set(root_key)
            heap.val_store = jax.tree_util.tree_map(
                lambda x, y: x.at[0].set(y), heap.val_store, root_val
            )
            return heap

        heap = jax.lax.cond(size > 1, BGPQ.delete_heapify, make_empty, heap)
        heap.size = heap.size - jnp.sum(jnp.isfinite(min_keys))
        return heap, min_keys, min_values
