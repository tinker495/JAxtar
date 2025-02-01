"""
Batched GPU Priority Queue (BGPQ) Implementation
This module provides a JAX-compatible priority queue optimized for GPU operations.
Key features:
- Fully batched operations for GPU efficiency
- Supports custom value types through dataclass
- Uses infinity padding for unused slots
- Maintains sorted order for efficient min/max operations
"""

from collections import namedtuple
from functools import partial

import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import KEY_DTYPE, SIZE_DTYPE
from JAxtar.util import set_array, set_tree

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys


def bgpq_value_dataclass(cls):
    """
    Decorator that creates a dataclass for HeapValue with additional functionality.
    Adds shape, dtype, getitem, and len properties to the class.

    Args:
        cls: The class to be decorated

    Returns:
        The decorated class with additional heap value functionality
    """
    cls = chex.dataclass(cls)

    shape_tuple = namedtuple("shape", cls.__annotations__.keys())

    def get_shape(self) -> shape_tuple:
        """Get shapes of all fields in the dataclass"""
        return shape_tuple(
            *[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        return type_tuple(
            *[
                jnp.dtype(getattr(self, field_name).dtype)
                for field_name in cls.__annotations__.keys()
            ]
        )

    setattr(cls, "dtype", property(get_type))

    def getitem(self, index):
        """Support indexing operations on the dataclass"""
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__getitem__"):
                new_values[field_name] = field_value[index]
            else:
                new_values[field_name] = field_value
        return cls(**new_values)

    setattr(cls, "__getitem__", getitem)

    def len(self):
        """Get length of the first field's first dimension"""
        return self.shape[0][0]

    setattr(cls, "__len__", len)

    # Ensure class has a default method for initialization
    assert hasattr(cls, "default"), "HeapValue class must have a default method."

    return cls


@bgpq_value_dataclass
class HeapValue:
    """
    Base class for heap values stored in the priority queue.
    This is a dummy implementation that should be subclassed with actual fields.
    Must implement the default() method to create default instances.
    """

    @staticmethod
    def default(_=None) -> "HeapValue":
        """Create a default instance of HeapValue"""
        pass


@chex.dataclass
class BGPQ:
    """
    Batched GPU Priority Queue implementation.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the queue can hold
        size: Current number of elements in the queue
        branch_size: Number of branches in the heap tree
        batch_size: Size of batched operations
        key_store: Array storing keys in a binary heap structure
        val_store: Array storing associated values
        key_buffer: Buffer for keys waiting to be inserted
        val_buffer: Buffer for values waiting to be inserted
    """

    max_size: int
    size: int
    branch_size: int
    batch_size: int
    key_store: chex.Array  # shape = (total_size, batch_size)
    val_store: HeapValue  # shape = (total_size, batch_size, ...)
    key_buffer: chex.Array  # shape = (batch_size - 1,)
    val_buffer: HeapValue  # shape = (batch_size - 1, ...)

    @staticmethod
    def build(total_size, batch_size, value_class=HeapValue):
        """
        Create a new BGPQ instance with specified capacity.

        Args:
            total_size: Total number of elements the queue can store
            batch_size: Size of batched operations
            value_class: Class to use for storing values (must implement default())

        Returns:
            BGPQ: A new priority queue instance initialized with empty storage
        """
        total_size = total_size
        # Calculate branch size, rounding up if total_size not divisible by batch_size
        branch_size = jnp.where(
            total_size % batch_size == 0, total_size // batch_size, total_size // batch_size + 1
        ).astype(SIZE_DTYPE)
        max_size = branch_size * batch_size
        size = SIZE_DTYPE(0)

        # Initialize storage arrays with infinity for unused slots
        key_store = jnp.full((branch_size, batch_size), jnp.inf, dtype=KEY_DTYPE)
        val_store = value_class.default((branch_size, batch_size))
        key_buffer = jnp.full((batch_size - 1,), jnp.inf, dtype=KEY_DTYPE)
        val_buffer = value_class.default((batch_size - 1,))

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
        Merge and split two sorted arrays while maintaining their relative order.
        This is a key operation for maintaining heap property in batched operations.

        Args:
            ak: First array of keys
            av: First array of values
            bk: Second array of keys
            bv: Second array of values

        Returns:
            tuple containing:
                - First half of merged and sorted keys
                - First half of corresponding values
                - Second half of merged and sorted keys
                - Second half of corresponding values
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
        Merge buffer contents with block contents, handling overflow conditions.

        This method is crucial for maintaining the heap property when inserting new elements.
        It handles the case where the buffer might overflow into the main storage.

        Args:
            blockk: Block keys array
            blockv: Block values
            bufferk: Buffer keys array
            bufferv: Buffer values

        Returns:
            tuple containing:
                - Updated block keys
                - Updated block values
                - Updated buffer keys
                - Updated buffer values
                - Boolean indicating if buffer overflow occurred
        """
        n = blockk.shape[0]
        # Concatenate block and buffer
        key = jnp.concatenate([blockk, bufferk])
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), blockv, bufferv)

        # Sort concatenated arrays
        idx = jnp.argsort(key, stable=SORT_STABLE)
        key = key[idx]
        val = jax.tree_util.tree_map(lambda x: x[idx], val)

        # Check for active elements (non-infinity)
        filled = jnp.isfinite(key)
        n_filled = jnp.sum(filled)
        buffer_overflow = n_filled >= n

        def overflowed(key, val):
            """Handle case where buffer overflows"""
            return key[:n], val[:n], key[n:], val[n:]

        def not_overflowed(key, val):
            """Handle case where buffer doesn't overflow"""
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
    @partial(jax.jit, static_argnums=(2))
    def make_batched(key: chex.Array, val: HeapValue, batch_size: int):
        """
        Convert unbatched arrays into batched format suitable for the queue.

        Args:
            key: Array of keys to batch
            val: HeapValue of values to batch
            batch_size: Desired batch size

        Returns:
            tuple containing:
                - Batched key array
                - Batched value array
        """
        n = key.shape[0]
        # Pad arrays to match batch size
        key = jnp.concatenate([key, jnp.full((batch_size - n,), jnp.inf, dtype=KEY_DTYPE)])
        val = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y]),
            val,
            val.default((batch_size - n,)),
        )
        return key, val

    @staticmethod
    def _next(current, target):
        """
        Calculate the next index in the heap traversal path.
        Uses leading zero count (clz) for efficient binary tree navigation.

        Args:
            current: Current index in the heap
            target: Target index to reach

        Returns:
            Next index in the path from current to target
        """
        clz_current = jax.lax.clz(current)
        clz_target = jax.lax.clz(target)
        shift_amount = clz_current - clz_target - 1
        next_index = target.astype(SIZE_DTYPE) >> shift_amount
        return next_index

    @staticmethod
    def _insert_heapify(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue):
        """
        Internal method to maintain heap property after insertion.
        Performs heapification by traversing up the tree and merging nodes.

        Args:
            heap: The priority queue instance
            block_key: Keys to insert
            block_val: Values to insert

        Returns:
            tuple containing:
                - Updated heap
                - Boolean indicating if insertion was successful
        """
        last_node = heap.size // heap.batch_size

        def _cond(var):
            """Continue while not reached last node"""
            _, _, _, _, n = var
            return n < last_node

        def insert_heapify(var):
            """Perform one step of heapification"""
            key_store, val_store, keys, values, n = var
            head, hvalues, keys, values = BGPQ.merge_sort_split(
                key_store[n], val_store[n], keys, values
            )
            key_store = set_array(key_store, head, n)
            val_store = set_tree(val_store, hvalues, n)
            return key_store, val_store, keys, values, BGPQ._next(n, last_node)

        heap.key_store, heap.val_store, keys, values, _ = jax.lax.while_loop(
            _cond,
            insert_heapify,
            (heap.key_store, heap.val_store, block_key, block_val, BGPQ._next(0, last_node)),
        )

        def _size_not_full(heap):
            """Insert remaining elements if heap not full"""
            heap.key_store = set_array(heap.key_store, keys, last_node)
            heap.val_store = set_tree(heap.val_store, values, last_node)
            return heap

        added = last_node < heap.branch_size
        heap = jax.lax.cond(added, _size_not_full, lambda heap: heap, heap)
        return heap, added

    @jax.jit
    def insert(heap: "BGPQ", block_key: chex.Array, block_val: HeapValue, added_size: int = None):
        """
        Insert new elements into the priority queue.
        Maintains heap property through merge operations and heapification.

        Args:
            heap: The priority queue instance
            block_key: Keys to insert
            block_val: Values to insert
            added_size: Optional size of insertion (calculated if None)

        Returns:
            Updated heap instance
        """
        if added_size is None:
            added_size = jnp.sum(jnp.isfinite(block_key))

        # Merge with root node
        root_key = heap.key_store[0]
        root_val = heap.val_store[0]
        root_key, root_val, block_key, block_val = BGPQ.merge_sort_split(
            root_key, root_val, block_key, block_val
        )
        heap.key_store = set_array(heap.key_store, root_key, 0)
        heap.val_store = set_tree(heap.val_store, root_val, 0)

        # Handle buffer overflow
        block_key, block_val, heap.key_buffer, heap.val_buffer, buffer_overflow = BGPQ.merge_buffer(
            block_key, block_val, heap.key_buffer, heap.val_buffer
        )

        # Perform heapification if needed
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
        """
        Maintain heap property after deletion of minimum elements.

        Args:
            heap: The priority queue instance

        Returns:
            Updated heap instance
        """
        size = SIZE_DTYPE(heap.size // heap.batch_size)
        last = size - 1

        # Move last node to root and clear last position
        last_key = heap.key_store[last]
        last_val = heap.val_store[last]

        heap.key_store = set_array(heap.key_store, jnp.inf, last)

        root_key, root_val, heap.key_buffer, heap.val_buffer = BGPQ.merge_sort_split(
            last_key, last_val, heap.key_buffer, heap.val_buffer
        )
        heap.key_store = set_array(heap.key_store, root_key, 0)
        heap.val_store = set_tree(heap.val_store, root_val, 0)

        def _lr(n):
            """Get left and right child indices"""
            left_child = n * 2 + 1
            right_child = n * 2 + 2
            return left_child, right_child

        def _cond(var):
            """Continue while heap property is violated"""
            key_store, val_store, c, l, r = var
            max_c = key_store[c][-1]
            min_l = key_store[l][0]
            min_r = key_store[r][0]
            min_lr = jnp.minimum(min_l, min_r)
            return max_c > min_lr

        def _f(var):
            """Perform one step of heapification"""
            key_store, val_store, current_node, left_child, right_child = var
            max_left_child = key_store[left_child][-1]
            max_right_child = key_store[right_child][-1]

            # Choose child with smaller key
            x, y = jax.lax.cond(
                max_left_child > max_right_child,
                lambda _: (left_child, right_child),
                lambda _: (right_child, left_child),
                None,
            )

            # Merge and swap nodes
            ky, vy, kx, vx = BGPQ.merge_sort_split(
                key_store[left_child],
                val_store[left_child],
                key_store[right_child],
                val_store[right_child],
            )
            kc, vc, ky, vy = BGPQ.merge_sort_split(
                key_store[current_node], val_store[current_node], ky, vy
            )
            key_store = set_array(set_array(set_array(key_store, ky, y), kc, current_node), kx, x)
            val_store = set_tree(set_tree(set_tree(val_store, vy, y), vc, current_node), vx, x)

            nc = y
            nl, nr = _lr(y)
            return key_store, val_store, nc, nl, nr

        c = SIZE_DTYPE(0)
        l, r = _lr(c)
        heap.key_store, heap.val_store, _, _, _ = jax.lax.while_loop(
            _cond, _f, (heap.key_store, heap.val_store, c, l, r)
        )
        return heap

    @jax.jit
    def delete_mins(heap: "BGPQ"):
        """
        Remove and return the minimum elements from the queue.

        Args:
            heap: The priority queue instance

        Returns:
            tuple containing:
                - Updated heap instance
                - Array of minimum keys removed
                - HeapValue of corresponding values
        """
        min_keys = heap.key_store[0]
        min_values = heap.val_store[0]
        size = SIZE_DTYPE(heap.size // heap.batch_size)

        def make_empty(heap: "BGPQ"):
            """Handle case where heap becomes empty"""
            root_key, root_val, heap.key_buffer, heap.val_buffer = BGPQ.merge_sort_split(
                jnp.full_like(heap.key_store[0], jnp.inf),
                heap.val_store[0],
                heap.key_buffer,
                heap.val_buffer,
            )
            heap.key_store = set_array(heap.key_store, root_key, 0)
            heap.val_store = set_tree(heap.val_store, root_val, 0)
            return heap

        heap = jax.lax.cond(size > 1, BGPQ.delete_heapify, make_empty, heap)
        heap.size = heap.size - jnp.sum(jnp.isfinite(min_keys))
        return heap, min_keys, min_values
