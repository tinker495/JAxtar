import operator

import chex
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict, Type, TypeVar
from puzzle.puzzle_base import Puzzle

T = TypeVar('T')
HASH_SIZE_MULTIPLIER = 2

def rotl(x, n):
    return (x << n) | (x >> (32 - n))

def hash_func_builder(x: Puzzle.State):
    """
    build a hash function for the state dataclass
    """
    default = x.default() # get the default state, reference for building the hash function

    @jax.jit
    def xxhash(x, seed):
        prime_1 = jnp.uint32(0x9E3779B1)
        prime_2 = jnp.uint32(0x85EBCA77)
        prime_3 = jnp.uint32(0xC2B2AE3D)
        prime_5 = jnp.uint32(0x165667B1)
        acc = jnp.uint32(seed) + prime_5
        for _ in range(4):
            lane = x & 255
            acc = (acc + lane * prime_5)
            acc = rotl(acc, 11) * prime_1
            x = x >> 8
        acc = acc ^ (acc >> 15)
        acc = acc * prime_2
        acc = acc ^ (acc >> 13)
        acc = acc * prime_3
        acc = acc ^ (acc >> 16)
        return acc

    def _get_leaf_hash_func(leaf):
        flatten_leaf = jnp.reshape(leaf, (-1,))
        bitlen = flatten_leaf.dtype.itemsize
        flatten_len = flatten_leaf.shape[0]
        chunk = int(jnp.maximum(jnp.ceil(4 / bitlen), 1))
        pad_len = jnp.where(flatten_len % chunk != 0, chunk - (flatten_len % chunk), 0)
        reshape_size = ((flatten_len + pad_len) // chunk, chunk)

        @jax.jit
        def _to_uint32(x):
            x = jnp.reshape(x, (flatten_len, ))
            x_padded = jnp.pad(x, (0, pad_len), mode='constant', constant_values=0)
            x_reshaped = jnp.reshape(x_padded, reshape_size)
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(x_reshaped).reshape(-1)

        @jax.jit
        def _h(x, seed):
            x = _to_uint32(x)
            def scan_body(seed, x): # scan body for the xxhash function
                result = xxhash(x, seed)
                return result, result
            final_result, _ = jax.lax.scan(scan_body, seed, x)
            return final_result
        
        return _h

    tree_flatten_func = jax.tree_util.tree_map(_get_leaf_hash_func, default) # each leaf has a hash function for each array shape and dtype
    def _h(x, seed): # hash function for the whole tree structure
        return jax.tree_util.tree_reduce(xxhash, jax.tree_util.tree_map(lambda f,l: f(l, seed), tree_flatten_func, x))
    return jax.jit(_h)

@chex.dataclass
class HashTable:
    """
    Cuckoo Hash Table
    """

    seed: int
    capacity: int
    _capacity: int
    size: int
    n_table: int # number of tables
    table: Puzzle.State # shape = State("args" = (capacity, cuckoo_len, ...), ...) 
    table_idx: chex.Array # shape = (capacity, ) dtype = jnp.uint4 is the index of the table in the cuckoo table.

    @staticmethod
    def build(statecls: Puzzle.State, seed: int, capacity: int, n_table: int = 2):
        """
        make a lookup table with the default state of the statecls
        """
        _capacity = jnp.array(HASH_SIZE_MULTIPLIER * capacity//n_table, jnp.uint32) # make the capacity a little bit bigger than the given capacity to avoid the infinite loop
        table = jax.vmap(jax.vmap(statecls.default))(jnp.zeros((_capacity, n_table)))
        table_idx = jnp.zeros((_capacity), dtype=jnp.uint8)
        return HashTable(seed=seed,
                        capacity=capacity,
                        _capacity=_capacity,
                        size=jnp.uint32(0),
                        n_table=n_table,
                        table=table,
                        table_idx=table_idx)

    @staticmethod
    def get_new_idx(hash_func: callable, table: "HashTable", input: Puzzle.State, seed: int):
        hash_value = hash_func(input, seed)
        idx = hash_value % table._capacity
        return idx

    @staticmethod
    def _lookup(hash_func: callable, table: "HashTable", input: Puzzle.State, idx: int, table_idx: int, seed: int, found: bool):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the unfilled index, cuckoo_idx and False
        this function could be used to check if the state is in the table or not, and insert it if it is not.
        """
        def _check_equal(state1, state2):
            tree_equal = jax.tree.map(lambda x, y: jnp.all(x == y), state1, state2)
            return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)
        
        def _cond(val):
            seed, idx, table_idx, found = val
            filled_idx = table.table_idx[idx]
            in_empty = table_idx >= filled_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val):
            seed, idx, table_idx, found = val

            def get_new_idx_and_table_idx(seed, idx, table_idx):
                next_table = table_idx >= (table.n_table - 1)
                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    lambda _: (seed+1, HashTable.get_new_idx(hash_func, table, input, seed+1), 0),
                    lambda _: (seed, idx, table_idx+1),
                    None
                )
                return seed, idx, table_idx

            state = table.table[idx, table_idx]
            found = _check_equal(state, input)
            seed, idx, table_idx = jax.lax.cond(
                found,
                lambda _: (seed, idx, table_idx),
                lambda _: get_new_idx_and_table_idx(seed, idx, table_idx),
                None
            )
            return seed, idx, table_idx, found
        
        state = table.table[idx, table_idx]
        found = jnp.logical_or(found, _check_equal(state, input))
        update_seed, idx, table_idx, found = jax.lax.while_loop(
            _cond,
            _while,
            (seed, idx, table_idx, found)
        )
        return update_seed, idx, table_idx, found
    
    @staticmethod
    def lookup(hash_func: callable, table: "HashTable", input: Puzzle.State):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the
        """
        index = HashTable.get_new_idx(hash_func, table, input, table.seed)
        _, idx, table_idx, found = HashTable._lookup(hash_func, table, input, index, 0, table.seed, False)
        return idx, table_idx, found

    @staticmethod
    def insert(hash_func: callable, table: "HashTable", input: Puzzle.State):
        """
        insert the state in the table
        """

        def _update_table(table: "HashTable", input: Puzzle.State, idx: int, table_idx: int):
            """
            insert the state in the table
            """
            table.table = jax.tree_util.tree_map(lambda x, y: x.at[idx,table_idx].set(y), table.table, input)
            table.table_idx = table.table_idx.at[idx].add(1)
            return table

        idx, table_idx, found = HashTable.lookup(hash_func, table, input)
        return jax.lax.cond(
            found,
            lambda _: table,
            lambda _: _update_table(table, input, idx, table_idx),
            None
        ), ~found
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0,2,))
    def make_batched(statecls: Puzzle.State, inputs: Puzzle.State, batch_size: int):
        """
        make a batched version of the inputs
        """
        count = len(inputs)
        batched = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), inputs, jax.vmap(statecls.default)(jnp.arange(batch_size - count)))
        filled = jnp.concatenate([jnp.ones(count), jnp.zeros(batch_size - count)], dtype=jnp.bool_)
        return batched, filled

    @staticmethod
    def parallel_insert(hash_func: callable, table: "HashTable", inputs: Puzzle.State, filled: chex.Array):
        """
        insert the states in the table at the same time
        """

        def _next_idx(seeds, _idxs, unupdateds):

            def get_new_idx_and_table_idx(seed, idx, table_idx, state):
                next_table = table_idx >= (table.n_table - 1)

                def next_table_fn(seed, table):
                    next_idx = HashTable.get_new_idx(hash_func, table, state, seed)
                    seed = seed + 1
                    return seed, next_idx, table.table_idx[next_idx].astype(jnp.uint32)
                
                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    next_table_fn,
                    lambda seed, _: (seed, idx, table_idx + 1),
                    seed, table
                )
                return seed, idx, table_idx
            
            idxs = _idxs[:, 0]
            table_idxs = _idxs[:, 1]
            seeds, idxs, table_idxs = jax.vmap(
                lambda unupdated, seed, idx, table_idx, state: 
                    jax.lax.cond(
                        unupdated,
                        lambda _: get_new_idx_and_table_idx(seed, idx, table_idx, state),
                        lambda _: (seed, idx, table_idx),
                        None
                    )
            )(unupdateds, seeds, idxs, table_idxs, inputs)
            _idxs = jnp.stack((idxs, table_idxs), axis=1)
            return seeds, _idxs

        def _cond(val):
            _, _, unupdated = val
            return jnp.any(unupdated)
        
        def _while(val):
            seeds, _idxs, unupdated = val
            seeds, _idxs = _next_idx(seeds, _idxs, unupdated)

            overflowed = _idxs[:, 1] >= table.n_table  # Overflowed index must be updated
            masked_idx = jnp.where(updatable[:, jnp.newaxis], _idxs, jnp.full_like(_idxs, -1))
            unique_idxs = jnp.unique(masked_idx, axis=0, size=batch_len, return_index=True)[1] # val = (unique_len, 2), unique_idxs = (unique_len,)
            not_uniques = jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False) # set the unique index to True

            unupdated = jnp.logical_and(updatable, not_uniques)
            unupdated = jnp.logical_or(unupdated, overflowed)
            return seeds, _idxs, unupdated

        initial_idx = jax.vmap(partial(HashTable.get_new_idx, hash_func),
                        in_axes=(
                                None,
                                0,
                                None))(
                        table,
                        inputs,
                        table.seed
                        )
        batch_len = initial_idx.shape[0]
        seeds, idx, table_idx, found = jax.vmap(partial(HashTable._lookup, hash_func),
                                        in_axes=(
                                                None,
                                                0,
                                                0,
                                                None,
                                                None,
                                                0))(
                                        table,
                                        inputs,
                                        initial_idx,
                                        0,
                                        table.seed,
                                        ~filled
                                        )
        _idxs = jnp.stack([idx, table_idx], axis=1)
        updatable = jnp.logical_and(~found, filled)

        masked_idx = jnp.where(updatable[:, jnp.newaxis], _idxs, jnp.full_like(_idxs, -1))
        unique_idxs = jnp.unique(masked_idx, axis=0, size=batch_len, return_index=True)[1] # val = (unique_len, 2), unique_idxs = (unique_len,)
        not_uniques = jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False) # set the unique index to True
        unupdated = jnp.logical_and(updatable, not_uniques) # remove the unique index from the unupdated index

        seeds, _idxs, _  = jax.lax.while_loop(
            _cond,
            _while,
            (seeds, _idxs, unupdated)
        )

        idx, table_idx = _idxs[:, 0], _idxs[:, 1]
        table.table = jax.tree_util.tree_map(lambda x, y: x.at[idx, table_idx].set(jnp.where(updatable.reshape(-1, *([1] * (len(y.shape) - 1))), y, x[idx, table_idx])), table.table, inputs)
        table.table_idx = table.table_idx.at[idx].add(updatable)
        table.size += jnp.sum(updatable)

        #get the idx and table_idx of the inputs
        _, idx, table_idx, _ = jax.vmap(partial(HashTable._lookup, hash_func),
                                        in_axes=(
                                                None,
                                                0,
                                                0,
                                                None,
                                                None,
                                                0))(
                                        table,
                                        inputs,
                                        initial_idx,
                                        0,
                                        table.seed,
                                        ~filled
                                        )
        return table, updatable, idx, table_idx