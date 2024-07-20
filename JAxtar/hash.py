import operator

import chex
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict, Type, TypeVar
from puzzle.puzzle_base import Puzzle

T = TypeVar('T')

def rotl(x, n):
    return (x << n) | (x >> (32 - n))

capacity = int(1024 * 1024 * 2)
max_try = 128

def hash_func_builder(x: Puzzle.State):
    """
    build a hash function for the state dataclass
    """
    default = x.default() # get the default state, reference for building the hash function

    def _get_leaf_hash_func(leaf):
        flatten_leaf = jnp.reshape(leaf, (-1,))
        bitlen = flatten_leaf.dtype.itemsize
        div = jnp.maximum(4 // bitlen, 1).astype(int)
        pad_len = jax.lax.cond(
            flatten_leaf.shape[0] % div == 0,
            lambda _: 0,
            lambda _: (div - flatten_leaf.shape[0] % div).astype(int),
            None
        )
        paded_len = len(flatten_leaf) + pad_len
        chunk = int(paded_len // div)

        def _to_uint32(x):
            x = jnp.reshape(x, (-1,))
            x_padded = jnp.pad(x, (0, pad_len), mode='constant', constant_values=0)
            x_reshaped = jnp.reshape(x_padded, (-1, div))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(x_reshaped).reshape(-1)

        def xxhash(x, seed):
            prime_1 = jnp.uint32(0x9E3779B1)
            prime_2 = jnp.uint32(0x85EBCA77)
            prime_3 = jnp.uint32(0xC2B2AE3D)
            prime_5 = jnp.uint32(0x165667B1)
            acc = jnp.uint32(seed) + prime_5
            for _ in range(4):
                lane = x & 255
                acc = acc + lane * prime_5
                acc = rotl(acc, 11) * prime_1
                x = x >> 8
            acc = acc ^ (acc >> 15)
            acc = acc * prime_2
            acc = acc ^ (acc >> 13)
            acc = acc * prime_3
            acc = acc ^ (acc >> 16)
            return acc

        def leaf_hashing(x, seed): # max capacity is 2 ** 24
            z = xxhash(x, seed) >> 8
            b1 = z & 255
            z = z >> 8
            b2 = z & 255
            b3 = z >> 8
            c1 = (b1 * capacity) >> 24
            c2 = (b2 * capacity) >> 16
            c3 = (b3 * capacity) >> 8
            return c1 + c2 + c3

        def _h(x, seed):
            # sum of hash * index for collision
            x = _to_uint32(x)
            hashs = jax.vmap(leaf_hashing, in_axes=(0, None))(x, seed)
            return jnp.sum(hashs * jnp.arange(1, chunk+1), dtype=jnp.uint32)
        
        return _h

    tree_flatten_func = jax.tree_map(_get_leaf_hash_func, default) # each leaf has a hash function for each array shape and dtype
    def _h(x, seed): # sum of all the hash functions for each leaf
        return jax.tree_util.tree_reduce(operator.add, jax.tree_map(lambda f,l: f(l, seed), tree_flatten_func, x))
    return jax.jit(_h)

@chex.dataclass
class HashTable:
    """
    Cuckoo Hash Table
    """

    seed: int
    capacity: int
    n_table: int # number of tables
    table: Puzzle.State # shape = State("args" = (capacity, cuckoo_len, ...), ...) 
    table_idx: chex.Array # shape = (capacity, ) dtype = jnp.uint4 is the index of the table in the cuckoo table.

    @staticmethod
    def make_lookup_table(statecls: Puzzle.State, seed: int, capacity: int, n_table: int = 2):
        """
        make a lookup table with the default state of the statecls
        """
        table = jax.vmap(jax.vmap(statecls.default))(jnp.zeros((capacity, n_table)))
        table_idx = jnp.zeros((capacity), dtype=jnp.uint8)
        return HashTable(seed=seed,
                        capacity=capacity,
                        n_table=n_table,
                        table=table,
                        table_idx=table_idx)



    @staticmethod
    def get_new_idx(hash_func: callable, table: "HashTable", input: Puzzle.State, seed: int):
        hash_value = hash_func(input, seed)
        idx = hash_value % table.capacity
        return idx

    @staticmethod
    def _lookup(hash_func: callable, table: "HashTable", input: Puzzle.State, idx: int, table_idx: int, seed: int, found: bool):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the unfilled index, cuckoo_idx and False
        this function could be used to check if the state is in the table or not, and insert it if it is not.
        """
        
        def _cond(val):
            seed, idx, table_idx, found = val
            filled_idx = table.table_idx[idx]
            in_empty = filled_idx == table_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val):
            seed, idx, table_idx, found = val

            def get_new_idx_and_table_idx(seed, idx, table_idx):
                next_table = table_idx >= table.n_table - 1
                idx, table_idx, seed = jax.lax.cond(
                    next_table,
                    lambda _: (HashTable.get_new_idx(hash_func, table, state, seed+1), 0, seed+1),
                    lambda _: (idx, table_idx+1, seed),
                    None
                )
                return idx, table_idx, seed
            
            def _check_equal(state1, state2):
                tree_equal = jax.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
                return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

            state = table.table[idx, table_idx]
            found = _check_equal(state, input)
            idx, table_idx, seed = jax.lax.cond(
                found,
                lambda _: (idx, table_idx, seed),
                lambda _: get_new_idx_and_table_idx(seed, idx, table_idx),
                None
            )
            return seed, idx, table_idx, found
        
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
            table.table = jax.tree_map(lambda x, y: x.at[idx,table_idx].set(y), table.table, input)
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
    def parallel_insert(hash_func: callable, table: "HashTable", inputs: Puzzle.State, filled: chex.Array):
        """
        insert the states in the table at the same time
        """
        def _get_next_indexs(table: "HashTable", inputs: Puzzle.State, idx, table_idx, seeds):
            seeds, idx, table_idx, found = jax.vmap(partial(HashTable._lookup, hash_func), in_axes=(None, 0, 0, 0, 0, None))(table, inputs, idx, table_idx, seeds, False)
            idxs = jnp.stack([idx, table_idx], axis=1)
            return seeds, idxs, idx, table_idx, ~found

        def _update_table(table: "HashTable", inputs: Puzzle.State, idx, table_idx, updatable):
            table.table = jax.tree_map(lambda x, y: x.at[idx,table_idx].set(jnp.where(updatable.reshape(-1, 1), y, x[idx,table_idx])), table.table, inputs)
            table.table_idx = table.table_idx.at[idx].add(updatable)
            return table
        
        def _cond(val):
            _, _, _, _, unupdated, _ = val
            return jnp.any(unupdated)
        
        def _while(val):
            seeds, idxs, idx, table_idx, unupdated, table = val
            idxs = jnp.where(unupdated.reshape(-1,1), idxs, jnp.ones_like(idxs) * table.capacity + 1) # set the idxs to the capacity + 1 if it is not updated
            _, unique_idxs = jnp.unique(idxs, axis=0, size=batch_len, return_index=True) # val = (unique_len, 2), unique_idxs = (unique_len,)
            unique_update = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(True) # set the unique index to True
            updatable = jnp.logical_and(unupdated, unique_update) # only update the unique index
            table = _update_table(table, inputs, idx, table_idx, updatable)
            seeds, idxs, idx, table_idx, unupdated = _get_next_indexs(table, inputs, idx, table_idx, seeds)
            unupdated = jnp.logical_and(unupdated, ~unique_update)
            return seeds, idxs, idx, table_idx, unupdated, table

        idxs = jax.vmap(jax.jit(partial(HashTable.get_new_idx, hash_func)), in_axes=(None, 0, None))(table, inputs, table.seed)
        batch_len = idxs.shape[0]
        seeds, idx, table_idx, found = jax.vmap(jax.jit(partial(HashTable._lookup, hash_func)), in_axes=(None, 0, 0, None, None, 0))(table, inputs, idxs, 0, table.seed, ~filled)
        idxs = jnp.stack([idx, table_idx], axis=1)
        inserted = ~found
        seeds, _, _, _, _, table  = jax.lax.while_loop(
            _cond,
            _while,
            (seeds, idxs, idx, table_idx, inserted, table)
        )
        return table, inserted