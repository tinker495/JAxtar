import jax
import jax.numpy as jnp
import click
import time

from functools import partial
from puzzle.slidepuzzle import SlidePuzzle
from JAxtar.hash import HashTable
from JAxtar.astar import astar_builder
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from heuristic.slidepuzzle_neural_heuristic import SlidePuzzleNeuralHeuristic


puzzle_dict = {
    "n-puzzle": lambda _: (SlidePuzzle(4), SlidePuzzleHeuristic(SlidePuzzle(4)).distance),
    "n-puzzle-nn": lambda _: (SlidePuzzle(4), SlidePuzzleNeuralHeuristic(SlidePuzzle(4)).distance)
}

@click.command()
@click.option("--puzzle", default="n-puzzle", type=click.Choice(puzzle_dict.keys()), help="Puzzle to solve")
@click.option("--max_node_size", default=2e7, help="Size of the puzzle")
@click.option("--batch_size", default=10000, help="Batch size for BGPQ")
@click.option("--astar_weight", default=1.0 - 1e-3, help="Weight for the A* search")
@click.option("--start_state_seed", default=32, help="Seed for the random puzzle")
@click.option("--seed", default=0, help="Seed for the random puzzle")
@click.option("--vmap_size", default=1, help="Size for the vmap")
def main(puzzle, max_node_size, batch_size, astar_weight, start_state_seed, seed, vmap_size):
    puzzle, heuristic_fn = puzzle_dict[puzzle](None)

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)
    start_state_seed = int(start_state_seed)
    seed = int(seed)

    states = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15], dtype=jnp.uint8))[jnp.newaxis, ...]
    target = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=jnp.uint8))

    astar_fn = astar_builder(puzzle, heuristic_fn, batch_size, max_node_size, astar_weight=astar_weight)

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("initializing jit")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),1))
    target = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=jnp.uint8))

    print("Start state")
    print(states[0])
    print("Target state")
    print(target)

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("\n\nJIT compiled")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(states, filled, target)
    end = time.time()
    single_search_time = end - start
    print(f"Time: {single_search_time:6.2f} seconds")
    print(f"Search states: {astar_result.hashtable.size} ({astar_result.hashtable.size / single_search_time:.2f} states/s)\n\n")

    if not solved:
        print("No solution found\n\n")
    else:
        print("Solution found\n\n")

        searched_states = astar_result.hashtable.size

        print(f"Search states: {searched_states}\n\n")
        parants = astar_result.parant
        table = astar_result.hashtable.table
        cost = astar_result.cost

        solved_st = astar_result.hashtable.table[solved_idx.index, solved_idx.table_index][0]
        solved_cost = astar_result.cost[solved_idx.index, solved_idx.table_index][0]

        path = []
        parant_last = parants[solved_idx.index, solved_idx.table_index][0]
        while True:
            if parant_last[0] == -1:
                break
            path.append(parant_last)
            parant_last = parants[*parant_last]
        for p in path[::-1]:
            state = table[p[0], p[1]]
            c = cost[p[0], p[1]]
            print(state)
            print(c)
        print(solved_st)
        print(solved_cost)

        print("\n\n")

    #astar_fn = astar_builder(puzzle, heuristic_fn, batch_size, max_node_size//vmap_size, astar_weight=astar_weight) # 10 times smaller size for memory usage
    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),vmap_size))

    print("Vmapped A* search, multiple initial state solution")
    print("Start state")
    print(states[0], f"\n.\n.\n. x {vmap_size}")
    print("Target state")
    print(target)

    states, filled = jax.vmap(lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0)(states)

    print("vmap astar")
    print("# astar_result, solved, solved_idx = jax.vmap(astar_fn, in_axes=(0, 0, None))(states, filled, target)")
    start = time.time()

    astar_result, solved, solved_idx = jax.vmap(astar_fn, in_axes=(0, 0, None))(states, filled, target)
    end = time.time()
    vmapped_search_time = end - start

    search_states = jnp.sum(astar_result.hashtable.size)

    print(f"Time: {vmapped_search_time:6.2f} seconds (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})")
    print(f"Search states: {search_states} ({search_states / vmapped_search_time:.2f} states/s)")
    print("Solution found:", f"{jnp.mean(solved)*100:.2f}%")
    # this means astart_fn is completely vmapable and jitable

if __name__ == "__main__":
    main()