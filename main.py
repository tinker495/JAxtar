import jax
import jax.numpy as jnp
import click
import time

from puzzle.slidepuzzle import SlidePuzzle
from JAxtar.hash import HashTable
from JAxtar.astar import astar_builder
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic


puzzle_dict = {
    "n-puzzle": lambda _: (SlidePuzzle(4), SlidePuzzleHeuristic(SlidePuzzle(4)).distance)
}

@click.command()
@click.option("--puzzle", default="n-puzzle", type=click.Choice(puzzle_dict.keys()), help="Puzzle to solve")
@click.option("--max_node_size", default=2e7, help="Size of the puzzle")
@click.option("--batch_size", default=10000, help="Batch size for BGPQ")
@click.option("--start_state_seed", default=32, help="Seed for the random puzzle")
@click.option("--seed", default=0, help="Seed for the random puzzle")
def main(puzzle, max_node_size, batch_size, start_state_seed, seed):
    puzzle, heuristic_fn = puzzle_dict[puzzle](None)

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)
    start_state_seed = int(start_state_seed)
    seed = int(seed)

    states = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15], dtype=jnp.uint8))[jnp.newaxis, ...]
    target = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=jnp.uint8))

    print("Start state\n\n")
    print(states[0])
    print("Target state\n\n")
    print(target)

    astar_fn = astar_builder(puzzle, heuristic_fn, batch_size, max_node_size)

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("initializing jit\n\n")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),1))
    target = puzzle.State(board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=jnp.uint8))

    print("Start state\n\n")
    print(states[0])
    print("Target state\n\n")
    print(target)

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("JIT compiled\n\n")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    if not solved:
        print("No solution found\n\n")
        return
    
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
    for i in range(100):
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

if __name__ == "__main__":
    main()