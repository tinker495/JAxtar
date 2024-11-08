import jax
import jax.numpy as jnp
import click
import time

from JAxtar.hash import HashTable
from JAxtar.astar import astar_builder
from puzzle_config import default_puzzle_sizes, puzzle_dict, puzzle_dict_hard, puzzle_heuristic_dict, puzzle_heuristic_dict_nn

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

@click.command()
@click.option("-p","--puzzle_name", default="n-puzzle", type=click.Choice(puzzle_dict.keys()), help="Puzzle to solve")
@click.option("-h","--hard", default=False, is_flag=True, help="Use the hard puzzle")
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--max_node_size", default=2e7, help="Size of the puzzle")
@click.option("--batch_size", default=8192, help="Batch size for BGPQ") # 1024 * 8 = 8192
@click.option("--astar_weight", default=1.0 - 1e-3, help="Weight for the A* search")
@click.option("--start_state_seed", default=32, help="Seed for the random puzzle")
@click.option("--seed", default=0, help="Seed for the random puzzle")
@click.option("--vmap_size", default=1, help="Size for the vmap")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--profile", is_flag=True, help="Profile mode")
@click.option("-nn","--neural_heuristic", is_flag=True, help="Use neural heuristic")
@click.option("-ns","--not_show_path", default=False, is_flag=True, help="Not show the path")
def main(puzzle_name, hard, puzzle_size, max_node_size, batch_size, astar_weight, start_state_seed, seed, vmap_size, debug, profile, neural_heuristic, not_show_path):
    if debug:
        #disable jit
        print("Disabling JIT")
        jax.config.update('jax_disable_jit', True)
        
        # scale down the sizes for debuging
        max_node_size = 10000 
        batch_size = 100
        not_show_path = True
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle_name]
    else:
        puzzle_size = int(puzzle_size)

    if hard:
        puzzle = puzzle_dict_hard[puzzle_name](puzzle_size)
    else:
        puzzle = puzzle_dict[puzzle_name](puzzle_size)

    if neural_heuristic:
        try:
            heuristic = puzzle_heuristic_dict_nn[puzzle_name](puzzle_size, puzzle, False)
        except KeyError:
            print("Neural heuristic not available for this puzzle")
            exit(1)
    else:
        heuristic = puzzle_heuristic_dict[puzzle_name](puzzle)

    heuristic_fn = heuristic.distance

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)
    start_state_seed = int(start_state_seed)
    seed = int(seed)

    states = puzzle.get_target_state()[jnp.newaxis, ...]
    target = puzzle.get_target_state()

    astar_result_build, astar_fn = astar_builder(puzzle, heuristic_fn, batch_size, max_node_size, astar_weight=astar_weight)

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("initializing jit")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(astar_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),1))
    heuristics = jax.vmap(heuristic_fn, in_axes=(0, None))(states, target)

    print("Start state")
    print(states[0])
    print("Target state")
    print(target)
    print(f"Heuristic: {heuristics[0]:.2f}")

    if profile:
        print("Profiling")
        jax.profiler.start_trace("tmp/tensorboard")
    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("\n\nJIT compiled")
    start = time.time()
    astar_result, solved, solved_idx = astar_fn(astar_result_build(), states, filled, target)
    end = time.time()
    single_search_time = end - start
    print(f"Time: {single_search_time:6.2f} seconds")
    print(f"Search states: {human_format(astar_result.hashtable.size)} ({human_format(astar_result.hashtable.size / single_search_time)} states/s)\n\n")
    if profile:
        jax.profiler.stop_trace()
    
    if not not_show_path:
        if solved:
            print("Solution found\n\n")

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
                parant_last = parants[parant_last[0], parant_last[1]]
            for p in path[::-1]:
                state = table[p[0], p[1]]
                c = cost[p[0], p[1]]
                print(state)
                print(c)
            print(solved_st)
            print(f"Cost: {solved_cost}")

            print("\n\n")
        else:
            print("No solution found\n\n")
    else:
        if solved:
            solved_st = astar_result.hashtable.table[solved_idx.index, solved_idx.table_index][0]
            solved_cost = astar_result.cost[solved_idx.index, solved_idx.table_index][0]
            print(f"Cost: {solved_cost}")
            print("Solution found\n\n")
        else:
            print("No solution found\n\n")

    if vmap_size == 1:
        return
    
    states = puzzle.get_target_state()[jnp.newaxis, ...]
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)
    states, filled = jax.vmap(lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0)(states)
    vmapped_astar = jax.jit(jax.vmap(astar_fn, in_axes=(None, 0, 0, None)))
    print("initializing vmapped jit")
    start = time.time()
    astar_result, solved, solved_idx = vmapped_astar(astar_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    # for benchmark, same initial states
    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),1))
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)
    #states = jax.vmap(puzzle.get_initial_state, in_axes=0)(key=jax.random.split(jax.random.PRNGKey(start_state_seed),vmap_size))

    print("Vmapped A* search, multiple initial state solution")
    print("Start state")
    print(states[0], f"\n.\n.\n. x {vmap_size}")
    print("Target state")
    print(target)

    states, filled = jax.vmap(lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0)(states)

    print("vmap astar")
    print("# astar_result, solved, solved_idx = jax.vmap(astar_fn, in_axes=(None, 0, 0, None))(astar_result_build(), states, filled, target)")
    start = time.time()

    astar_result, solved, solved_idx = vmapped_astar(astar_result_build(), states, filled, target)
    end = time.time()
    vmapped_search_time = end - start # subtract jit time from the vmapped search time

    search_states = jnp.sum(astar_result.hashtable.size)

    print(f"Time: {vmapped_search_time:6.2f} seconds (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})")
    print(f"Search states: {human_format(search_states)} ({human_format(search_states / vmapped_search_time)} states/s)")
    print("Solution found:", f"{jnp.mean(solved)*100:.2f}%")
    # this means astart_fn is completely vmapable and jitable

if __name__ == "__main__":
    main()