import time
from functools import wraps

import click
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from JAxtar.astar import astar_builder
from JAxtar.hash import HashTable
from JAxtar.qstar import qstar_builder
from puzzle_config import (
    default_puzzle_sizes,
    puzzle_dict,
    puzzle_dict_hard,
    puzzle_heuristic_dict,
    puzzle_heuristic_dict_nn,
    puzzle_q_dict,
    puzzle_q_dict_nn,
)
from qfunction.q_base import QFunction


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


@click.group()
def main():
    pass


def puzzle_options(func: callable) -> callable:
    @click.option(
        "-p",
        "--puzzle",
        default="n-puzzle",
        type=click.Choice(puzzle_dict.keys()),
        help="Puzzle to solve",
    )
    @click.option("-h", "--hard", default=False, is_flag=True, help="Use the hard puzzle")
    @click.option("-ps", "--puzzle_size", default="default", type=str, help="Size of the puzzle")
    @click.option("--start_state_seed", default=32, help="Seed for the random puzzle")
    @click.option("--seed", default=0, help="Seed for the random puzzle")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def search_options(func: callable) -> callable:
    @click.option("-m", "--max_node_size", default=2e7, help="Size of the puzzle")
    @click.option("-b", "--batch_size", default=8192, help="Batch size for BGPQ")  # 1024 * 8 = 8192
    @click.option("-w", "--cost_weight", default=1.0 - 1e-3, help="Weight for the A* search")
    @click.option("-v", "--vmap_size", default=1, help="Size for the vmap")
    @click.option("--debug", is_flag=True, help="Debug mode")
    @click.option("--profile", is_flag=True, help="Profile mode")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def heuristic_options(func: callable) -> callable:
    @click.option("-nn", "--neural_heuristic", is_flag=True, help="Use neural heuristic")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def qfunction_options(func: callable) -> callable:
    @click.option("-nn", "--neural_qfunction", is_flag=True, help="Use neural q function")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def visualize_options(func: callable) -> callable:
    @click.option("-v", "--visualize", is_flag=True, help="Visualize the path")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar(
    puzzle,
    hard,
    puzzle_size,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seed,
    seed,
    vmap_size,
    debug,
    profile,
    neural_heuristic,
    visualize,
):
    if debug:
        # disable jit
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)

        # scale down the sizes for debugging
        max_node_size = 10000
        batch_size = 100
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle)

    puzzle_name = puzzle
    if hard:
        puzzle = puzzle_dict_hard[puzzle](puzzle_size)
    else:
        puzzle = puzzle_dict[puzzle](puzzle_size)

    if neural_heuristic:
        try:
            heuristic: Heuristic = puzzle_heuristic_dict_nn[puzzle_name](puzzle_size, puzzle, False)
        except KeyError:
            print("Neural heuristic not available for this puzzle")
            print(f"list of neural heuristic: {puzzle_heuristic_dict_nn.keys()}")
            exit(1)
    else:
        heuristic: Heuristic = puzzle_heuristic_dict[puzzle_name](puzzle)

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)
    start_state_seed = int(start_state_seed)
    seed = int(seed)

    states = puzzle.get_target_state()[jnp.newaxis, ...]
    target = puzzle.get_target_state()

    search_result_build, astar_fn = astar_builder(
        puzzle, heuristic, batch_size, max_node_size, cost_weight=cost_weight
    )

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("initializing jit")
    start = time.time()
    search_result, solved, solved_idx = astar_fn(search_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds")
    print("JIT compiled\n\n")

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(
        key=jax.random.split(jax.random.PRNGKey(start_state_seed), 1)
    )
    heuristic_values = heuristic.batched_distance(states, target)

    print("Start state")
    print(states[0])
    print("Target state")
    print(target)
    print(f"Heuristic: {heuristic_values[0]:.2f}")

    if profile:
        print("Profiling")
        jax.profiler.start_trace("tmp/tensorboard")
    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    start = time.time()
    search_result, solved, solved_idx = astar_fn(search_result_build(), states, filled, target)
    end = time.time()
    single_search_time = end - start
    print(f"Time: {single_search_time:6.2f} seconds")
    print(
        f"Search states: {human_format(search_result.hashtable.size)}"
        f"({human_format(search_result.hashtable.size / single_search_time)} states/s)\n\n"
    )
    if profile:
        jax.profiler.stop_trace()

    if visualize:
        if solved:
            print("Solution found\n\n")

            parents = search_result.parent
            parent_action = search_result.parent_action
            table = search_result.hashtable.table
            cost = search_result.cost

            solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index][0]
            solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index][0]

            path = [(solved_idx.index[0], solved_idx.table_index[0])]
            parent_last = parents[solved_idx.index, solved_idx.table_index][0]
            while True:
                if parent_last[0] == -1:
                    break
                path.append(parent_last)
                parent_last = parents[parent_last[0], parent_last[1]]
            for (p0, p1) in zip(path[::-1], path[::-1][1:]):
                state = table[p0[0], p0[1]]
                c = cost[p0[0], p0[1]]
                a = parent_action[p1[0], p1[1]]
                print(state)
                print(f"Cost: {c} | Action: {puzzle.action_to_string(a)}")
            print(solved_st)
            print(f"Cost: {solved_cost}")

            print("\n\n")
        else:
            print("No solution found\n\n")
    else:
        if solved:
            solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index][0]
            print(f"Cost: {solved_cost}")
            print("Solution found\n\n")
        else:
            print("No solution found\n\n")

    if vmap_size == 1:
        return

    states = puzzle.get_target_state()[jnp.newaxis, ...]
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)
    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)
    vmapped_astar = jax.jit(jax.vmap(astar_fn, in_axes=(None, 0, 0, None)))
    print("initializing vmapped jit")
    start = time.time()
    search_result, solved, solved_idx = vmapped_astar(search_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    # for benchmark, same initial states
    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(
        key=jax.random.split(jax.random.PRNGKey(start_state_seed), 1)
    )
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)

    print("Vmapped Q* search, multiple initial state solution")
    print("Start states")
    print(states)
    print("Target state")
    print(target)

    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)

    print("vmap astar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(astar_fn, in_axes=(None, 0, 0, None))"
        "(search_result_build(), states, filled, target)"
    )
    start = time.time()

    search_result, solved, solved_idx = vmapped_astar(search_result_build(), states, filled, target)
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    search_states = jnp.sum(search_result.hashtable.size)

    print(
        f"Time: {vmapped_search_time:6.2f} seconds (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})"
    )
    print(
        f"Search states: {human_format(search_states)} ({human_format(search_states / vmapped_search_time)} states/s)"
    )
    print("Solution found:", f"{jnp.mean(solved)*100:.2f}%")
    # this means astart_fn is completely vmapable and jitable


@main.command()
@puzzle_options
@search_options
@qfunction_options
@visualize_options
def qstar(
    puzzle,
    hard,
    puzzle_size,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seed,
    seed,
    vmap_size,
    debug,
    profile,
    neural_qfunction,
    visualize,
):
    if debug:
        # disable jit
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)

        # scale down the sizes for debugging
        max_node_size = 10000
        batch_size = 100
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle)

    puzzle_name = puzzle
    if hard:
        puzzle = puzzle_dict_hard[puzzle](puzzle_size)
    else:
        puzzle = puzzle_dict[puzzle](puzzle_size)

    if neural_qfunction:
        try:
            qfunction: QFunction = puzzle_q_dict_nn[puzzle_name](puzzle_size, puzzle, False)
        except KeyError:
            print("Neural qfunction not available for this puzzle")
            print(f"list of neural qfunction: {puzzle_q_dict_nn.keys()}")
            exit(1)
    else:
        qfunction: QFunction = puzzle_q_dict[puzzle_name](puzzle)

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)
    start_state_seed = int(start_state_seed)
    seed = int(seed)

    states = puzzle.get_target_state()[jnp.newaxis, ...]
    target = puzzle.get_target_state()

    search_result_build, qstar_fn = qstar_builder(
        puzzle, qfunction, batch_size, max_node_size, cost_weight=cost_weight
    )

    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    print("initializing jit")
    start = time.time()
    search_result, solved, solved_idx = qstar_fn(search_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds")
    print("JIT compiled\n\n")

    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(
        key=jax.random.split(jax.random.PRNGKey(start_state_seed), 1)
    )
    qvalues = qfunction.batched_q_value(states, target)[0]

    print("Start state")
    print(states[0])
    print("Target state")
    print(target)
    print("qvalues: ", end="")
    print(
        " | ".join(
            f"'{puzzle.action_to_string(i)}': {qvalues[i]:.1f}" for i in range(qvalues.shape[0])
        )
    )
    print()

    if profile:
        print("Profiling")
        jax.profiler.start_trace("tmp/tensorboard")
    states, filled = HashTable.make_batched(puzzle.State, states, batch_size)
    start = time.time()
    search_result, solved, solved_idx = qstar_fn(search_result_build(), states, filled, target)
    end = time.time()
    single_search_time = end - start
    print(f"Time: {single_search_time:6.2f} seconds")
    print(
        f"Search states: {human_format(search_result.hashtable.size)}"
        f"({human_format(search_result.hashtable.size / single_search_time)} states/s)\n\n"
    )
    if profile:
        jax.profiler.stop_trace()

    if visualize:
        if solved:
            print("Solution found\n\n")

            parents = search_result.parent
            parent_action = search_result.parent_action
            table = search_result.hashtable.table
            cost = search_result.cost

            solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index][0]
            solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index][0]

            path = [(solved_idx.index[0], solved_idx.table_index[0])]
            parent_last = parents[solved_idx.index, solved_idx.table_index][0]
            while True:
                if parent_last[0] == -1:
                    break
                path.append(parent_last)
                parent_last = parents[parent_last[0], parent_last[1]]
            for (p0, p1) in zip(path[::-1], path[::-1][1:]):
                state = table[p0[0], p0[1]]
                c = cost[p0[0], p0[1]]
                a = parent_action[p1[0], p1[1]]
                print(state)
                print(f"Cost: {c} | Action: '{puzzle.action_to_string(a)}'")
            print(solved_st)
            print(f"Cost: {solved_cost}")

            print("\n\n")
        else:
            print("No solution found\n\n")
    else:
        if solved:
            solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index][0]
            print(f"Cost: {solved_cost}")
            print("Solution found\n\n")
        else:
            print("No solution found\n\n")

    if vmap_size == 1:
        return

    states = puzzle.get_target_state()[jnp.newaxis, ...]
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)
    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)
    vmapped_qstar = jax.jit(jax.vmap(qstar_fn, in_axes=(None, 0, 0, None)))
    print("initializing vmapped jit")
    start = time.time()
    search_result, solved, solved_idx = vmapped_qstar(search_result_build(), states, filled, target)
    end = time.time()
    print(f"Time: {end - start:6.2f} seconds\n\n")

    # for benchmark, same initial states
    states = jax.vmap(puzzle.get_initial_state, in_axes=0)(
        key=jax.random.split(jax.random.PRNGKey(start_state_seed), 1)
    )
    states = jax.tree_util.tree_map(lambda x: jnp.tile(x, (vmap_size, 1)), states)

    print("Vmapped Q* search, multiple initial state solution")
    print("Start states")
    print(states)
    print("Target state")
    print(target)

    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)

    print("vmap qstar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(qstar_fn, in_axes=(None, 0, 0, None))"
        "(search_result_build(), states, filled, target)"
    )
    start = time.time()

    search_result, solved, solved_idx = vmapped_qstar(search_result_build(), states, filled, target)
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    search_states = jnp.sum(search_result.hashtable.size)

    print(
        f"Time: {vmapped_search_time:6.2f} seconds (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})"
    )
    print(
        f"Search states: {human_format(search_states)} ({human_format(search_states / vmapped_search_time)} states/s)"
    )
    print("Solution found:", f"{jnp.mean(solved)*100:.2f}%")


if __name__ == "__main__":
    main.add_command(astar)
    main.add_command(qstar)
    main()
