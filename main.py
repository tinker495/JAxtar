import time

import click
import jax
import jax.numpy as jnp

from JAxtar.astar import astar_builder
from JAxtar.hash import HashTable
from JAxtar.qstar import qstar_builder
from JAxtar.util import set_tree
from options import (
    heuristic_options,
    human_play_options,
    puzzle_options,
    qfunction_options,
    search_options,
    visualize_options,
)
from util import human_format, vmapping_search


@click.group()
def main():
    pass


@main.command()
@puzzle_options
@human_play_options
def human_play(puzzle, start_state_seed):

    has_target = puzzle.has_target

    init_state, target_state = puzzle.get_init_target_state_pair(
        jax.random.PRNGKey(start_state_seed)
    )
    _, costs = puzzle.get_neighbours(init_state)
    n_actions = costs.shape[0]
    print("Initial state")
    print(init_state)
    if has_target:
        print("Target state")
        print(target_state)
    print("Next states")

    print("Use number keys to move the point.")
    print("Use ESC to exit.")
    current_state = init_state
    sum_cost = 0
    while True:
        print(current_state)
        print(f"Costs: {sum_cost}")
        print(
            f"Actions: {'|'.join(f'{i+1}: {puzzle.action_to_string(i)}' for i in range(n_actions))}"
        )
        neighbors, costs = puzzle.get_neighbours(current_state)
        key = click.getchar()
        if key == "\x1b":  # ESC
            break
        try:
            action = int(key) - 1
            if costs[action] == jnp.inf:
                print("Invalid move!")
                continue
            current_state, cost = neighbors[action], costs[action]
            sum_cost += cost
        except ValueError:
            print("Invalid input!")
        except IndexError:
            print("Invalid action index!")
        if puzzle.is_solved(current_state, target_state):
            print(f"Solution found! Cost: {sum_cost}")
            break


@main.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar(
    puzzle,
    heuristic,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seeds,
    vmap_size,
    profile,
    show_compile_time,
    visualize,
):
    has_target = puzzle.has_target

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)

    search_result_build, astar_fn = astar_builder(
        puzzle,
        heuristic,
        batch_size,
        max_node_size,
        cost_weight=cost_weight,
        show_compile_time=show_compile_time,
    )

    total_search_times = []
    total_states = []
    total_solved = []
    for start_state_seed in start_state_seeds:
        states, target = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
        heuristic_values = heuristic.distance(states, target)

        print("Start state")
        print(states)
        if has_target:
            print("Target state")
            print(target)
        print(f"Heuristic: {heuristic_values:.2f}")

        if profile:
            print("Profiling")
            jax.profiler.start_trace("tmp/tensorboard")
        states, filled = HashTable.make_batched(puzzle.State, states[jnp.newaxis, ...], batch_size)
        inital_search_result = search_result_build()

        start = time.time()
        search_result, solved, solved_idx = astar_fn(inital_search_result, states, filled, target)
        end = time.time()
        single_search_time = end - start
        states_per_second = search_result.hashtable.size / single_search_time

        if not has_target:
            if solved:
                solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index]
                print("Solution state")
                print(solved_st)
                print()

        print(f"Search Time: {single_search_time:6.2f} seconds")
        print(
            f"Search states: {human_format(search_result.hashtable.size)}"
            f"({human_format(states_per_second)} states/s)\n\n"
        )

        total_search_times.append(single_search_time)
        total_states.append(search_result.hashtable.size)
        total_solved.append(solved)
        if profile:
            jax.profiler.stop_trace()

        if visualize:
            if solved:
                print("Solution found\n\n")

                parents = search_result.parent
                parent_action = search_result.parent_action
                table = search_result.hashtable.table
                cost = search_result.cost

                solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index]
                solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index]

                path = [solved_idx]
                parent_last = parents[solved_idx.index, solved_idx.table_index]
                while True:
                    if parent_last.index == -1:
                        break
                    path.append(parent_last)
                    parent_last = parents[parent_last.index, parent_last.table_index]
                for (p0, p1) in zip(path[::-1], path[::-1][1:]):
                    state = table[p0.index, p0.table_index]
                    c = cost[p0.index, p0.table_index]
                    a = parent_action[p1.index, p1.table_index]
                    print(state)
                    print(f"Cost: {c} | Action: {puzzle.action_to_string(a)}")
                print(solved_st)
                print(f"Cost: {solved_cost}")

                print("\n\n")
            else:
                print("No solution found\n\n")
        else:
            if solved:
                solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index]
                print(f"Cost: {solved_cost:.1f}")
                print("Solution found\n\n")
            else:
                print("No solution found\n\n")

    if len(start_state_seeds) > 1:
        total_search_times = jnp.array(total_search_times)
        total_states = jnp.array(total_states)
        total_solved = jnp.array(total_solved)
        print(f"Seed: {', '.join(str(x) for x in start_state_seeds)}")
        print(
            f"Search time: {', '.join(f'{x:.2f}' for x in total_search_times)} seconds "
            f"(total: {jnp.sum(total_search_times):.2f}, avg: {jnp.mean(total_search_times):.2f})"
        )
        print(
            f"Search states: {', '.join(human_format(x) for x in total_states)} "
            f"(total: {human_format(jnp.sum(total_states))}, avg: {human_format(jnp.mean(total_states))})"
        )
        print(
            f"Solution found: {', '.join('O' if x else 'X' for x in total_solved)} "
            f"(total: {jnp.sum(total_solved)}, avg: {jnp.mean(total_solved)*100:.2f}%)"
        )

    if vmap_size == 1:
        return

    vmapped_astar = vmapping_search(
        puzzle, search_result_build, astar_fn, vmap_size, batch_size, show_compile_time
    )

    # for benchmark, same initial states
    start_state_seed = start_state_seeds[0]
    states, targets = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), states[jnp.newaxis, ...]
    )
    targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), targets[jnp.newaxis, ...]
    )

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_state, new_target = puzzle.get_init_target_state_pair(
                jax.random.PRNGKey(start_state_seed)
            )
            states = set_tree(
                states,
                new_state,
                i + 1,
            )
            targets = set_tree(
                targets,
                new_target,
                i + 1,
            )

    print("Vmapped A* search, multiple initial state solution")
    print("Start states")
    print(states)
    print("Target state")
    print(targets)

    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)

    print("vmap astar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(astar_fn, in_axes=(None, 0, 0, None))"
        "(inital_search_result, states, filled, target)"
    )
    start = time.time()

    search_result, solved, solved_idx = vmapped_astar(inital_search_result, states, filled, targets)
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    search_states = jnp.sum(search_result.hashtable.size)
    vmapped_states_per_second = search_states / vmapped_search_time

    if len(start_state_seeds) > 1:
        sizes = search_result.hashtable.size
        print(
            f"Search Time: {vmapped_search_time:6.2f} seconds "
            f"(x{vmapped_search_time/jnp.sum(total_search_times)*vmap_size:.1f}/{vmap_size})"
        )
        print(
            f"Search states: {', '.join(human_format(x) for x in sizes)} "
            f"(total: {human_format(jnp.sum(sizes))}, avg: {human_format(jnp.mean(sizes))})"
            f" (x{vmapped_states_per_second/states_per_second:.1f} faster)"
        )
        print(
            f"Solution found: {', '.join('O' if x else 'X' for x in solved)} "
            f"(total: {jnp.sum(solved)}, avg: {jnp.mean(solved)*100:.2f}%)"
        )
    else:
        print(
            f"Search Time: {vmapped_search_time:6.2f} seconds "
            f"(x{vmapped_search_time/single_search_time:.1f}/{vmap_size})"
        )
        print(
            f"Search states: {human_format(search_states)}"
            f" ({human_format(vmapped_states_per_second)} states/s)"
            f" (x{vmapped_states_per_second/states_per_second:.1f} faster)"
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
    qfunction,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seeds,
    vmap_size,
    profile,
    show_compile_time,
    visualize,
):

    has_target = puzzle.has_target

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)

    search_result_build, qstar_fn = qstar_builder(
        puzzle,
        qfunction,
        batch_size,
        max_node_size,
        cost_weight=cost_weight,
        show_compile_time=show_compile_time,
    )

    total_search_times = []
    total_states = []
    total_solved = []
    for start_state_seed in start_state_seeds:
        states, target = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
        qvalues = qfunction.q_value(states, target)

        print("Start state")
        print(states)
        if has_target:
            print("Target state")
            print(target)
        print("qvalues: ", end="")
        print(
            " | ".join(
                f"'{puzzle.action_to_string(i)}': {float(qvalues[i]):.1f}"
                for i in range(qvalues.shape[0])
            )
        )

        if profile:
            print("Profiling")
            jax.profiler.start_trace("tmp/tensorboard")
        states, filled = HashTable.make_batched(puzzle.State, states[jnp.newaxis, ...], batch_size)
        inital_search_result = search_result_build()

        start = time.time()
        search_result, solved, solved_idx = qstar_fn(inital_search_result, states, filled, target)
        end = time.time()
        single_search_time = end - start
        states_per_second = search_result.hashtable.size / single_search_time

        if not has_target:
            if solved:
                solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index]
                print("Solution state")
                print(solved_st)
                print()

        print(f"Search Time: {single_search_time:6.2f} seconds")
        print(
            f"Search states: {human_format(search_result.hashtable.size)}"
            f"({human_format(states_per_second)} states/s)\n\n"
        )

        total_search_times.append(single_search_time)
        total_states.append(search_result.hashtable.size)
        total_solved.append(solved)
        if profile:
            jax.profiler.stop_trace()

        if visualize:
            if solved:
                print("Solution found\n\n")

                parents = search_result.parent
                parent_action = search_result.parent_action
                table = search_result.hashtable.table
                cost = search_result.cost

                solved_st = search_result.hashtable.table[solved_idx.index, solved_idx.table_index]
                solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index]

                path = [solved_idx]
                parent_last = parents[solved_idx.index, solved_idx.table_index]
                while True:
                    if parent_last.index == -1:
                        break
                    path.append(parent_last)
                    parent_last = parents[parent_last.index, parent_last.table_index]
                for (p0, p1) in zip(path[::-1], path[::-1][1:]):
                    state = table[p0.index, p0.table_index]
                    c = cost[p0.index, p0.table_index]
                    a = parent_action[p1.index, p1.table_index]
                    print(state)
                    print(f"Cost: {c} | Action: {puzzle.action_to_string(a)}")
                print(solved_st)
                print(f"Cost: {solved_cost}")

                print("\n\n")
            else:
                print("No solution found\n\n")
        else:
            if solved:
                solved_cost = search_result.cost[solved_idx.index, solved_idx.table_index]
                print(f"Cost: {solved_cost:.1f}")
                print("Solution found\n\n")
            else:
                print("No solution found\n\n")

    if len(start_state_seeds) > 1:
        total_search_times = jnp.array(total_search_times)
        total_states = jnp.array(total_states)
        total_solved = jnp.array(total_solved)
        print(f"Seed: {', '.join(str(x) for x in start_state_seeds)}")
        print(
            f"Search time: {', '.join(f'{x:.2f}' for x in total_search_times)} seconds "
            f"(total: {jnp.sum(total_search_times):.2f}, avg: {jnp.mean(total_search_times):.2f})"
        )
        print(
            f"Search states: {', '.join(human_format(x) for x in total_states)} "
            f"(total: {human_format(jnp.sum(total_states))}, avg: {human_format(jnp.mean(total_states))})"
        )
        print(
            f"Solution found: {', '.join('O' if x else 'X' for x in total_solved)} "
            f"(total: {jnp.sum(total_solved)}, avg: {jnp.mean(total_solved)*100:.2f}%)"
        )

    if vmap_size == 1:
        return

    vmapped_qstar = vmapping_search(
        puzzle, search_result_build, qstar_fn, vmap_size, batch_size, show_compile_time
    )

    # for benchmark, same initial states
    start_state_seed = start_state_seeds[0]
    states, targets = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), states[jnp.newaxis, ...]
    )
    targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (vmap_size,) + (1,) * len(x.shape[1:])), targets[jnp.newaxis, ...]
    )

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_state, new_target = puzzle.get_init_target_state_pair(
                jax.random.PRNGKey(start_state_seed)
            )
            states = set_tree(
                states,
                new_state,
                i + 1,
            )
            targets = set_tree(
                targets,
                new_target,
                i + 1,
            )

    print("Vmapped Q* search, multiple initial state solution")
    print("Start states")
    print(states)
    print("Target state")
    print(targets)

    states, filled = jax.vmap(
        lambda x: HashTable.make_batched(puzzle.State, x[jnp.newaxis, ...], batch_size), in_axes=0
    )(states)

    print("vmap qstar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(qstar_fn, in_axes=(None, 0, 0, None))"
        "(inital_search_result, states, filled, target)"
    )
    start = time.time()

    search_result, solved, solved_idx = vmapped_qstar(inital_search_result, states, filled, targets)
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    search_states = jnp.sum(search_result.hashtable.size)
    vmapped_states_per_second = search_states / vmapped_search_time

    if len(start_state_seeds) > 1:
        sizes = search_result.hashtable.size
        print(
            f"Search Time: {vmapped_search_time:6.2f} seconds "
            f"(x{vmapped_search_time/jnp.sum(total_search_times)*vmap_size:.1f}/{vmap_size})"
        )
        print(
            f"Search states: {', '.join(human_format(x) for x in sizes)} "
            f"(total: {human_format(jnp.sum(sizes))}, avg: {human_format(jnp.mean(sizes))})"
            f" (x{vmapped_states_per_second/states_per_second:.1f} faster)"
        )
        print(
            f"Solution found: {', '.join('O' if x else 'X' for x in solved)} "
            f"(total: {jnp.sum(solved)}, avg: {jnp.mean(solved)*100:.2f}%)"
        )
    else:
        print(
            f"Search Time: {vmapped_search_time:6.2f} seconds "
            f"(x{vmapped_search_time/single_search_time:.1f}/{vmap_size})"
        )
        print(
            f"Search states: {human_format(search_states)}"
            f" ({human_format(vmapped_states_per_second)} states/s)"
            f" (x{vmapped_states_per_second/states_per_second:.1f} faster)"
        )
        print("Solution found:", f"{jnp.mean(solved)*100:.2f}%")


if __name__ == "__main__":
    main.add_command(human_play)
    main.add_command(astar)
    main.add_command(qstar)
    main()
