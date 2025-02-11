import time

import click
import jax
import jax.numpy as jnp

from JAxtar.astar import astar_builder
from JAxtar.qstar import qstar_builder
from options import (
    heuristic_options,
    human_play_options,
    puzzle_options,
    qfunction_options,
    search_options,
    visualize_options,
)
from util import (
    human_format,
    vmapping_get_state,
    vmapping_init_target,
    vmapping_search,
    window,
)


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

    action_strs = [puzzle.action_to_string(i) for i in range(n_actions)]
    arrow_characters = ["←", "→", "↑", "↓"]

    # New WASD mapping for arrow controls
    wasd_mapping = {
        "w": "↑",
        "a": "←",
        "s": "↓",
        "d": "→",
    }

    arrow_flag = any(arrow in s for s in action_strs for arrow in arrow_characters)

    print("Initial state")
    print(init_state)
    if has_target:
        print("Target state")
        print(target_state)
    print("Next states")
    print("Use number keys, [WASD] or arrow keys to move the point.")
    print("Use ESC to exit.")

    current_state = init_state
    sum_cost = 0
    while True:
        print(current_state)
        print(f"Costs: {sum_cost}")
        if arrow_flag:
            print(
                f"Actions: {'|'.join(f'{k.upper()}: {v}' for k, v in list(wasd_mapping.items())[:n_actions])}"
            )
        else:
            print(f"Actions: {'|'.join(f'{i+1}: {action_strs[i]}' for i in range(n_actions))}")
        neighbors, costs = puzzle.get_neighbours(current_state)
        key = click.getchar()
        if key == "\x1b":  # ESC
            break
        try:
            print("Key pressed:", key)
            if arrow_flag:
                if key in arrow_characters:
                    action = arrow_characters.index(key)
                elif key.lower() in wasd_mapping:
                    action = arrow_characters.index(wasd_mapping[key.lower()])
                else:
                    action = int(key) - 1
            else:
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
            print(current_state)
            print(f"Solution found! Cost: {sum_cost}")
            break


@main.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar(
    puzzle,
    puzzle_name,
    heuristic,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seeds,
    vmap_size,
    profile,
    show_compile_time,
    visualize_terminal,
    visualize_imgs,
):
    has_target = puzzle.has_target

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)

    astar_fn = astar_builder(
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
        state, target = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
        heuristic_values = heuristic.distance(state, target)

        print("Start state")
        print(state)
        if has_target:
            print("Target state")
            print(target)
        print(f"Heuristic: {heuristic_values:.2f}")

        if profile:
            print("Profiling")
            jax.profiler.start_trace("tmp/tensorboard")

        start = time.time()
        search_result = astar_fn(state, target)
        solved = search_result.solved.block_until_ready()
        end = time.time()
        single_search_time = end - start
        states_per_second = search_result.hashtable.size / single_search_time

        if not has_target:
            if solved:
                solved_st = search_result.get_state(search_result.solved_idx)
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

        if solved:
            solved_idx = search_result.solved_idx
            solved_cost = search_result.get_cost(solved_idx)

            print(f"Cost: {solved_cost:.1f}")
            print("Solution found\n\n")
            if visualize_terminal or visualize_imgs:
                path = search_result.get_solved_path()

                if visualize_terminal:
                    for p0, p1 in window(path):
                        print(search_result.get_state(p0))
                        print(f"Cost: {search_result.get_cost(p0)}")
                        print(
                            f"Action: {puzzle.action_to_string(search_result.get_parent_action(p1))}"
                        )

                    print(search_result.get_state(path[-1]))
                    print(f"Cost: {search_result.get_cost(path[-1])}")
                    print("\n\n")
                elif visualize_imgs:
                    import os
                    from datetime import datetime

                    import cv2
                    import imageio

                    imgs = []
                    logging_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    logging_name = f"{puzzle_name}_{logging_time}"
                    os.makedirs(f"tmp/{logging_name}", exist_ok=True)
                    path_states = [search_result.get_state(p) for p in path]
                    for idx, p in enumerate(path):
                        img = search_result.get_state(p).img(
                            idx=idx, path=path_states, target=target
                        )
                        imgs.append(img)
                        cv2.imwrite(
                            f"tmp/{logging_name}/img_{idx}.png",
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        )
                    gif_path = f"tmp/{logging_name}/animation.gif"
                    imageio.mimsave(gif_path, imgs, fps=4)
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

    vmapped_astar = vmapping_search(puzzle, astar_fn, vmap_size, show_compile_time)

    # for benchmark, same initial states
    states, targets = vmapping_init_target(puzzle, vmap_size, start_state_seeds)

    print("Vmapped A* search, multiple initial state solution")
    print("Start states")
    print(states)
    if has_target:
        print("Target state")
        print(targets)

    print("vmap astar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(astar_fn, in_axes=(None, 0, 0, None))"
        "(inital_search_result, states, filled, target)"
    )
    start = time.time()
    search_result = vmapped_astar(states, targets)
    solved = search_result.solved.block_until_ready()
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    if not has_target:
        if solved.any():
            solved_st = vmapping_get_state(search_result, search_result.solved_idx)
            print("Solution state")
            print(solved_st)
            print()

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
    puzzle_name,
    qfunction,
    max_node_size,
    batch_size,
    cost_weight,
    start_state_seeds,
    vmap_size,
    profile,
    show_compile_time,
    visualize_terminal,
    visualize_imgs,
):

    has_target = puzzle.has_target

    max_node_size = int(max_node_size)
    batch_size = int(batch_size)

    qstar_fn = qstar_builder(
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
        state, target = puzzle.get_init_target_state_pair(jax.random.PRNGKey(start_state_seed))
        qvalues = qfunction.q_value(state, target)

        print("Start state")
        print(state)
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

        start = time.time()
        search_result = qstar_fn(state, target)
        solved = search_result.solved.block_until_ready()
        end = time.time()
        single_search_time = end - start
        states_per_second = search_result.hashtable.size / single_search_time

        if not has_target:
            if solved:
                solved_st = search_result.get_state(search_result.solved_idx)
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

        if solved:
            solved_idx = search_result.solved_idx
            solved_cost = search_result.get_cost(solved_idx)

            print(f"Cost: {solved_cost:.1f}")
            print("Solution found\n\n")
            if visualize_terminal or visualize_imgs:
                path = search_result.get_solved_path()

                if visualize_terminal:
                    for p0, p1 in window(path):
                        print(search_result.get_state(p0))
                        print(f"Cost: {search_result.get_cost(p0)}")
                        print(
                            f"Action: {puzzle.action_to_string(search_result.get_parent_action(p1))}"
                        )

                    print(search_result.get_state(path[-1]))
                    print(f"Cost: {search_result.get_cost(path[-1])}")
                    print("\n\n")
                elif visualize_imgs:
                    import os
                    from datetime import datetime

                    import cv2
                    import imageio

                    imgs = []
                    logging_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    logging_name = f"{puzzle_name}_{logging_time}"
                    os.makedirs(f"tmp/{logging_name}", exist_ok=True)
                    path_states = [search_result.get_state(p) for p in path]
                    for idx, p in enumerate(path):
                        img = search_result.get_state(p).img(
                            idx=idx, path=path_states, target=target
                        )
                        imgs.append(img)
                        cv2.imwrite(
                            f"tmp/{logging_name}/img_{idx}.png",
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        )
                    gif_path = f"tmp/{logging_name}/animation.gif"
                    imageio.mimsave(gif_path, imgs, fps=4)
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

    vmapped_qstar = vmapping_search(puzzle, qstar_fn, vmap_size, show_compile_time)

    # for benchmark, same initial states
    states, targets = vmapping_init_target(puzzle, vmap_size, start_state_seeds)

    print("Vmapped Q* search, multiple initial state solution")
    print("Start states")
    print(states)
    if has_target:
        print("Target state")
        print(targets)

    print("vmap qstar")
    print(
        "# search_result, solved, solved_idx ="
        "jax.vmap(qstar_fn, in_axes=(None, 0, 0, None))"
        "(inital_search_result, states, filled, target)"
    )
    start = time.time()
    search_result = vmapped_qstar(states, targets)
    solved = search_result.solved.block_until_ready()
    end = time.time()
    vmapped_search_time = end - start  # subtract jit time from the vmapped search time

    if not has_target:
        if solved.any():
            solved_st = vmapping_get_state(search_result, search_result.solved_idx)
            print("Solution state")
            print(solved_st)
            print()

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
