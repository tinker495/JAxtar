import time

import click
import jax
import jax.numpy as jnp
from puxle import Puzzle
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.pydantic_models import SearchOptions, VisualizeOptions
from helpers import (
    heuristic_dist_format,
    human_format,
    qfunction_dist_format,
    vmapping_get_state,
    vmapping_init_target,
    vmapping_search,
)
from helpers.config_printer import print_config
from helpers.rich_progress import tqdm
from helpers.visualization import (
    build_human_play_layout,
    build_human_play_setup_panel,
    build_seed_setup_panel,
    build_solution_path_panel,
    build_vmapped_setup_panel,
    save_solution_animation_and_frames,
)
from heuristic.heuristic_base import Heuristic
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFunction

from .options import (
    heuristic_options,
    human_play_options,
    puzzle_options,
    qfunction_options,
    search_options,
    visualize_options,
)


@click.command()
@puzzle_options
@human_play_options
def human_play(puzzle: Puzzle, seed: int, **kwargs):
    console = Console()
    has_target = puzzle.has_target

    solve_config, init_state = puzzle.get_inits(jax.random.PRNGKey(seed))
    _, costs = puzzle.get_neighbours(solve_config, init_state)
    n_actions = costs.shape[0]

    action_strs = [puzzle.action_to_string(i) for i in range(n_actions)]
    arrow_characters = ["←", "→", "↑", "↓"]

    arrow_flag = any(arrow in s for s in action_strs for arrow in arrow_characters)

    console.print(
        build_human_play_setup_panel(
            has_target=has_target, solve_config=solve_config, init_state=init_state
        )
    )

    console.print("Use number keys, [bold yellow]WASD[/bold yellow] or arrow keys to move.")
    console.print("Use [bold red]ESC[/bold red] to exit.")

    current_state = init_state
    sum_cost = 0

    def get_action(n_actions: int, arrow_flag: bool) -> int | None:
        """Get user input and return the corresponding action."""
        _, costs = puzzle.get_neighbours(solve_config, current_state)
        while True:
            key = click.getchar()
            if key == "\x1b":  # ESC
                return None
            try:
                action = -1
                if arrow_flag:
                    action_map = {}
                    for i, s in enumerate(action_strs):
                        if "↑" in s:
                            action_map["w"] = i
                        if "↓" in s:
                            action_map["s"] = i
                        if "←" in s:
                            action_map["a"] = i
                        if "→" in s:
                            action_map["d"] = i

                    if key.lower() in action_map:
                        action = action_map[key.lower()]
                    else:
                        raise ValueError("Invalid WASD key")
                else:
                    action = int(key) - 1

                if not (0 <= action < n_actions and costs[action] != jnp.inf):
                    raise ValueError("Invalid or impossible action")

                return action
            except (ValueError, IndexError):
                continue

    def generate_layout() -> Group:
        return build_human_play_layout(
            current_state=current_state,
            solve_config=solve_config,
            sum_cost=sum_cost,
            action_strs=action_strs,
            arrow_flag=arrow_flag,
        )

    with Live(generate_layout(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            # Get user input
            action = get_action(n_actions, arrow_flag)
            if action is None:
                break

            # Update the state based on the chosen action
            neighbors, costs = puzzle.get_neighbours(solve_config, current_state)
            current_state, cost = neighbors[action], costs[action]
            sum_cost += cost
            live.update(generate_layout(), refresh=True)

        if puzzle.is_solved(solve_config, current_state):
            console.print(
                Panel(
                    current_state.str(solve_config=solve_config),
                    title="[bold green]Solution Found![/bold green]",
                    expand=False,
                )
            )
            console.print(f"Total Cost: [bold green]{sum_cost}[/bold green]")


def search_samples(
    search_fn,
    puzzle: Puzzle,
    puzzle_name: str,
    dist_fn,
    dist_fn_format,
    seeds: list[int],
    search_options: SearchOptions,
    visualize_options: VisualizeOptions,
):
    console = Console()
    has_target = puzzle.has_target

    total_search_times = []
    total_states = []
    total_solved = []
    total_costs = []
    states_per_sec_list = []

    iterable = seeds
    if len(seeds) > 1:
        iterable = tqdm(seeds, desc=f"Running {puzzle_name} search", leave=True, unit="seed")

    for seed in iterable:
        solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))
        dist_values = dist_fn(solve_config, state)

        if len(seeds) == 1:
            console.print(
                build_seed_setup_panel(
                    puzzle=puzzle,
                    has_target=has_target,
                    solve_config=solve_config,
                    state=state,
                    dist_text=dist_fn_format(puzzle, dist_values),
                    seed=seed,
                )
            )

        if search_options.profile:
            console.print("[yellow]Profiling enabled, starting trace...[/yellow]")
            jax.profiler.start_trace("tmp/tensorboard")

        start = time.time()
        search_result = search_fn(solve_config, state)
        solved = search_result.solved.block_until_ready()
        end = time.time()
        single_search_time = end - start
        states_per_second = search_result.generated_size / single_search_time

        if not has_target and solved:
            solved_st = search_result.get_state(search_result.solved_idx)
            console.print(
                Panel(
                    Text.from_ansi(str(solved_st)),
                    title="[bold green]Solution State[/bold green]",
                    expand=False,
                )
            )

        total_search_times.append(single_search_time)
        total_states.append(search_result.generated_size)
        total_solved.append(solved)
        states_per_sec_list.append(states_per_second)

        if search_options.profile:
            jax.profiler.stop_trace()

        result_table = Table(
            title=f"[bold]Search Result for Seed {seed}[/bold]",
        )
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", justify="right")
        if solved:
            solved_cost = search_result.get_cost(search_result.solved_idx)
            result_table.add_row("Status", "[bold green]Solution Found[/bold green]")
            result_table.add_row("Cost", f"{solved_cost:.1f}")
        else:
            result_table.add_row("Status", "[bold red]No Solution Found[/bold red]")

        result_table.add_row("Search Time", f"{single_search_time:.2f} s")
        result_table.add_row(
            "Search States",
            f"{human_format(search_result.generated_size)}",
        )
        result_table.add_row("States/s", f"{human_format(states_per_second)}")
        console.print(result_table)

        if solved:
            solved_idx = search_result.solved_idx
            solved_cost = search_result.get_cost(solved_idx)
            total_costs.append(solved_cost)

            if visualize_options.visualize_terminal or visualize_options.visualize_imgs:
                path = search_result.get_solved_path()

                if visualize_options.visualize_terminal:
                    console.print(
                        build_solution_path_panel(
                            console=console,
                            search_result=search_result,
                            path=path,
                            puzzle=puzzle,
                            solve_config=solve_config,
                            cost_weight=search_options.cost_weight,
                        )
                    )

                if visualize_options.visualize_imgs:
                    gif_path = save_solution_animation_and_frames(
                        search_result=search_result,
                        path=path,
                        puzzle_name=puzzle_name,
                        solve_config=solve_config,
                        max_animation_time=visualize_options.max_animation_time,
                    )
                    console.print(f"Saved solution animation to [cyan]{gif_path}[/cyan]")
        else:
            total_costs.append(jnp.inf)

        if len(seeds) > 1:
            iterable.set_postfix(
                {
                    "solved": f"{sum(total_solved)}/{len(total_solved)}",
                    "avg_time": f"{jnp.mean(jnp.array(total_search_times)):.2f}s",
                    "avg_states": human_format(jnp.mean(jnp.array(total_states))),
                }
            )

    if len(seeds) > 1:
        summary_table = Table(title=f"[bold]Summary for {puzzle_name} ({len(seeds)} seeds)[/bold]")
        summary_table.add_column("Seed", justify="right", style="cyan")
        summary_table.add_column("Solved", justify="center")
        summary_table.add_column("Search Time (s)", justify="right")
        summary_table.add_column("Num States", justify="right")
        summary_table.add_column("States/s", justify="right")
        summary_table.add_column("Cost", justify="right")

        for i, seed in enumerate(seeds):
            summary_table.add_row(
                str(seed),
                "[green]Yes[/green]" if total_solved[i] else "[red]No[/red]",
                f"{total_search_times[i]:.2f}",
                human_format(total_states[i]),
                human_format(states_per_sec_list[i]),
                f"{total_costs[i]:.1f}" if total_solved[i] else "N/A",
            )

        summary_table.add_section()
        summary_table.add_row(
            "[bold]Average[/bold]",
            f"{jnp.mean(jnp.array(total_solved))*100:.1f}%",
            f"{jnp.mean(jnp.array(total_search_times)):.2f}",
            human_format(jnp.mean(jnp.array(total_states))),
            human_format(jnp.mean(jnp.array(states_per_sec_list))),
            f"{jnp.mean(jnp.array([c for c in total_costs if c != jnp.inf])):.1f}",
        )
        console.print(summary_table)

    return total_search_times, states_per_second, single_search_time


def vmapped_search_samples(
    vmapped_search,
    puzzle: Puzzle,
    seeds: list[int],
    search_options: SearchOptions,
    total_search_times: jnp.ndarray,
    states_per_second: float,
    single_search_time: float,
):
    console = Console()
    has_target = puzzle.has_target
    vmap_size = search_options.vmap_size

    solve_configs, states = vmapping_init_target(puzzle, vmap_size, seeds)

    console.print(
        build_vmapped_setup_panel(has_target=has_target, solve_configs=solve_configs, states=states)
    )

    start = time.time()
    search_result = vmapped_search(solve_configs, states)
    solved = search_result.solved.block_until_ready()
    end = time.time()
    vmapped_search_time = end - start

    if not has_target and solved.any():
        solved_st = vmapping_get_state(search_result, search_result.solved_idx)
        grid = Table.grid(expand=False)
        grid.add_column()
        grid.add_row(Align.center("[bold green]Solution State[/bold green]"))
        grid.add_row(Text.from_ansi(str(solved_st)))
        console.print(Panel(grid, title="[bold green]Vmapped Solution[/bold green]", expand=False))

    search_states = jnp.sum(search_result.generated_size)
    vmapped_states_per_second = search_states / vmapped_search_time

    if len(seeds) > 1:
        sizes = search_result.generated_size
        table = Table(title="[bold]Vmapped Search Results[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row(
            "Search Time",
            f"{vmapped_search_time:6.2f}s "
            f"(x{vmapped_search_time/jnp.sum(total_search_times)*vmap_size:.1f}/{vmap_size})",
        )
        table.add_row(
            "Total Search States",
            f"{human_format(jnp.sum(sizes))} (avg: {human_format(jnp.mean(sizes))})",
        )
        table.add_row(
            "States per Second",
            f"{human_format(vmapped_states_per_second)} (x{vmapped_states_per_second/states_per_second:.1f} faster)",
        )
        table.add_row(
            "Solutions Found",
            f"{jnp.sum(solved)}/{len(solved)} ({jnp.mean(solved)*100:.2f}%)",
        )
        console.print(table)
    else:
        table = Table(title="[bold]Vmapped Search Result[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row(
            "Search Time",
            f"{vmapped_search_time:6.2f}s (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})",
        )
        table.add_row(
            "Search States",
            f"{human_format(search_states)} ({human_format(vmapped_states_per_second)} states/s)",
        )
        table.add_row("Speedup", f"x{vmapped_states_per_second/states_per_second:.1f}")
        table.add_row("Solutions Found", f"{jnp.mean(solved)*100:.2f}%")
        console.print(table)


@click.command()
@puzzle_options
@search_options
@heuristic_options
@visualize_options
def astar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    heuristic: Heuristic,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    config = {
        "puzzle_name": puzzle_name,
        "search_options": search_options.dict(),
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": getattr(heuristic, "metadata", {}),
        "visualize_options": visualize_options.dict(),
    }
    print_config("A* Search Configuration", config)
    astar_fn = astar_builder(
        puzzle,
        heuristic,
        search_options.batch_size,
        search_options.get_max_node_size(),
        pop_ratio=search_options.pop_ratio,
        cost_weight=search_options.cost_weight,
        show_compile_time=search_options.show_compile_time,
    )
    dist_fn = heuristic.distance
    total_search_times, states_per_second, single_search_time = search_samples(
        search_fn=astar_fn,
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        dist_fn=dist_fn,
        dist_fn_format=heuristic_dist_format,
        seeds=seeds,
        search_options=search_options,
        visualize_options=visualize_options,
    )

    if search_options.vmap_size == 1:
        return

    vmapped_search_samples(
        vmapped_search=vmapping_search(
            puzzle, astar_fn, search_options.vmap_size, search_options.show_compile_time
        ),
        puzzle=puzzle,
        seeds=seeds,
        search_options=search_options,
        total_search_times=total_search_times,
        states_per_second=states_per_second,
        single_search_time=single_search_time,
    )


@click.command()
@puzzle_options
@search_options
@qfunction_options
@visualize_options
def qstar(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    qfunction: QFunction,
    visualize_options: VisualizeOptions,
    **kwargs,
):
    config = {
        "puzzle_name": puzzle_name,
        "search_options": search_options.dict(),
        "qfunction": qfunction.__class__.__name__,
        "qfunction_metadata": getattr(qfunction, "metadata", {}),
        "visualize_options": visualize_options.dict(),
    }
    print_config("Q* Search Configuration", config)
    qstar_fn = qstar_builder(
        puzzle,
        qfunction,
        search_options.batch_size,
        search_options.get_max_node_size(),
        pop_ratio=search_options.pop_ratio,
        cost_weight=search_options.cost_weight,
        show_compile_time=search_options.show_compile_time,
    )
    dist_fn = qfunction.q_value
    total_search_times, states_per_second, single_search_time = search_samples(
        search_fn=qstar_fn,
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        dist_fn=dist_fn,
        dist_fn_format=qfunction_dist_format,
        seeds=seeds,
        search_options=search_options,
        visualize_options=visualize_options,
    )

    if search_options.vmap_size == 1:
        return

    vmapped_search_samples(
        vmapped_search=vmapping_search(
            puzzle, qstar_fn, search_options.vmap_size, search_options.show_compile_time
        ),
        puzzle=puzzle,
        seeds=seeds,
        search_options=search_options,
        total_search_times=total_search_times,
        states_per_second=states_per_second,
        single_search_time=single_search_time,
    )
