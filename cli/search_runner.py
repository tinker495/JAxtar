import json
import time

import jax
import jax.numpy as jnp
from puxle import Puzzle
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.pydantic_models import SearchOptions, VisualizeOptions
from helpers import (
    human_format,
    vmapping_init_target,
    vmapping_search,
)
from helpers.config_printer import print_config
from helpers.rich_progress import tqdm
from helpers.visualization import (
    build_result_table,
    build_seed_setup_panel,
    build_solution_path_panel,
    build_summary_table,
    build_vmapped_results_table_multi,
    build_vmapped_results_table_single,
    build_vmapped_setup_panel,
    save_solution_animation_and_frames,
)
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.search_base import SearchResult
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction

from .config_utils import enrich_config
from .search_outcome import normalise_search_result, with_solution_path


def search_samples(
    search_fn,
    puzzle: Puzzle,
    puzzle_name: str,
    dist_fn,
    dist_fn_format,
    seeds: list[int],
    search_options: SearchOptions,
    visualize_options: VisualizeOptions,
    heuristic: Heuristic | None = None,
    qfunction: QFunction | None = None,
):
    console = Console()
    has_goal_data = puzzle.has_goal_data

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
        if heuristic is not None:
            dist_parameters = heuristic.prepare_heuristic_parameters(solve_config)
        elif qfunction is not None:
            dist_parameters = qfunction.prepare_q_parameters(solve_config)
        else:
            dist_parameters = solve_config
        dist_values = dist_fn(dist_parameters, state)

        if len(seeds) == 1:
            console.print(
                build_seed_setup_panel(
                    puzzle=puzzle,
                    has_goal_data=has_goal_data,
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
        outcome = normalise_search_result(
            search_result,
            emit_workload_signature=getattr(search_options, "emit_workload_signature", False),
        )
        end = time.time()
        single_search_time = end - start
        states_per_second = outcome.generated_size / single_search_time

        def ensure_path_outcome():
            nonlocal outcome
            if outcome.solved and not outcome.path_steps:
                outcome = with_solution_path(
                    outcome,
                    search_result,
                    puzzle=puzzle,
                    solve_config=solve_config,
                    initial_state=state,
                    heuristic=heuristic,
                    qfunction=qfunction,
                )
            return outcome

        if (not has_goal_data) and outcome.solved:
            path_outcome = ensure_path_outcome()
            if path_outcome.solution_state is not None:
                console.print(
                    Panel(
                        Text.from_ansi(str(path_outcome.solution_state)),
                        title="[bold green]Solution State[/bold green]",
                        expand=False,
                    )
                )

        total_search_times.append(single_search_time)
        total_states.append(outcome.generated_size)
        total_solved.append(outcome.solved)
        states_per_sec_list.append(states_per_second)

        if search_options.profile:
            jax.profiler.stop_trace()

        result_table = build_result_table(
            solved=outcome.solved,
            single_search_time=single_search_time,
            generated_size=outcome.generated_size,
            states_per_second=states_per_second,
            solved_cost=outcome.solved_cost,
            seed=seed,
        )
        console.print(result_table)
        if outcome.workload_signature:
            console.print(
                Panel(
                    Text(json.dumps(outcome.workload_signature, indent=2)),
                    title="[bold cyan]Xtructure Workload Signature[/bold cyan]",
                    expand=False,
                )
            )

        if outcome.solved:
            total_costs.append(outcome.solved_cost if outcome.solved_cost is not None else 0.0)

            if visualize_options.visualize_terminal or visualize_options.visualize_imgs:
                path_steps = list(ensure_path_outcome().path_steps)

                if visualize_options.visualize_terminal:
                    console.print(
                        build_solution_path_panel(
                            console=console,
                            path_steps=path_steps,
                            puzzle=puzzle,
                            solve_config=solve_config,
                            cost_weight=search_options.cost_weight,
                        )
                    )

                if visualize_options.visualize_imgs:
                    gif_path = save_solution_animation_and_frames(
                        path_steps=path_steps,
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
        summary_table = build_summary_table(
            puzzle_name=puzzle_name,
            seeds=seeds,
            total_solved=total_solved,
            total_search_times=total_search_times,
            total_states=total_states,
            states_per_sec_list=states_per_sec_list,
            total_costs=total_costs,
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
    has_goal_data = puzzle.has_goal_data
    vmap_size = search_options.vmap_size

    solve_configs, states = vmapping_init_target(puzzle, vmap_size, seeds)

    console.print(
        build_vmapped_setup_panel(
            has_goal_data=has_goal_data,
            solve_configs=solve_configs,
            states=states,
        )
    )

    start = time.time()
    search_result = vmapped_search(solve_configs, states)
    solved = search_result.solved.block_until_ready()
    end = time.time()
    vmapped_search_time = end - start

    if not has_goal_data and solved.any():
        solved_st = jax.vmap(SearchResult.get_state, in_axes=(0, 0))(
            search_result, search_result.solved_idx
        )
        grid = Table.grid(expand=False)
        grid.add_column()
        grid.add_row(Align.center("[bold green]Solution State[/bold green]"))
        grid.add_row(Text.from_ansi(str(solved_st)))
        console.print(Panel(grid, title="[bold green]Vmapped Solution[/bold green]", expand=False))

    search_states = jnp.sum(search_result.generated_size)
    vmapped_states_per_second = search_states / vmapped_search_time

    if len(seeds) > 1:
        sizes = search_result.generated_size
        table = build_vmapped_results_table_multi(
            vmapped_search_time=vmapped_search_time,
            total_search_times=jnp.array(total_search_times),
            vmap_size=vmap_size,
            sizes=sizes,
            vmapped_states_per_second=vmapped_states_per_second,
            states_per_second=states_per_second,
            solved=solved,
        )
        console.print(table)
    else:
        table = build_vmapped_results_table_single(
            vmapped_search_time=vmapped_search_time,
            single_search_time=single_search_time,
            vmap_size=vmap_size,
            search_states=search_states,
            vmapped_states_per_second=vmapped_states_per_second,
            states_per_second=states_per_second,
            solved_mean=float(jnp.mean(solved)),
        )
        console.print(table)


def run_search_command(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: list[int],
    search_options: SearchOptions,
    visualize_options: VisualizeOptions,
    builder_fn: callable,
    component_name: str,
    component,
    dist_fn,
    dist_fn_format,
    config_title: str,
):
    config = {
        "puzzle_name": puzzle_name,
        "search_options": search_options.model_dump(),
        component_name: component.__class__.__name__,
        f"{component_name}_metadata": getattr(component, "metadata", {}),
        "visualize_options": visualize_options.model_dump(),
    }
    print_config(config_title, enrich_config(config))

    warmup_seed = seeds[0] if seeds else 0
    warmup_config, warmup_state = puzzle.get_inits(jax.random.PRNGKey(warmup_seed))

    spec = SearchBuildSpec(
        pop_ratio=search_options.pop_ratio,
        cost_weight=search_options.cost_weight,
        show_compile_time=search_options.show_compile_time,
        warmup_inputs=(warmup_config, warmup_state),
        emit_workload_signature=getattr(search_options, "emit_workload_signature", False),
    )
    search_fn = builder_fn(
        puzzle,
        component,
        search_options.batch_size,
        search_options.get_max_node_size(),
        spec,
    )

    total_search_times, states_per_second, single_search_time = search_samples(
        search_fn=search_fn,
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        dist_fn=dist_fn,
        dist_fn_format=dist_fn_format,
        seeds=seeds,
        search_options=search_options,
        visualize_options=visualize_options,
        **{component_name: component},
    )

    if search_options.vmap_size == 1:
        return

    vmapped_search_samples(
        vmapped_search=vmapping_search(
            puzzle,
            search_fn,
            search_options.vmap_size,
            search_options.show_compile_time,
        ),
        puzzle=puzzle,
        seeds=seeds,
        search_options=search_options,
        total_search_times=total_search_times,
        states_per_second=states_per_second,
        single_search_time=single_search_time,
    )
