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
    vmapping_get_state,
    vmapping_init_target,
    vmapping_search,
)
from helpers.config_printer import print_config
from helpers.rich_progress import tqdm
from helpers.visualization import (
    build_path_steps_from_actions,
    build_path_steps_from_nodes,
    build_result_table,
    build_seed_setup_panel,
    build_solution_path_panel,
    build_summary_table,
    build_vmapped_results_table_multi,
    build_vmapped_results_table_single,
    build_vmapped_setup_panel,
    save_solution_animation_and_frames,
)
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction

from .config_utils import enrich_config


def _prepare_dist_parameters(
    solve_config: Puzzle.SolveConfig,
    *,
    heuristic: Heuristic | None,
    qfunction: QFunction | None,
):
    """
    Normalize distance/Q-value function inputs.

    - Heuristics are defined as distance(heuristic_parameters, state)
    - Q-functions are defined as q_value(q_parameters, state)

    Most simple implementations use the raw solve_config as parameters, but neural
    implementations typically require (params, solve_config). Some older code uses
    `prepare_parameters` instead of the base-class `prepare_*_parameters` hook, so we
    support both here.
    """
    if heuristic is not None:
        if hasattr(heuristic, "prepare_parameters"):
            return heuristic.prepare_parameters(solve_config)
        return heuristic.prepare_heuristic_parameters(solve_config)

    if qfunction is not None:
        if hasattr(qfunction, "prepare_parameters"):
            return qfunction.prepare_parameters(solve_config)
        return qfunction.prepare_q_parameters(solve_config)

    return solve_config


def _to_python_float(value: jnp.ndarray | float | int | None) -> float | None:
    """Convert scalar-like JAX/NumPy values (including shape-(1,)) to Python float."""
    if value is None:
        return None
    return float(jnp.ravel(jnp.asarray(value))[0])


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
        dist_parameters = _prepare_dist_parameters(
            solve_config, heuristic=heuristic, qfunction=qfunction
        )
        dist_values = dist_fn(dist_parameters, state)

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
        is_bidirectional = (
            hasattr(search_result, "meeting")
            and hasattr(search_result, "forward")
            and hasattr(search_result, "backward")
        )

        if is_bidirectional:
            solved = search_result.meeting.found.block_until_ready()
            generated_size = search_result.total_generated
            solved_cost = (
                search_result.meeting.total_cost
                if bool(jax.device_get(search_result.meeting.found))
                else None
            )
        else:
            solved = search_result.solved.block_until_ready()
            generated_size = search_result.generated_size
            solved_cost = search_result.get_cost(search_result.solved_idx) if solved else None
        end = time.time()
        single_search_time = end - start
        states_per_second = generated_size / single_search_time

        if (not has_target) and solved and (not is_bidirectional):
            solved_st = search_result.get_state(search_result.solved_idx)
            console.print(
                Panel(
                    Text.from_ansi(str(solved_st)),
                    title="[bold green]Solution State[/bold green]",
                    expand=False,
                )
            )

        total_search_times.append(single_search_time)
        total_states.append(generated_size)
        total_solved.append(solved)
        states_per_sec_list.append(states_per_second)

        if search_options.profile:
            jax.profiler.stop_trace()

        solved_cost_value = _to_python_float(solved_cost)
        result_table = build_result_table(
            solved=solved,
            single_search_time=single_search_time,
            generated_size=generated_size,
            states_per_second=states_per_second,
            solved_cost=solved_cost_value,
            seed=seed,
        )
        console.print(result_table)

        if solved:
            total_costs.append(solved_cost_value if solved_cost_value is not None else 0.0)

            if visualize_options.visualize_terminal or visualize_options.visualize_imgs:
                if is_bidirectional:
                    from JAxtar.bi_stars.bi_search_base import (
                        reconstruct_bidirectional_path,
                    )

                    bi_pairs = reconstruct_bidirectional_path(search_result, puzzle)
                    actions = [a for a, _ in bi_pairs[1:]]
                    states_trace = [s for _, s in bi_pairs]
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=state,
                        actions=actions,
                        heuristic=heuristic,
                        q_fn=qfunction,
                        states=states_trace,
                    )
                elif hasattr(search_result, "solution_trace"):
                    (
                        states_trace,
                        costs_trace,
                        dists_trace,
                        actions_trace,
                    ) = search_result.solution_trace()
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=state,
                        actions=actions_trace,
                        heuristic=heuristic,
                        q_fn=qfunction,
                        states=states_trace,
                        costs=costs_trace,
                        dists=dists_trace,
                    )
                elif hasattr(search_result, "solution_actions"):
                    actions = search_result.solution_actions()
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=state,
                        actions=actions,
                        heuristic=heuristic,
                        q_fn=qfunction,
                    )
                else:
                    path = search_result.get_solved_path()
                    path_steps = build_path_steps_from_nodes(
                        search_result=search_result,
                        path=path,
                        puzzle=puzzle,
                        solve_config=solve_config,
                    )

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
            avg_time_seconds = float(jnp.mean(jnp.array(total_search_times)))
            iterable.set_postfix(
                {
                    "solved": f"{sum(total_solved)}/{len(total_solved)}",
                    "avg_time": "{:.2f}s".format(avg_time_seconds),
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
        "search_options": search_options.dict(),
        component_name: component.__class__.__name__,
        f"{component_name}_metadata": getattr(component, "metadata", {}),
        "visualize_options": visualize_options.dict(),
    }
    print_config(config_title, enrich_config(config))

    warmup_seed = seeds[0] if seeds else 0
    warmup_config, warmup_state = puzzle.get_inits(jax.random.PRNGKey(warmup_seed))

    search_fn = builder_fn(
        puzzle,
        component,
        search_options.batch_size,
        search_options.get_max_node_size(),
        pop_ratio=search_options.pop_ratio,
        cost_weight=search_options.cost_weight,
        show_compile_time=search_options.show_compile_time,
        warmup_inputs=(warmup_config, warmup_state),
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
            puzzle, search_fn, search_options.vmap_size, search_options.show_compile_time
        ),
        puzzle=puzzle,
        seeds=seeds,
        search_options=search_options,
        total_search_times=total_search_times,
        states_per_second=states_per_second,
        single_search_time=single_search_time,
    )
