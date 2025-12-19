from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import jax.numpy as jnp
import numpy as np
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from helpers.formatting import human_format
from JAxtar.annotate import ACTION_DTYPE

ACTION_PAD_INT = int(np.iinfo(np.dtype(ACTION_DTYPE)).max)


def _require_cv2():  # pragma: no cover
    global cv2, _CV2_IMPORT_ERROR
    if cv2 is not None:
        return cv2
    if _CV2_IMPORT_ERROR is not None:
        raise ImportError(
            "OpenCV (cv2) is required for saving solution animations/frames. "
            "Install a NumPy-compatible build (e.g., downgrade to `numpy<2` or reinstall `opencv-python`)."
        ) from _CV2_IMPORT_ERROR
    try:
        import cv2 as cv2_mod  # type: ignore
    except Exception as exc:
        _CV2_IMPORT_ERROR = exc
        raise ImportError(
            "OpenCV (cv2) is required for saving solution animations/frames. "
            "Install a NumPy-compatible build (e.g., downgrade to `numpy<2` or reinstall `opencv-python`)."
        ) from exc
    cv2 = cv2_mod
    return cv2_mod


cv2 = None  # type: ignore[assignment]
_CV2_IMPORT_ERROR: Exception | None = None


def build_human_play_setup_panel(has_target, solve_config, init_state) -> Panel:
    grid = Table.grid(expand=False)
    if has_target:
        grid.add_column(justify="center")
        states_grid = Table.grid(expand=False)
        states_grid.add_row(
            Align.center("[bold blue]Start State[/bold blue]"),
            "",
            Align.center("[bold blue]Target State[/bold blue]"),
        )
        states_grid.add_row(
            init_state.str(solve_config=solve_config),
            Align.center("[bold blue] → [/bold blue]", vertical="middle"),
            str(solve_config),
        )
        grid.add_row(states_grid)
    else:
        grid.add_column()
        grid.add_row(Align.center("[bold blue]Start State[/bold blue]"))
        grid.add_row(init_state.str(solve_config=solve_config))
    return Panel(grid, title="[bold blue]Human Play Setup[/bold blue]", expand=False)


def build_human_play_layout(
    current_state,
    solve_config,
    sum_cost: float,
    action_strs: List[str],
    arrow_flag: bool,
) -> Group:
    state_panel = Panel(
        Align.center(current_state.str(solve_config=solve_config)),
        title="[bold blue]Current State[/bold blue]",
        border_style="blue",
        expand=False,
    )
    cost_text = f"Accumulated Cost: [bold green]{sum_cost}[/bold green]"

    action_table = Table(
        show_edge=True,
        title="[bold magenta]Available Actions[/bold magenta]",
        border_style="magenta",
        show_lines=True,
        expand=False,
    )

    if arrow_flag:
        action_table.show_header = False
        action_table.add_column(style="bold cyan", justify="right")
        action_table.add_column(style="bold cyan", justify="center")
        action_table.add_column(style="bold cyan", justify="left")

        up_action = next((s for s in action_strs if "↑" in s), None)
        down_action = next((s for s in action_strs if "↓" in s), None)
        left_action = next((s for s in action_strs if "←" in s), None)
        right_action = next((s for s in action_strs if "→" in s), None)

        up_display = f"[bold yellow]W[/bold yellow]\n{up_action}" if up_action else ""
        down_display = f"[bold yellow]S[/bold yellow]\n{down_action}" if down_action else ""
        left_display = f"[bold yellow]A[/bold yellow]\n{left_action}" if left_action else ""
        right_display = f"[bold yellow]D[/bold yellow]\n{right_action}" if right_action else ""

        action_table.add_row("", up_display, "")
        action_table.add_row(left_display, down_display, right_display)
    else:
        action_table.show_header = True
        action_table.header_style = "bold yellow"
        action_table.add_column("Index", justify="center")
        action_table.add_column("Action", justify="left")
        for i, action_str in enumerate(action_strs):
            action_table.add_row(f"{i+1}", f"[bold cyan]{action_str}[/bold cyan]")

    return Group(state_panel, cost_text, action_table)


def build_seed_setup_panel(
    puzzle,
    has_target: bool,
    solve_config,
    state,
    dist_text: Text,
    seed: int,
) -> Panel:
    grid = Table.grid(expand=False)

    if has_target:
        grid.add_column(justify="center")
        states_grid = Table.grid(expand=False)
        states_grid.add_row(
            Align.center("[bold blue]Start State[/bold blue]"),
            "",
            Align.center("[bold blue]Target State[/bold blue]"),
        )
        states_grid.add_row(
            Text.from_ansi(state.str(solve_config=solve_config)),
            Align.center("[bold blue] → [/bold blue]", vertical="middle"),
            Text.from_ansi(str(solve_config)),
        )
        grid.add_row(states_grid)
    else:
        grid.add_column()
        grid.add_row(Align.center("[bold blue]Start State[/bold blue]"))
        grid.add_row(Text.from_ansi(state.str(solve_config=solve_config)))

    grid.add_row(Text.assemble(Text("Dist: ", style="bold"), dist_text))
    return Panel(grid, title=f"[bold blue]Seed {seed}[/bold blue]", expand=False)


def build_result_table(
    solved: bool,
    single_search_time: float,
    generated_size: int,
    states_per_second: float,
    solved_cost: float | None,
    seed: int | None = None,
) -> Table:
    title = "[bold]Search Result[/bold]"
    if seed is not None:
        title = f"[bold]Search Result for Seed {seed}[/bold]"

    result_table = Table(title=title)
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", justify="right")
    if solved:
        result_table.add_row("Status", "[bold green]Solution Found[/bold green]")
        if solved_cost is not None:
            result_table.add_row("Cost", f"{solved_cost:.1f}")
    else:
        result_table.add_row("Status", "[bold red]No Solution Found[/bold red]")

    result_table.add_row("Search Time", f"{single_search_time:.2f} s")
    result_table.add_row(
        "Search States",
        f"{human_format(generated_size)}",
    )
    result_table.add_row("States/s", f"{human_format(states_per_second)}")
    return result_table


def build_summary_table(
    puzzle_name: str,
    seeds: List[int],
    total_solved: List[bool],
    total_search_times: List[float],
    total_states: List[int],
    states_per_sec_list: List[float],
    total_costs: List[float],
) -> Table:
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

    avg_solved = jnp.mean(jnp.array(total_solved)) * 100
    avg_time = jnp.mean(jnp.array(total_search_times))
    avg_states = jnp.mean(jnp.array(total_states))
    avg_sps = jnp.mean(jnp.array(states_per_sec_list))

    valid_costs = [c for c in total_costs if c != jnp.inf]
    avg_cost = jnp.mean(jnp.array(valid_costs)) if valid_costs else 0.0

    summary_table.add_row(
        "[bold]Average[/bold]",
        f"{avg_solved:.1f}%",
        f"{avg_time:.2f}",
        human_format(avg_states),
        human_format(avg_sps),
        f"{avg_cost:.1f}",
    )
    return summary_table


@dataclass
class PathStep:
    state: object
    cost: float
    dist: float | None
    action: int | None


def build_solution_path_panel(
    console: Console,
    path_steps: List[PathStep],
    puzzle,
    solve_config,
    cost_weight: float,
) -> Panel:
    solution_panels = []
    for step in path_steps[:-1]:
        g = step.cost
        h = step.dist if step.dist is not None else 0.0
        f = cost_weight * g + h
        title = (
            f"[bold red]g:{g:4.1f}[/bold red]|"
            f"[bold blue]h:{h:4.1f}[/bold blue]|"
            f"[bold green]f:{f:4.1f}[/bold green]"
        )
        panel_content = Table.grid(expand=False)
        panel_content.add_row(
            Align.center(Text.from_ansi(step.state.str(solve_config=solve_config)))
        )
        if step.action is None:
            action_text = "Start"
        else:
            try:
                action_text = puzzle.action_to_string(step.action)
            except (ValueError, IndexError):
                action_text = f"Action {step.action}"
        panel_content.add_row(Align.center(Text.from_ansi(f"Action: {action_text}")))
        panel_content = Align.center(panel_content)
        solution_panels.append(Panel(panel_content, title=title, border_style="blue", expand=False))

    final_step = path_steps[-1]
    g = final_step.cost
    h = final_step.dist if final_step.dist is not None else 0.0
    f = cost_weight * g + h
    final_state_title = (
        f"[bold red]g:{g:4.1f}[/bold red]|"
        f"[bold blue]h:{h:4.1f}[/bold blue]|"
        f"[bold green]f:{f:4.1f}[/bold green]"
    )
    final_panel_content = Table.grid(expand=False)
    final_panel_content.add_row(
        Align.center(Text.from_ansi(final_step.state.str(solve_config=solve_config)))
    )
    final_panel_content.add_row(Align.center("[bold green]Solved![/bold green]"))
    final_panel_content = Align.center(final_panel_content)
    solution_panels.append(
        Panel(final_panel_content, title=final_state_title, border_style="green", expand=False)
    )

    solution_path_group = []
    if solution_panels:
        arrow_width = 3  # For " → "
        available_width = console.width - 4  # Account for panel borders

        panel_widths = [console.measure(panel).maximum for panel in solution_panels]
        max_panel_width = max(panel_widths) if panel_widths else 20

        if max_panel_width > 0:
            states_per_row = (available_width + arrow_width) // (max_panel_width + arrow_width)
        else:
            states_per_row = 1
        states_per_row = max(1, min(states_per_row, len(solution_panels)))
    else:
        states_per_row = 1

    for i in range(0, len(solution_panels), states_per_row):
        chunk = solution_panels[i : i + states_per_row]
        row_grid = Table.grid(expand=False)
        row_widgets = []

        if i > 0:
            row_widgets.append(Align.center("[bold blue]↓[/bold blue]", vertical="middle"))
        else:
            row_widgets.append("  ")

        for j, panel in enumerate(chunk):
            row_widgets.append(panel)
            if j < len(chunk) - 1:
                row_widgets.append(Align.center("[bold blue] → [/bold blue]", vertical="middle"))

        row_grid.add_row(*row_widgets)
        solution_path_group.append(row_grid)

    return Panel(
        Group(*solution_path_group), title="[bold green]Solution Path[/bold green]", expand=False
    )


def save_solution_animation_and_frames(
    path_steps: List[PathStep],
    puzzle_name: str,
    solve_config,
    max_animation_time: float,
) -> str:
    import imageio

    cv2_mod = _require_cv2()
    imgs = []
    logging_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging_name = f"{puzzle_name}_{logging_time}"
    os.makedirs(f"tmp/{logging_name}", exist_ok=True)
    path_states = [step.state for step in path_steps]
    for idx, step in enumerate(path_steps):
        img = step.state.img(idx=idx, path=path_states, solve_config=solve_config)
        imgs.append(img)
        cv2_mod.imwrite(
            (
                f"tmp/{logging_name}/img_{idx}_c"
                f"{step.cost:.1f}_d"
                f"{(step.dist if step.dist is not None else 0.0):.1f}.png"
            ),
            cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2RGB),
        )
    gif_path = f"tmp/{logging_name}/animation.gif"
    num_frames = len(imgs)
    fps = 4
    if num_frames / fps > max_animation_time:
        fps = num_frames / max_animation_time
    imageio.mimsave(gif_path, imgs, fps=fps)
    return gif_path


def build_path_steps_from_nodes(
    search_result,
    path,
    puzzle,
    solve_config,
) -> List[PathStep]:
    steps: List[PathStep] = []
    for node in path:
        state = search_result.get_state(node)
        cost = float(search_result.get_cost(node))
        dist = float(search_result.get_dist(node))
        action = getattr(node, "action", None)
        if action is not None:
            action = int(action)
            if action == ACTION_PAD_INT:
                action = None
        steps.append(
            PathStep(
                state=state,
                cost=cost,
                dist=dist,
                action=action,
            )
        )
    return steps


def build_path_steps_from_actions(
    puzzle,
    solve_config,
    initial_state,
    actions: List[int],
    heuristic=None,
    q_fn=None,
    states=None,
    costs=None,
    dists=None,
) -> List[PathStep]:
    steps: List[PathStep] = []
    heuristic_params = (
        heuristic.prepare_heuristic_parameters(solve_config) if heuristic is not None else None
    )
    q_fn_params = q_fn.prepare_q_parameters(solve_config) if q_fn is not None else None

    # Normalise actions to Python ints and drop any padding sentinel.
    action_sequence: List[int] = []
    for raw_action in actions:
        action_val = int(raw_action)
        if action_val == ACTION_PAD_INT:
            break
        action_sequence.append(action_val)

    states_list = list(states) if states is not None else None
    costs_arr = np.asarray(costs) if costs is not None else None
    can_use_trace = (
        states_list is not None
        and costs_arr is not None
        and len(costs_arr) >= (len(action_sequence) + 1)
    )
    state = states_list[0] if states_list else initial_state
    running_cost = 0.0

    def _trace_cost(idx: int, default_val: float) -> float:
        if costs is not None and idx < len(costs):
            return float(costs[idx])
        return default_val

    def _trace_dist(idx: int, current_state) -> float | None:
        if dists is not None and idx < len(dists):
            val = dists[idx]
            if val is None:
                return None
            if isinstance(val, jnp.ndarray):
                val = float(val)
            return float(val)
        if heuristic is not None:
            # Prefer prepared params when available to mirror search-time evaluation.
            params = heuristic_params if heuristic_params is not None else solve_config
            return float(heuristic.distance(params, current_state))
        if q_fn is not None:
            params = q_fn_params if q_fn_params is not None else solve_config
            q_vals = q_fn.q_value(params, current_state)
            return float(jnp.min(jnp.asarray(q_vals)))
        return None

    start_action = action_sequence[0] if action_sequence else None
    start_cost = _trace_cost(0, running_cost)
    start_dist = _trace_dist(0, state)
    steps.append(PathStep(state=state, cost=start_cost, dist=start_dist, action=start_action))

    for depth, action_val in enumerate(action_sequence):
        if can_use_trace and depth + 1 < len(states_list):
            next_state = states_list[depth + 1]
            next_cost = float(costs_arr[depth + 1])
            step_cost = next_cost - float(costs_arr[depth])
            running_cost = next_cost
        else:
            neighbours, transition_cost = puzzle.get_neighbours(solve_config, state, True)
            step_cost = float(transition_cost[action_val])
            if not np.isfinite(step_cost):
                break
            running_cost += step_cost
            next_state = neighbours[action_val]
            if states_list is not None and depth + 1 < len(states_list):
                next_state = states_list[depth + 1]

        cost_val = _trace_cost(depth + 1, running_cost)
        dist_val = _trace_dist(depth + 1, next_state)

        next_action: int | None = None
        if depth + 1 < len(action_sequence):
            next_action = action_sequence[depth + 1]

        steps.append(
            PathStep(
                state=next_state,
                cost=cost_val,
                dist=dist_val,
                action=next_action,
            )
        )
        state = next_state

    return steps


def build_vmapped_setup_panel(has_target, solve_configs, states) -> Panel:
    grid = Table.grid(expand=False)
    grid.add_row(Text.from_ansi(states.str(solve_config=solve_configs)))
    if has_target:
        grid.add_row(Align.center("[bold blue]↓[/bold blue]\n", vertical="middle"))
        grid.add_row(Text.from_ansi(str(solve_configs)))
    return Panel(grid, title="[bold blue]Vmapped Search Setup[/bold blue]", expand=False)


def build_vmapped_results_table_multi(
    vmapped_search_time: float,
    total_search_times,
    vmap_size: int,
    sizes,
    vmapped_states_per_second: float,
    states_per_second: float,
    solved,
) -> Table:
    table = Table(title="[bold]Vmapped Search Results[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row(
        "Search Time",
        f"{vmapped_search_time:6.2f}s "
        f"(x{vmapped_search_time/total_search_times.sum()*vmap_size:.1f}/{vmap_size})",
    )
    table.add_row(
        "Total Search States",
        f"{int(sizes.sum())} (avg: {int(sizes.mean())})",
    )
    table.add_row(
        "States per Second",
        f"{vmapped_states_per_second:.2f} (x{vmapped_states_per_second/states_per_second:.1f} faster)",
    )
    table.add_row(
        "Solutions Found", f"{int(solved.sum())}/{len(solved)} ({float(solved.mean())*100:.2f}%)"
    )
    return table


def build_vmapped_results_table_single(
    vmapped_search_time: float,
    single_search_time: float,
    vmap_size: int,
    search_states: int,
    vmapped_states_per_second: float,
    states_per_second: float,
    solved_mean: float,
) -> Table:
    table = Table(title="[bold]Vmapped Search Result[/bold]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row(
        "Search Time",
        f"{vmapped_search_time:6.2f}s (x{vmapped_search_time/single_search_time:.1f}/{vmap_size})",
    )
    table.add_row(
        "Search States",
        f"{human_format(int(search_states))} ({human_format(vmapped_states_per_second)} states/s)",
    )
    table.add_row("Speedup", f"x{vmapped_states_per_second/states_per_second:.1f}")
    table.add_row("Solutions Found", f"{solved_mean*100:.2f}%")
    return table
