from __future__ import annotations

import os
from datetime import datetime
from typing import List

import cv2
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
) -> Table:
    result_table = Table(title="[bold]Search Result[/bold]")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", justify="right")
    if solved:
        result_table.add_row("Status", "[bold green]Solution Found[/bold green]")
        if solved_cost is not None:
            result_table.add_row("Cost", f"{solved_cost:.1f}")
    else:
        result_table.add_row("Status", "[bold red]No Solution Found[/bold red]")

    result_table.add_row("Search Time", f"{single_search_time:.2f} s")
    result_table.add_row("Search States", f"{generated_size}")
    result_table.add_row("States/s", f"{states_per_second:.2f}")
    return result_table


def build_solution_path_panel(
    console: Console,
    search_result,
    path,
    puzzle,
    solve_config,
    cost_weight: float,
) -> Panel:
    solution_panels = []
    for p in path[:-1]:
        g = search_result.get_cost(p)
        h = search_result.get_dist(p)
        f = cost_weight * g + h
        title = (
            f"[bold red]g:{g:4.1f}[/bold red]|"
            f"[bold blue]h:{h:4.1f}[/bold blue]|"
            f"[bold green]f:{f:4.1f}[/bold green]"
        )
        panel_content = Table.grid(expand=False)
        panel_content.add_row(
            Align.center(Text.from_ansi(search_result.get_state(p).str(solve_config=solve_config)))
        )
        panel_content.add_row(
            Align.center(Text.from_ansi(f"Action: {puzzle.action_to_string(p.action)}"))
        )
        panel_content = Align.center(panel_content)
        solution_panels.append(Panel(panel_content, title=title, border_style="blue", expand=False))

    g = search_result.get_cost(path[-1])
    h = 0.0
    f = cost_weight * g + h
    final_state_title = (
        f"[bold red]g:{g:4.1f}[/bold red]|"
        f"[bold blue]h:{h:4.1f}[/bold blue]|"
        f"[bold green]f:{f:4.1f}[/bold green]"
    )
    final_panel_content = Table.grid(expand=False)
    final_panel_content.add_row(
        Align.center(
            Text.from_ansi(search_result.get_state(path[-1]).str(solve_config=solve_config))
        )
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
    search_result,
    path,
    puzzle_name: str,
    solve_config,
    max_animation_time: float,
) -> str:
    import imageio

    imgs = []
    logging_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging_name = f"{puzzle_name}_{logging_time}"
    os.makedirs(f"tmp/{logging_name}", exist_ok=True)
    path_states = [search_result.get_state(p) for p in path]
    for idx, p in enumerate(path):
        img = search_result.get_state(p).img(idx=idx, path=path_states, solve_config=solve_config)
        imgs.append(img)
        cv2.imwrite(
            (
                f"tmp/{logging_name}/img_{idx}_c"
                f"{search_result.get_cost(p):.1f}_d"
                f"{search_result.get_dist(p):.1f}.png"
            ),
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )
    gif_path = f"tmp/{logging_name}/animation.gif"
    num_frames = len(imgs)
    fps = 4
    if num_frames / fps > max_animation_time:
        fps = num_frames / max_animation_time
    imageio.mimsave(gif_path, imgs, fps=fps)
    return gif_path


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
        f"{int(search_states)} ({vmapped_states_per_second:.2f} states/s)",
    )
    table.add_row("Speedup", f"x{vmapped_states_per_second:.1f}")
    table.add_row("Solutions Found", f"{solved_mean*100:.2f}%")
    return table
