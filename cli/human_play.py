import click
import jax
import jax.numpy as jnp
from puxle import Puzzle
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from helpers.visualization import build_human_play_layout, build_human_play_setup_panel

from .options import human_play_options, puzzle_options


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
                print("Invalid input, try again")
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
