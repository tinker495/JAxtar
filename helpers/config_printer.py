import json
from numbers import Number

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from helpers.formatting import human_format
from helpers.util import convert_to_serializable_dict


def _is_single_level(config: dict) -> bool:
    if not isinstance(config, dict):
        return True
    return not any(isinstance(v, dict) for v in config.values())


def _format_value(value, apply_human_format=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return human_format(value) if apply_human_format else value
    if isinstance(value, dict):
        return {
            k: _format_value(v, apply_human_format=apply_human_format) for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        formatted = [_format_value(v, apply_human_format=False) for v in value]
        if isinstance(value, tuple):
            return tuple(formatted)
        if isinstance(value, set):
            return set(formatted)
        return formatted
    return value


def print_config(title: str, config: dict):
    config = convert_to_serializable_dict(config)
    config = _format_value(config)
    console = Console()
    TRUNCATE_LENGTH = 80
    TRUNCATE_SEQ_LENGTH = 10

    def maybe_truncate(value):
        if isinstance(value, str):
            if len(value) > TRUNCATE_LENGTH:
                return value[:TRUNCATE_LENGTH] + " ..."
            return value
        elif isinstance(value, (tuple, list, set)):
            if len(value) > TRUNCATE_SEQ_LENGTH:
                half = TRUNCATE_SEQ_LENGTH // 2
                shown_front = list(value)[:half]
                shown_back = list(value)[-half:]
                str_value = f"[{', '.join(str(x) for x in shown_front)} ... {', '.join(str(x) for x in shown_back)}]"
                return str_value
            return value
        elif isinstance(value, dict) or value is None:
            return value
        s = str(value)
        if len(s) > TRUNCATE_LENGTH:
            return s[:TRUNCATE_LENGTH] + " ..."
        return value

    def add_node(parent: Tree, key: str, value):
        value = _format_value(value)
        if isinstance(value, dict):
            style = "bold magenta"
            branch = parent.add(f"[{style}]{key}[/{style}]")
            for k, v in value.items():
                add_node(branch, k, v)
        else:
            display_value = maybe_truncate(value)

            item_grid = Table.grid(padding=(0, 1))
            item_grid.add_column(no_wrap=True)
            item_grid.add_column()

            if isinstance(display_value, Text):
                value_renderable = display_value
            elif isinstance(display_value, str):
                value_renderable = Text(display_value)
            else:
                value_renderable = Pretty(display_value)

            item_grid.add_row(Text(f"{key}:", style="bold magenta"), value_renderable)
            parent.add(item_grid)

    if not config:
        layout = Text("Configuration is empty.", justify="center")
    elif len(config) <= 2 or _is_single_level(config):
        layout = Tree("", guide_style="bright_blue")
        for key, value in config.items():
            add_node(layout, key, value)
    else:
        main_key = max(config.keys(), key=lambda k: len(json.dumps(config.get(k), default=str)))

        main_tree = Tree("", guide_style="bright_blue")
        add_node(main_tree, main_key, config[main_key])

        side_tree = Tree("", guide_style="bright_blue")
        for key, value in config.items():
            if key != main_key:
                add_node(side_tree, key, value)

        layout = Columns([main_tree, side_tree], equal=True, expand=True)

    panel = Panel(
        layout,
        title=f"[bold green]{title}[/bold green]",
        border_style="dim",
        expand=False,
    )
    console.print(panel)
