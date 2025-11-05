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
    """
    Checks if the config dictionary has only one level of hierarchy.
    """
    if not isinstance(config, dict):
        return True  # Not a dictionary, so can't have multiple levels.
    return not any(isinstance(v, dict) for v in config.values())


def _format_value(value):
    """Recursively apply human-friendly formatting to numeric values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return human_format(value)
    if isinstance(value, dict):
        return {k: _format_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        formatted = [_format_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(formatted)
        if isinstance(value, set):
            return set(formatted)
        return formatted
    return value


def print_config(title: str, config: dict):
    """
    Prints a configuration in a hierarchical tree layout, wrapped in a panel.
    Uses a two-column layout for a larger number of top-level items.
    """
    config = convert_to_serializable_dict(config)
    config = _format_value(config)
    console = Console()
    TRUNCATE_LENGTH = 80  # Character length threshold for truncation
    TRUNCATE_SEQ_LENGTH = 10  # Number of items to show for long tuples/lists/sets

    def maybe_truncate(value):
        # Truncate long strings
        if isinstance(value, str):
            if len(value) > TRUNCATE_LENGTH:
                return value[:TRUNCATE_LENGTH] + " ..."
            return value
        # Truncate long tuples/lists/sets
        elif isinstance(value, (tuple, list, set)):
            if len(value) > TRUNCATE_SEQ_LENGTH:
                # Symmetrical truncation
                half = TRUNCATE_SEQ_LENGTH // 2
                shown_front = list(value)[:half]
                shown_back = list(value)[-half:]
                str_value = f"[{', '.join(str(x) for x in shown_front)} ... {', '.join(str(x) for x in shown_back)}]"
                return str_value
            return value
        # Don't truncate dicts or None
        elif isinstance(value, dict) or value is None:
            return value
        # Fallback: truncate string representation if too long
        s = str(value)
        if len(s) > TRUNCATE_LENGTH:
            return s[:TRUNCATE_LENGTH] + " ..."
        return value

    def add_node(parent: Tree, key: str, value):
        """Recursively adds config items to the tree."""
        value = _format_value(value)
        if isinstance(value, dict):
            # Create a new branch for a dictionary
            style = "bold magenta"
            branch = parent.add(f"[{style}]{key}[/{style}]")
            for k, v in value.items():
                add_node(branch, k, v)
        else:
            # Add a leaf node for a simple value, truncating if necessary.
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
    # For a small number of items or a single-level hierarchy, use a single, wide tree.
    elif len(config) <= 2 or _is_single_level(config):
        layout = Tree("", guide_style="bright_blue")
        for key, value in config.items():
            add_node(layout, key, value)
    # For more items, put the longest item in the first column, and the rest in the second.
    else:
        main_key = None
        try:
            # Select the key with the longest JSON string representation as the main item.
            main_key = max(config.keys(), key=lambda k: len(json.dumps(config.get(k), default=str)))
        except (TypeError, OverflowError, ValueError):
            # Fallback if max() is empty or another error occurs.
            main_key = None

        # If a main key is found, use the main/side layout.
        if main_key:
            main_tree = Tree("", guide_style="bright_blue")
            add_node(main_tree, main_key, config[main_key])

            side_tree = Tree("", guide_style="bright_blue")
            for key, value in config.items():
                if key != main_key:
                    add_node(side_tree, key, value)

            layout = Columns([main_tree, side_tree], equal=True, expand=True)
        # Fallback to the 50/50 split if no main key was determined.
        else:
            items = list(config.items())
            midpoint = (len(items) + 1) // 2
            left_items, right_items = items[:midpoint], items[midpoint:]

            left_tree = Tree("", guide_style="bright_blue")
            for key, value in left_items:
                add_node(left_tree, key, value)

            right_tree = Tree("", guide_style="bright_blue")
            if right_items:
                for key, value in right_items:
                    add_node(right_tree, key, value)

            layout = Columns([left_tree, right_tree], equal=True, expand=True)

    panel = Panel(
        layout, title=f"[bold green]{title}[/bold green]", border_style="dim", expand=False
    )
    console.print(panel)
