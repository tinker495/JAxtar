import json

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text


def print_config(title: str, config: dict):
    """
    Prints a configuration in a dynamic main/side panel layout.

    Truncates long non-dict values (especially long tuples, lists, and strings).
    """
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
                # Show first N items, then ...
                shown_front = list(value)[: TRUNCATE_SEQ_LENGTH // 2 - 1]
                shown_back = list(value)[-TRUNCATE_SEQ_LENGTH // 2 + 1 :]
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

    main_key = None
    # Use sidebar layout only if there are more than 2 config items.
    if len(config) > 2:
        # Only consider dict values for main_key selection.
        dict_keys = [k for k, v in config.items() if isinstance(v, dict)]
        if dict_keys:
            try:
                # Select the dict key with the longest JSON string representation.
                main_key = max(
                    dict_keys, key=lambda k: len(json.dumps(config.get(k, ""), default=str))
                )
            except (TypeError, OverflowError):
                main_key = None

    # If main_key is determined, use sidebar layout.
    if main_key:
        side_items = {k: v for k, v in config.items() if k != main_key}
        main_item_value = config[main_key]

        layout = Table.grid(expand=True, padding=1)
        layout.add_column(ratio=35)  # Sidebar
        layout.add_column(ratio=65)  # Main content

        side_table = Table.grid(padding=(0, 1))
        for key, value in side_items.items():
            display_value = maybe_truncate(value)
            side_table.add_row(Text(f"{key}:", style="bold magenta"), Pretty(display_value))

        main_content = Table.grid(padding=(0, 1))
        main_content.add_row(
            Text(f"{main_key}:", style="bold magenta"), Pretty(main_item_value, expand_all=True)
        )

        layout.add_row(side_table, main_content)
    # Otherwise, use the previous multi-column layout.
    else:
        renderables = []
        if config:
            for key, value in config.items():
                display_value = maybe_truncate(value)
                item_grid = Table.grid(padding=(0, 1), expand=True)
                item_grid.add_column(style="bold magenta", no_wrap=True)
                item_grid.add_column(ratio=1)
                item_grid.add_row(f"{key}:", Pretty(display_value, overflow="fold"))
                renderables.append(item_grid)
        layout = Columns(renderables, equal=True, expand=True) if renderables else Text("")

    panel = Panel(layout, title=f"[bold green]{title}[/bold green]", border_style="dim")
    console.print(panel)
