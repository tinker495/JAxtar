import json

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text


def print_config(title: str, config: dict):
    """Prints a configuration in a dynamic main/side panel layout."""
    console = Console()

    main_key = None
    # 3개 이상의 설정 항목이 있을 때만 사이드바 레이아웃을 사용합니다.
    if len(config) > 2:
        # 가장 긴 값(JSON 문자열 기준)을 가진 키를 main_key로 동적 결정합니다.
        try:
            main_key = max(config, key=lambda k: len(json.dumps(config.get(k, ""), default=str)))
        except (TypeError, OverflowError):
            # json.dumps가 실패할 경우를 대비한 안전 장치
            main_key = None

    # main_key가 결정되었으면 사이드바 레이아웃을 사용합니다.
    if main_key:
        side_items = {k: v for k, v in config.items() if k != main_key}
        main_item_value = config[main_key]

        layout = Table.grid(expand=True, padding=1)
        layout.add_column(ratio=35)  # 사이드바
        layout.add_column(ratio=65)  # 메인 콘텐츠

        side_table = Table.grid(padding=(0, 1))
        for key, value in side_items.items():
            side_table.add_row(Text(f"{key}:", style="bold magenta"), Pretty(value))

        main_content = Table.grid(padding=(0, 1))
        main_content.add_row(
            Text(f"{main_key}:", style="bold magenta"), Pretty(main_item_value, expand_all=True)
        )

        layout.add_row(side_table, main_content)
    # 그렇지 않으면, 이전의 다중 열 레이아웃을 사용합니다.
    else:
        renderables = []
        if config:
            for key, value in config.items():
                item_grid = Table.grid(padding=(0, 1), expand=True)
                item_grid.add_column(style="bold magenta", no_wrap=True)
                item_grid.add_column(ratio=1)
                item_grid.add_row(f"{key}:", Pretty(value, overflow="fold"))
                renderables.append(item_grid)
        layout = Columns(renderables, equal=True, expand=True) if renderables else Text("")

    panel = Panel(layout, title=f"[bold green]{title}[/bold green]", border_style="dim")
    console.print(panel)
