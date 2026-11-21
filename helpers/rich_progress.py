import threading
from collections.abc import Sized
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


class MetricsColumn(ProgressColumn):
    def __init__(self, metrics: Dict[str, Any] | None = None):
        super().__init__()
        self.metrics = metrics or {}

    def render(self, task):
        if not self.metrics:
            return Text("")

        metric_parts = []
        for key, value in self.metrics.items():
            metric_parts.append(f"{key}: {_format_metric_value(value)}")

        return Text(" | ".join(metric_parts), style="cyan")


class RichProgressBar:
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        leave: bool = True,
        disable: bool = False,
        initial: int = 0,
        **_ignored_tqdm_kwargs,
    ):
        self.desc = desc or "Processing"
        self.total = total
        self.leave = leave
        self.disable = disable
        self.initial = initial
        self.n = initial

        self.current_desc = desc or "Processing"
        self.current_desc_dict = {}

        self.console = Console()
        self.metrics = {}
        self.metrics_column = MetricsColumn(self.metrics)

        self.progress = Progress(
            SpinnerColumn(),
            BarColumn(complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            self.metrics_column,
            console=self.console,
            refresh_per_second=10,
            # rich's default 30s speed window goes empty when a single step
            # takes longer than that, collapsing the ETA to "-:--:--".
            speed_estimate_period=24 * 60 * 60,
        )

        self.task_id: Optional[TaskID] = None
        self.live: Optional[Live] = None
        self._lock = threading.Lock()
        self._pause_depth = 0

        if iterable is not None:
            if total is None:
                try:
                    self.total = len(iterable) if isinstance(iterable, Sized) else None
                except (TypeError, AttributeError):
                    self.total = None
            self.iterable = iterable
        else:
            self.iterable = None

    def _create_display_layout(self):
        table = Table.grid(padding=(0, 0))
        table.add_column(justify="left")

        if self.current_desc:
            desc_text = Text(self.current_desc, style="bold blue")
            table.add_row(desc_text)

        if self.current_desc_dict:
            for key, value in self.current_desc_dict.items():
                table.add_row(Text(f"{key} : {_format_metric_value(value)}", style="cyan"))

        table.add_row(self.progress)

        return table

    def __enter__(self):
        if not self.disable:
            self.task_id = self.progress.add_task(
                "",
                total=self.total,
                completed=self.initial,
            )
            self.live = Live(
                self._create_display_layout(),
                console=self.console,
                refresh_per_second=10,
                transient=not self.leave,
            )
            self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable and self.live:
            self.live.stop()

    def __iter__(self):
        if self.iterable is None:
            raise TypeError("'RichProgressBar' object is not iterable")

        with self:
            for obj in self.iterable:
                yield obj
                self.update(1)

    def update(self, n: int = 1):
        if self.disable or self.task_id is None:
            return

        with self._lock:
            self.n += n
            self.metrics_column.metrics = self.metrics.copy()
            self.progress.update(self.task_id, advance=n)
            self._refresh_display()

    def set_description(
        self,
        desc: Optional[str] = None,
        desc_dict: Optional[Dict[str, Any]] = None,
        refresh: bool = True,
    ):
        if desc is None and desc_dict is None:
            raise ValueError("Either desc or desc_dict must be provided")

        if desc is not None:
            self.current_desc = desc
            self.desc = desc

        if desc_dict is not None:
            self.current_desc_dict = desc_dict.copy()

        if refresh:
            self._refresh_display()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs):
        if ordered_dict is not None:
            self.metrics.update(ordered_dict)
        if kwargs:
            self.metrics.update(kwargs)

        if not self.disable and self.task_id is not None:
            with self._lock:
                self.metrics_column.metrics = self.metrics.copy()
                if refresh:
                    self._refresh_display()

    @contextmanager
    def pause(self):
        """Pause live updates, allowing other rich output to render cleanly."""
        if self.disable or self.live is None:
            yield
            return

        self._pause_depth += 1
        if self._pause_depth > 1:
            try:
                yield
            finally:
                self._pause_depth -= 1
            return

        current_live = self.live
        current_live.stop()
        self.live = None
        self.console.rule()
        try:
            yield
        finally:
            try:
                self.console.rule()
                self.live = Live(
                    self._create_display_layout(),
                    console=self.console,
                    refresh_per_second=10,
                    transient=not self.leave,
                )
                self.live.start()
                self._refresh_display()
            finally:
                self._pause_depth -= 1

    def _refresh_display(self):
        if not self.disable and self.live:
            self.live.update(self._create_display_layout())


def trange(*args, **kwargs):
    if len(args) == 1:
        kwargs["total"] = args[0]
        iterable = range(args[0])
    elif len(args) == 2:
        start, stop = args
        kwargs["total"] = stop - start
        iterable = range(start, stop)
    elif len(args) == 3:
        start, stop, step = args
        kwargs["total"] = len(range(start, stop, step))
        iterable = range(start, stop, step)
    else:
        raise TypeError("trange expected at most 3 arguments, got {}".format(len(args)))

    return RichProgressBar(iterable=iterable, **kwargs)


def tqdm(iterable=None, **kwargs):
    return RichProgressBar(iterable=iterable, **kwargs)
