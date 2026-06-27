"""
Rich-based progress bar that provides tqdm-compatible API with enhanced visual features.

This module provides a drop-in replacement for tqdm that uses rich for beautiful,
real-time progress display with enhanced metrics visualization.
"""

import threading
import time
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
    """Custom column for displaying training metrics."""

    def __init__(self, metrics: Dict[str, Any] = None):
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
    """Rich-based progress bar with tqdm-compatible API."""

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        leave: bool = True,
        file=None,
        ncols: Optional[int] = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: Optional[int] = None,
        ascii: Optional[bool] = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: Optional[str] = None,
        initial: int = 0,
        position: Optional[int] = None,
        postfix: Optional[Dict] = None,
        unit_divisor: int = 1000,
        write_bytes: bool = False,
        lock_args: Optional[tuple] = None,
        nrows: Optional[int] = None,
        colour: Optional[str] = None,
        delay: float = 0,
        **kwargs,
    ):
        self.desc = desc or "Processing"
        self.total = total
        self.leave = leave
        self.disable = disable
        self.unit = unit
        self.initial = initial
        self.position = position
        self.n = initial
        self.last_print_n = initial
        self.start_time = time.time()
        self.last_time = self.start_time

        # Store description components separately
        self.current_desc = desc or "Processing"
        self.current_desc_dict = {}

        # Rich components
        self.console = Console()
        self.metrics = {}
        self.metrics_column = MetricsColumn(self.metrics)

        # Create progress without description in the task line
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
        )

        self.task_id: Optional[TaskID] = None
        self.live: Optional[Live] = None
        self._lock = threading.Lock()
        self._pause_depth = 0

        # Handle iterable
        if iterable is not None:
            if total is None:
                try:
                    self.total = len(iterable)
                except (TypeError, AttributeError):
                    self.total = None
            self.iterable = iterable
        else:
            self.iterable = None

    def _create_display_layout(self):
        """Create the display layout with description above progress bar."""
        # Create a table to hold description lines and progress
        table = Table.grid(padding=(0, 0))
        table.add_column(justify="left")

        # Add main description line
        if self.current_desc:
            desc_text = Text(self.current_desc, style="bold blue")
            table.add_row(desc_text)

        # Add each metric as a separate line with tracking info
        if self.current_desc_dict:
            for key, value in self.current_desc_dict.items():
                table.add_row(Text(f"{key} : {_format_metric_value(value)}", style="cyan"))

        # Add progress bar
        table.add_row(self.progress)

        return table

    def __enter__(self):
        if not self.disable:
            self.task_id = self.progress.add_task(
                "",
                total=self.total,
                completed=self.initial,
            )
            # Start Live display with custom layout
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
        """Update the progress bar."""
        if self.disable or self.task_id is None:
            return

        with self._lock:
            self.n += n
            # Update the metrics column with fresh data
            self.metrics_column.metrics = self.metrics.copy()
            self.progress.update(self.task_id, advance=n)
            self._refresh_display()

    def set_description(
        self,
        desc: Optional[str] = None,
        desc_dict: Optional[Dict[str, Any]] = None,
        refresh: bool = True,
    ):
        """Set the description of the progress bar.

        Args:
            desc: Optional string description
            desc_dict: Optional dictionary of key-value pairs to show.
            refresh: Whether to refresh the progress bar after updating
        """
        if desc is None and desc_dict is None:
            raise ValueError("Either desc or desc_dict must be provided")

        # Update the stored description components
        if desc is not None:
            self.current_desc = desc
            self.desc = desc

        if desc_dict is not None:
            self.current_desc_dict = desc_dict.copy()

        # Refresh display if requested
        if refresh:
            self._refresh_display()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs):
        """Set postfix (additional stats) with automatic formatting."""
        if ordered_dict is not None:
            self.metrics.update(ordered_dict)
        if kwargs:
            self.metrics.update(kwargs)

        if not self.disable and self.task_id is not None:
            with self._lock:
                # Update metrics in the column
                self.metrics_column.metrics = self.metrics.copy()
                # Force refresh by updating the display
                if refresh:
                    self._refresh_display()

    @contextmanager
    def pause(self):
        """Pause live updates, allowing other rich output to render cleanly."""
        if self.disable or self.live is None:
            yield
            return

        # Support nested pauses gracefully
        self._pause_depth += 1
        if self._pause_depth > 1:
            try:
                yield
            finally:
                self._pause_depth -= 1
            return

        # Capture current live instance and stop refreshing the progress display
        current_live = self.live
        current_live.stop()
        self.live = None
        self.console.rule()
        try:
            yield
        finally:
            try:
                self.console.rule()
                # Recreate live display to resume progress rendering
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

    def close(self):
        """Clean up the progress bar."""
        if not self.disable and self.live:
            self.live.stop()

    def clear(self, nolock: bool = False):
        """Clear the progress bar."""
        pass  # Rich handles this automatically

    def refresh(self):
        """Force refresh the progress bar."""
        if not self.disable:
            self._refresh_display()

    def reset(self, total: Optional[int] = None):
        """Reset the progress bar."""
        if total is not None:
            self.total = total
        self.n = self.initial
        if not self.disable and self.task_id is not None:
            with self._lock:
                self.progress.reset(self.task_id, total=self.total)
                self._refresh_display()

    @property
    def format_dict(self):
        """Return a dictionary with current progress information."""
        elapsed = time.time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0

        return {
            "n": self.n,
            "total": self.total,
            "elapsed": elapsed,
            "rate": rate,
            "desc": self.desc,
            "percentage": (self.n / self.total * 100) if self.total else 0,
        }

    def _refresh_display(self):
        """Refresh the live display with updated layout."""
        if not self.disable and self.live:
            self.live.update(self._create_display_layout())


def trange(*args, **kwargs):
    """Rich-based replacement for tqdm.trange."""
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
    """Rich-based replacement for tqdm.tqdm."""
    return RichProgressBar(iterable=iterable, **kwargs)
