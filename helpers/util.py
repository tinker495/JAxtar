import json
import re
import sys
import time
from collections.abc import MutableMapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from pydantic import BaseModel

from JAxtar.stars.search_base import Current, SearchResult


def convert_to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        # Recursively process the dict representation
        return convert_to_serializable_dict(obj.dict())
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable_dict(i) for i in obj]
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_hashable(val):
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return val


def display_value(val):
    # Convert tuples back to lists for display, and pretty-print JSON strings
    if isinstance(val, tuple):
        return str(list(val))
    try:
        loaded = json.loads(val)
        if isinstance(loaded, dict) or isinstance(loaded, list):
            return json.dumps(loaded, indent=2)
    except Exception:
        pass
    return str(val)


def vmapping_init_target(
    puzzle: Puzzle, vmap_size: int, start_state_seeds: list[int]
) -> tuple[Puzzle.SolveConfig, Puzzle.State]:
    start_state_seed = start_state_seeds[0]
    solve_configs, states = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
    solve_configs = xnp.tile(solve_configs[jnp.newaxis, ...], (vmap_size, 1))
    states = xnp.tile(states[jnp.newaxis, ...], (vmap_size, 1))

    if len(start_state_seeds) > 1:
        for i, start_state_seed in enumerate(start_state_seeds[1:vmap_size]):
            new_solve_configs, new_states = puzzle.get_inits(jax.random.PRNGKey(start_state_seed))
            states = states.at[i + 1].set(new_states)
            solve_configs = solve_configs.at[i + 1].set(new_solve_configs)
    return solve_configs, states


def vmapping_search(
    puzzle: Puzzle,
    star_fn: callable,
    vmap_size: int,
    show_compile_time: bool = False,
):
    """
    Vmap the search function over the batch dimension.
    """

    empty_states = puzzle.State.default((vmap_size,))
    empty_solve_configs = puzzle.SolveConfig.default((vmap_size,))
    vmapped_star = jax.jit(jax.vmap(star_fn, in_axes=(0, 0)))
    if show_compile_time:
        print("initializing vmapped jit")
        start = time.time()
    vmapped_star(empty_solve_configs, empty_states)
    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")
    return vmapped_star


def vmapping_get_state(search_result: SearchResult, idx: Current):
    return jax.vmap(SearchResult.get_state, in_axes=(0, 0))(search_result, idx)


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _LogSanitizer:
    """Simulate terminal control sequences to keep progress output tidy."""

    def __init__(self):
        self._lines: list[str] = [""]
        self._cursor_row = 0
        self._cursor_col = 0
        self._saved_cursor: Optional[tuple[int, int]] = None
        self._line_terminated = False

    def _ensure_row(self, row: int):
        while len(self._lines) <= row:
            self._lines.append("")

    def _set_line(self, row: int, value: str):
        self._ensure_row(row)
        self._lines[row] = value

    def _get_line(self, row: int) -> str:
        self._ensure_row(row)
        return self._lines[row]

    def _write_char(self, char: str):
        line = self._get_line(self._cursor_row)
        if self._cursor_col > len(line):
            line = line + " " * (self._cursor_col - len(line))
        if self._cursor_col == len(line):
            line = line + char
        else:
            line = line[: self._cursor_col] + char + line[self._cursor_col + 1 :]
        self._set_line(self._cursor_row, line)
        self._cursor_col += 1
        self._line_terminated = False

    def _newline(self):
        self._cursor_row += 1
        self._ensure_row(self._cursor_row)
        self._cursor_col = 0
        self._line_terminated = True

    def _clear_line_from_cursor(self):
        line = self._get_line(self._cursor_row)
        if self._cursor_col <= len(line):
            line = line[: self._cursor_col]
            self._set_line(self._cursor_row, line)

    def _clear_entire_line(self):
        self._set_line(self._cursor_row, "")
        self._cursor_col = 0

    def _clear_screen_from_cursor(self):
        self._clear_line_from_cursor()
        del self._lines[self._cursor_row + 1 :]
        self._ensure_row(self._cursor_row + 1)

    def _clear_screen(self):
        self._lines = [""]
        self._cursor_row = 0
        self._cursor_col = 0
        self._line_terminated = False

    def _handle_csi(self, sequence: str):
        if not sequence.startswith("\x1b["):
            return
        params_part = sequence[2:-1]
        command = sequence[-1]

        # Ignore private mode settings (e.g., ?25l)
        if params_part.startswith("?"):
            return

        params = []
        if params_part:
            for part in params_part.split(";"):
                if part == "":
                    params.append(0)
                else:
                    try:
                        params.append(int(part))
                    except ValueError:
                        params.append(0)

        if command == "K":
            mode = params[0] if params else 0
            if mode in (0, 2):
                self._clear_line_from_cursor()
                if mode == 2:
                    self._cursor_col = 0
            elif mode == 1:
                line = self._get_line(self._cursor_row)
                if self._cursor_col <= len(line):
                    line = line[self._cursor_col :]
                    self._set_line(self._cursor_row, line)
                    self._cursor_col = 0
        elif command == "J":
            mode = params[0] if params else 0
            if mode in (0, 1):
                self._clear_screen_from_cursor()
            elif mode == 2:
                self._clear_screen()
        elif command == "A":
            move = params[0] if params else 1
            self._cursor_row = max(0, self._cursor_row - move)
            line = self._get_line(self._cursor_row)
            self._cursor_col = min(self._cursor_col, len(line))
        elif command == "B":
            move = params[0] if params else 1
            self._cursor_row = min(self._cursor_row + move, len(self._lines) - 1)
            line = self._get_line(self._cursor_row)
            self._cursor_col = min(self._cursor_col, len(line))
        elif command == "C":
            move = params[0] if params else 1
            self._cursor_col += move
        elif command == "D":
            move = params[0] if params else 1
            self._cursor_col = max(0, self._cursor_col - move)
        elif command in ("E", "F"):
            move = params[0] if params else 1
            if command == "E":
                self._cursor_row = min(self._cursor_row + move, len(self._lines) - 1)
            else:
                self._cursor_row = max(0, self._cursor_row - move)
            self._cursor_col = 0
        elif command == "G":
            column = params[0] if params else 1
            self._cursor_col = max(0, column - 1)
        elif command in ("H", "f"):
            row = params[0] if params else 1
            col = params[1] if len(params) > 1 else 1
            self._cursor_row = max(0, row - 1)
            self._cursor_col = max(0, col - 1)
            self._ensure_row(self._cursor_row)
        elif command == "s":
            self._saved_cursor = (self._cursor_row, self._cursor_col)
        elif command == "u":
            if self._saved_cursor:
                row, col = self._saved_cursor
                self._cursor_row = min(row, len(self._lines) - 1)
                self._cursor_col = col
        # Ignore styling commands (m), visibility toggles, etc.

    def process(self, data: str):
        if not data:
            return
        idx = 0
        length = len(data)
        while idx < length:
            char = data[idx]
            if char == "\x1b":
                match = _ANSI_ESCAPE_RE.match(data, idx)
                if match:
                    self._handle_csi(match.group())
                    idx = match.end()
                    continue
            if char == "\r":
                self._cursor_col = 0
            elif char == "\n":
                self._newline()
            elif char == "\b":
                if self._cursor_col > 0:
                    self._cursor_col -= 1
            else:
                self._write_char(char)
            idx += 1

    def finalize(self) -> str:
        # Trim trailing empty lines for cleanliness but preserve intentional blank lines.
        lines = self._lines.copy()
        # Remove extra empty lines at the end that result from clears.
        while len(lines) > 1 and lines[-1] == "":
            lines.pop()
        result = "\n".join(line.rstrip() for line in lines)
        if self._line_terminated:
            result = result + "\n"
        return result


class _TeeStream(TextIOBase):
    """Mirror writes to the primary stream and capture sanitized output."""

    def __init__(self, primary: TextIOBase, sanitizer: _LogSanitizer):
        self._primary = primary
        self._sanitizer = sanitizer

    def write(self, data):
        self._primary.write(data)
        self._sanitizer.process(data)
        return len(data)

    def flush(self):
        self._primary.flush()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()


@contextmanager
def tee_console(log_path: Path):
    """Context manager that mirrors stdout/stderr into a sanitized log file."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("a", encoding="utf-8") as log_file:
        header = f"\n===== Run started {datetime.now().isoformat()} =====\n"
        log_file.write(header)
        log_file.flush()

        sanitizer = _LogSanitizer()
        tee_out = _TeeStream(original_stdout, sanitizer)
        tee_err = _TeeStream(original_stderr, sanitizer)

        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                yield
        finally:
            final_output = sanitizer.finalize()
            if final_output:
                log_file.write(final_output)
            footer = f"\n===== Run ended {datetime.now().isoformat()} =====\n"
            log_file.write(footer)
            log_file.flush()
