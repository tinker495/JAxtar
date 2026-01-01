import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from typing import Optional

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

    def render(self, *, final: bool = False) -> str:
        # Trim trailing empty lines for cleanliness but preserve intentional blank lines.
        lines = self._lines.copy()
        # Remove extra empty lines at the end that result from clears.
        while len(lines) > 1 and lines[-1] == "":
            lines.pop()
        result = "\n".join(line.rstrip() for line in lines)
        if final:
            if self._line_terminated or result:
                result = result + "\n"
        elif self._line_terminated:
            result = result + "\n"
        return result


class _LogCapture:
    """Incrementally mirror sanitized output into the log file."""

    def __init__(self, log_file, offset: int):
        self._file = log_file
        self._offset = offset
        self._sanitizer = _LogSanitizer()

    def write(self, data: str):
        if not data:
            return
        self._sanitizer.process(data)
        self._flush(final=False)

    def flush(self):
        self._flush(final=False)

    def finalize(self):
        self._flush(final=True)

    def _flush(self, *, final: bool):
        if self._file.closed:
            return
        content = self._sanitizer.render(final=final)
        self._file.seek(self._offset)
        if content:
            self._file.write(content)
        else:
            # Ensure the body is cleared if nothing remains
            pass
        self._file.truncate()
        self._file.flush()


class _TeeStream(TextIOBase):
    """Mirror writes to the primary stream and capture sanitized output."""

    def __init__(self, primary: TextIOBase, capture: _LogCapture):
        self._primary = primary
        self._capture = capture

    def write(self, data):
        self._primary.write(data)
        self._capture.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._capture.flush()

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

    with log_path.open("w+", encoding="utf-8") as log_file:
        header = f"\n===== Run started {datetime.now().isoformat()} =====\n"
        log_file.write(header)
        log_file.flush()
        body_offset = log_file.tell()

        capture = _LogCapture(log_file, body_offset)
        tee_out = _TeeStream(original_stdout, capture)
        tee_err = _TeeStream(original_stderr, capture)

        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                yield
        finally:
            capture.finalize()
            footer = f"\n===== Run ended {datetime.now().isoformat()} =====\n"
            log_file.write(footer)
            log_file.flush()
