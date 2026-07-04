import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from io import TextIOBase
from pathlib import Path

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _LogSanitizer:
    """Keep mirrored logs readable by stripping ANSI and folding progress updates."""

    def __init__(self):
        self._lines: list[str] = [""]
        self._line_terminated = False

    def process(self, data: str):
        if not data:
            return
        for char in _ANSI_ESCAPE_RE.sub("", data):
            if char == "\r":
                self._lines[-1] = ""
                self._line_terminated = False
            elif char == "\n":
                self._lines.append("")
                self._line_terminated = True
            elif char == "\b":
                self._lines[-1] = self._lines[-1][:-1]
                self._line_terminated = False
            else:
                self._lines[-1] += char
                self._line_terminated = False

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
