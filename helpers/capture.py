import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from io import TextIOBase
from pathlib import Path

_CSI_RE = re.compile(r"\x1B\[([0-?]*)[ -/]*([@-~])")
# A CSI sequence split across two write() calls must be carried over, or its
# tail would leak into the log as plain text.
_PARTIAL_CSI_RE = re.compile(r"\x1B(\[[0-?]*[ -/]*)?\Z")


class _LogSanitizer:
    """Fold console control sequences into readable, bounded log lines.

    Interprets the control characters progress renderers actually emit —
    ``\\r``, ``\\b``, erase-line (``ESC[nK``) and cursor-up (``ESC[nA``, which
    rich.Live uses to repaint its frame in place) — so every repaint
    overwrites the previous frame instead of appending it. All other CSI
    sequences are colour/cursor noise and are dropped.

    Only the trailing ``max_pending_lines`` lines stay mutable in memory;
    older lines are returned from ``process``/``drain`` exactly once and never
    revisited. Memory stays O(max_pending_lines) for arbitrarily long runs; a
    live frame taller than the window degrades to appending, never to data
    loss.
    """

    def __init__(self, max_pending_lines: int = 128):
        self._max_pending = max_pending_lines
        self._pending: list[str] = [""]
        self._carry = ""

    def process(self, data: str) -> list[str]:
        """Consume raw console output; return lines that became immutable."""
        if not data:
            return []
        data = self._carry + data
        self._carry = ""
        partial = _PARTIAL_CSI_RE.search(data)
        if partial:
            self._carry = data[partial.start() :]
            data = data[: partial.start()]

        pos = 0
        for match in _CSI_RE.finditer(data):
            self._feed_text(data[pos : match.start()])
            self._apply_csi(match.group(1), match.group(2))
            pos = match.end()
        self._feed_text(data[pos:])

        if len(self._pending) <= self._max_pending:
            return []
        finalized = [line.rstrip() for line in self._pending[: -self._max_pending]]
        del self._pending[: -self._max_pending]
        return finalized

    def drain(self) -> list[str]:
        """Return all remaining lines; trailing blanks left by clears are dropped."""
        # A dangling partial escape sequence at end of output is control noise.
        self._carry = ""
        lines = self._pending
        self._pending = [""]
        while lines and lines[-1] == "":
            lines.pop()
        return [line.rstrip() for line in lines]

    def pending_text(self) -> str:
        """Render the mutable tail (the last line may still be rewritten)."""
        return "\n".join(line.rstrip() for line in self._pending)

    def _feed_text(self, text: str) -> None:
        for char in text:
            if char == "\r":
                self._pending[-1] = ""
            elif char == "\n":
                self._pending.append("")
            elif char == "\b":
                self._pending[-1] = self._pending[-1][:-1]
            else:
                self._pending[-1] += char

    def _apply_csi(self, params: str, final: str) -> None:
        if final == "A":
            # Cursor up: rich.Live moves up over its previous frame to repaint
            # it, so the lines it climbs over are obsolete. Never climb past
            # the pending window — earlier lines are already immutable.
            count = int(params) if params.isdigit() else 1
            keep = max(1, len(self._pending) - count)
            del self._pending[keep:]
        elif final == "K":
            # Erase line. Column position is not tracked, so partial erases
            # (0K/1K) conservatively clear the whole line like 2K.
            self._pending[-1] = ""


class _LogCapture:
    """Incrementally mirror sanitized output into the log file.

    Immutable lines are appended exactly once; only the sanitizer's bounded
    pending tail is rewritten in place. Memory and per-write I/O stay
    O(pending tail) no matter how much the run prints.
    """

    def __init__(self, log_file, offset: int):
        self._file = log_file
        self._tail_offset = offset
        self._sanitizer = _LogSanitizer()

    def write(self, data: str):
        if not data:
            return
        self._flush(self._sanitizer.process(data))

    def flush(self):
        self._flush([])

    def finalize(self):
        self._flush(self._sanitizer.drain())

    def _flush(self, finalized: list[str]) -> None:
        if self._file.closed:
            return
        self._file.seek(self._tail_offset)
        for line in finalized:
            self._file.write(line + "\n")
        self._tail_offset = self._file.tell()
        tail = self._sanitizer.pending_text()
        if tail:
            self._file.write(tail)
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
