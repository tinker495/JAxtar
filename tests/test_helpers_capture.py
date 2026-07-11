import sys
from pathlib import Path

from helpers.capture import _LogSanitizer, tee_console


def test_log_sanitizer_handles_carriage_returns_and_backspaces():
    sanitizer = _LogSanitizer()
    sanitizer.process("ab\b!")
    assert sanitizer.pending_text() == "a!"

    sanitizer.process("c\rxy")
    assert sanitizer.pending_text() == "xy"


def test_log_sanitizer_strips_ansi_sequences():
    sanitizer = _LogSanitizer()
    sanitizer.process("\x1b[?25l\x1b[32mhello\x1b[0m")
    assert sanitizer.pending_text() == "hello"


def test_log_sanitizer_clear_line_sequence_and_multiline_output():
    sanitizer = _LogSanitizer()
    sanitizer.process("first\nabc\r\x1b[2Knext")
    assert sanitizer.pending_text() == "first\nnext"


def test_log_sanitizer_folds_cursor_up_repaints():
    # rich.Live repaints its frame with \r ESC[2K (ESC[1A ESC[2K)*(h-1); the
    # repainted frame must replace the previous one instead of appending.
    sanitizer = _LogSanitizer()
    sanitizer.process("intro\n")
    sanitizer.process("old1\nold2\nold3")
    sanitizer.process("\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K")
    sanitizer.process("new1\nnew2\nnew3")
    assert sanitizer.pending_text() == "intro\nnew1\nnew2\nnew3"


def test_log_sanitizer_handles_csi_split_across_writes():
    sanitizer = _LogSanitizer()
    sanitizer.process("abc\x1b[")
    sanitizer.process("2Kxyz")
    assert sanitizer.pending_text() == "xyz"


def test_log_sanitizer_promotes_old_lines_and_bounds_memory():
    sanitizer = _LogSanitizer(max_pending_lines=3)
    finalized = sanitizer.process("l1\nl2\nl3\nl4\nl5\nl6")
    assert finalized == ["l1", "l2", "l3"]
    assert sanitizer.pending_text() == "l4\nl5\nl6"

    # Cursor-up never climbs past the pending window into finalized lines.
    sanitizer.process("\x1b[99A")
    assert sanitizer.pending_text() == "l4"

    assert sanitizer.drain() == ["l4"]
    assert sanitizer.pending_text() == ""


def test_log_sanitizer_drain_trims_trailing_blank_lines():
    sanitizer = _LogSanitizer()
    sanitizer.process("kept\n\nlast\n\r\x1b[2K")
    assert sanitizer.drain() == ["kept", "", "last"]


def test_tee_console_captures_stdout_and_stderr_into_log(tmp_path: Path):
    log_path = tmp_path / "run.log"

    with tee_console(log_path):
        print("hello")
        print("stderr-line", file=sys.stderr)

    content = log_path.read_text()

    assert "===== Run started" in content
    assert "===== Run ended" in content
    assert "hello" in content
    assert "stderr-line" in content
    assert "\x1b[" not in content


def test_tee_console_log_size_stays_bounded_under_repaints(tmp_path: Path):
    log_path = tmp_path / "run.log"

    with tee_console(log_path):
        print("progress start")
        sys.stdout.write("frame initial\nline two of frame initial")
        for i in range(500):
            # Simulated live repaint: climb over the 2-line frame and redraw.
            sys.stdout.write("\r\x1b[2K\x1b[1A\x1b[2K")
            sys.stdout.write(f"frame {i}\nline two of frame {i}")
        sys.stdout.write("\n")
        print("progress end")

    content = log_path.read_text()
    assert "progress start" in content
    assert "frame 499" in content
    assert "progress end" in content
    assert "frame 42" not in content  # folded away, not appended
