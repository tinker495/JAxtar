import sys
from pathlib import Path

from helpers.capture import _LogSanitizer, tee_console


def test_log_sanitizer_handles_carriage_returns_and_backspaces():
    sanitizer = _LogSanitizer()
    sanitizer.process("ab\b!")
    assert sanitizer.render() == "a!"

    sanitizer.process("c\rxy")
    assert sanitizer.render() == "xy"


def test_log_sanitizer_strips_ansi_sequences():
    sanitizer = _LogSanitizer()
    sanitizer.process("\x1b[?25l\x1b[32mhello\x1b[0m")
    assert sanitizer.render() == "hello"


def test_log_sanitizer_clear_line_sequence_and_multiline_output():
    sanitizer = _LogSanitizer()
    sanitizer.process("first\nabc\r\x1b[2Knext")
    assert sanitizer.render() == "first\nnext"


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
