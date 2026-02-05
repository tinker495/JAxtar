#!/usr/bin/env python
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path.cwd() / ".env")
_load_dotenv(Path(__file__).resolve().parent / ".env")

# Set default logging level to FATAL (3) to suppress XLA buffer_comparator errors
# This hides "Difference at X: Y, expected Z" messages during quantization
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
