#!/usr/bin/env python
import os

# Set default logging level to FATAL (3) to suppress XLA buffer_comparator errors
# This hides "Difference at X: Y, expected Z" messages during quantization
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
