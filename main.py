#!/usr/bin/env python
import os

from cli import cli

# Set default logging level to FATAL (3) to suppress XLA buffer_comparator errors
# This hides "Difference at X: Y, expected Z" messages during quantization
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if __name__ == "__main__":
    cli()
