"""Shared tyro presentation and CLI error boundary."""

import sys
from collections.abc import Callable, Sequence
from typing import Any

import tyro
from rich.console import Console

CLI_CONFIG = (tyro.conf.OmitArgPrefixes, tyro.conf.FlagCreatePairsOff)


def run_cli(parse: Callable[..., Any], args: Sequence[str] | None = None, **kwargs: Any) -> Any:
    # JAxtar has always used -h for --hard; help is requested with --help.
    args = ["--hard" if arg == "-h" else arg for arg in (sys.argv[1:] if args is None else args)]
    try:
        return parse(args=args, config=CLI_CONFIG, **kwargs)
    except ValueError as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False)
        raise SystemExit(2) from exc
