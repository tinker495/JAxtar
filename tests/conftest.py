import importlib
import sys
from pathlib import Path

# Ensure the project root is importable when running `pytest` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

# Some pytest/plugin environments import an unrelated top-level ``cli`` module
# before test collection. Clear that module so imports resolve to JAxtar's
# local ``cli/`` package.
cli_module = sys.modules.get("cli")
cli_paths = [Path(path).resolve() for path in getattr(cli_module, "__path__", [])]
if cli_module is not None and (PROJECT_ROOT / "cli").resolve() not in cli_paths:
    sys.modules.pop("cli", None)
    importlib.import_module("cli")
