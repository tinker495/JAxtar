"""Compare full-product Pyrefly diagnostics with the actual PR target revision."""

import argparse
import io
import json
import subprocess
import sys
import tarfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict


class Diagnostic(TypedDict):
    path: str
    line: int
    column: int
    name: str
    description: str
    severity: str
    baselined: bool


def check(root: Path) -> list[Diagnostic]:
    command = [
        sys.executable,
        "-m",
        "pyrefly",
        "check",
        "--config",
        "pyrefly.toml",
        "--python-interpreter-path",
        sys.executable,
        "--output-format=json",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    print(result.stderr, file=sys.stderr, end="")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Pyrefly failed with exit code {result.returncode}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload["errors"], list):
        raise ValueError("Invalid Pyrefly diagnostics payload")
    errors = payload["errors"]
    diagnostics: list[Diagnostic] = []
    for error in errors:
        if not isinstance(error, dict):
            raise ValueError("Invalid Pyrefly diagnostic")
        for field in ("path", "name", "description", "severity"):
            if not isinstance(error[field], str):
                raise ValueError(f"Invalid diagnostic field: {field}")
        if not isinstance(error["line"], int) or not isinstance(error["column"], int):
            raise ValueError("Invalid diagnostic location")
        diagnostics.append(
            Diagnostic(
                path=error["path"],
                line=error["line"],
                column=error["column"],
                name=error["name"],
                description=error["description"].replace(str(root), "<repository>"),
                severity=error["severity"],
                baselined=False,
            )
        )
    if result.returncode == 1 and not errors:
        raise RuntimeError("Pyrefly failed without reporting diagnostics")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="PR target revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}"],
        cwd=root,
        text=True,
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    print(f"Pyrefly {version('pyrefly')} | target={base} | head={head}")
    with TemporaryDirectory(prefix="pyrefly-pr-") as directory:
        target = Path(directory) / "target"
        target.mkdir()
        archive = subprocess.check_output(["git", "archive", base], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(target, filter="data")
        # Both revisions use the PR configuration and the same dependency environment.
        (target / "pyrefly.toml").write_bytes((root / "pyrefly.toml").read_bytes())
        previous = check(target)
        current = check(root)
    # Count identical diagnostics: line movement is harmless, additional copies are new.
    remaining = Counter(
        (error["path"], error["name"], error["description"], error["severity"])
        for error in previous
    )
    for error in current:
        key = (error["path"], error["name"], error["description"], error["severity"])
        if remaining[key] > 0:
            error["baselined"] = True
            remaining[key] -= 1
    new = [error for error in current if not error["baselined"] and error["severity"] == "error"]
    for error in current:
        label = "EXISTING" if error["baselined"] else "NEW"
        print(
            f"{label} {error['path']}:{error['line']}:{error['column']} "
            f"[{error['name']}] {error['description']}"
        )
    print(f"Diagnostics: target={len(previous)}, current={len(current)}, new={len(new)}")
    print(f"New diagnostics gate: {'FAIL' if new else 'PASS'}")
    print(f"Full type check: {'FAIL' if current else 'PASS'}")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
