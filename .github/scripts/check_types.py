"""Compare full-product Pyrefly diagnostics with the actual PR target revision."""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
from collections import Counter
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

EDITABLE_DEPENDENCY_PACKAGES: tuple[str, ...] = ("xtructure", "puxle")


class Diagnostic(TypedDict):
    path: str
    line: int
    column: int
    name: str
    description: str
    severity: str
    baselined: bool


def _editable_search_paths(interpreter: Path) -> list[Path]:
    if interpreter.resolve() != Path(sys.executable).resolve():
        return []
    paths: list[Path] = []
    for package in EDITABLE_DEPENDENCY_PACKAGES:
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        pkg_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        if "site-packages" in pkg_dir.parts or "dist-packages" in pkg_dir.parts:
            continue
        parent = pkg_dir.parent
        if parent not in paths:
            paths.append(parent)
    return paths


def check(root: Path, interpreter: Path) -> list[Diagnostic]:
    search_paths = _editable_search_paths(interpreter)
    command = [
        sys.executable,
        "-m",
        "pyrefly",
        "check",
        "--config",
        "pyrefly.toml",
        "--python-interpreter-path",
        str(interpreter),
        "--output-format=json",
    ]
    for search_path in search_paths:
        command.extend(["--search-path", str(search_path)])
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
        description = (
            error["description"]
            .replace(str(interpreter.parent.parent), "<python-environment>")
            .replace(str(root), "<repository>")
        )
        for search_path in search_paths:
            description = description.replace(str(search_path), "<python-environment>")
        diagnostics.append(
            Diagnostic(
                path=error["path"],
                line=error["line"],
                column=error["column"],
                name=error["name"],
                description=description,
                severity=error["severity"],
                baselined=False,
            )
        )
    if result.returncode == 1 and not errors:
        raise RuntimeError("Pyrefly failed without reporting diagnostics")
    return diagnostics


def target_interpreter(target: Path) -> Path:
    manifest = tomllib.loads((target / "pyproject.toml").read_text())
    command = ["uv", "sync", "--project", str(target), "--python", sys.executable]
    if (target / "uv.lock").is_file():
        command.append("--locked")
        print("Target dependencies: using the target revision's uv.lock", flush=True)
    else:
        print("Target has no uv.lock; resolving its dependency declarations", flush=True)
    if "dependency-groups" in manifest and "dev" in manifest["dependency-groups"]:
        command.extend(["--group", "dev"])
    elif (
        "optional-dependencies" in manifest["project"]
        and "dev" in manifest["project"]["optional-dependencies"]
    ):
        command.extend(["--extra", "dev"])
    subprocess.run(
        command,
        cwd=target,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(target / ".venv")},
        check=True,
    )
    return target / ".venv" / "bin" / "python"


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
        # Keep the checker/config fixed while loading each revision's dependencies.
        (target / "pyrefly.toml").write_bytes((root / "pyrefly.toml").read_bytes())
        previous = check(target, target_interpreter(target))
        current = check(root, Path(sys.executable))
    renamed = subprocess.check_output(
        ["git", "diff", "--name-status", "-z", "--find-renames", "--diff-filter=R", base, head],
        cwd=root,
        text=True,
    ).split("\0")[:-1]
    paths = dict(zip(renamed[1::3], renamed[2::3], strict=True))
    for error in previous:
        if error["path"] in paths:
            error["path"] = paths[error["path"]]
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
