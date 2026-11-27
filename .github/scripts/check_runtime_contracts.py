#!/usr/bin/env python3
"""Verify standalone dependency, import-symbol, lazy-export, and CLI contracts."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

SOURCE_ROOTS: tuple[str, ...] = (
    "JAxtar",
    "cli",
    "config",
    "helpers",
    "heuristic",
    "qfunction",
    "train_util",
    "neural_util",
)
EXPECTED_GIT_DEPS: dict[str, str] = {
    "puxle": "git+https://github.com/tinker495/PuXle.git@main",
    "xtructure": "git+https://github.com/tinker495/xtructure.git@main",
}
EXTRA_CLI_HELP_INVOCATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("JAxtar.entrypoints:cli", ("test", "--help")),
    ("JAxtar.entrypoints:cli", ("benchmark", "--help")),
    ("JAxtar.entrypoints:cli", ("distance-train", "--help")),
)
EXCLUDED_PARTS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        ".git",
        "build",
        "dist",
        "runs",
        "docs",
        "tests",
    }
)


def _normalize_dep(dep: str) -> str:
    return "".join(dep.split()).replace('"', "'")


def check_pyproject_contracts(root: Path) -> list[str]:
    errors: list[str] = []
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest.get("project", {})
    deps: list[str] = list(project.get("dependencies", ()))
    normalized = [_normalize_dep(dep) for dep in deps]

    if not any(dep.startswith("jax[cuda]>=0.11.2;sys_platform=='linux'") for dep in normalized):
        errors.append(
            "pyproject.toml must declare `jax[cuda]>=0.11.2; sys_platform == 'linux'` "
            "in [project].dependencies so default Linux installs include CUDA."
        )

    optional = project.get("optional-dependencies", {})
    for forbidden_extra in ("cuda", "gpu", "benchmark", "logging", "train"):
        if forbidden_extra in optional:
            errors.append(
                f"pyproject.toml must not define optional extra `{forbidden_extra}`; "
                "fold runtime dependencies into [project].dependencies."
            )

    uv_sources = manifest.get("tool", {}).get("uv", {}).get("sources", {})
    for pkg in EXPECTED_GIT_DEPS:
        if pkg in uv_sources:
            errors.append(
                f"pyproject.toml [tool.uv.sources] must not override `{pkg}` in standalone repos."
            )

    for pkg, expected_url in EXPECTED_GIT_DEPS.items():
        matching = [dep for dep in deps if dep.split("@", 1)[0].strip().lower() == pkg.lower()]
        if len(matching) != 1:
            errors.append(
                f"pyproject.toml must declare `{pkg} @ {expected_url}` in [project].dependencies."
            )
            continue
        _, _, raw_target = matching[0].partition("@")
        target = raw_target.strip()
        if target.lower() != expected_url.lower():
            errors.append(
                f"pyproject.toml dependency `{matching[0]}` must track `@main` "
                f"(`{pkg} @ {expected_url}`)."
            )

    return errors


def check_lockfile_main_commits(root: Path, offline: bool) -> list[str]:
    errors: list[str] = []
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        return ["uv.lock is missing."]

    lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {
        pkg["name"].lower(): pkg
        for pkg in lock_data.get("package", ())
        if isinstance(pkg, dict) and "name" in pkg
    }

    for pkg_name in EXPECTED_GIT_DEPS:
        pkg = packages.get(pkg_name.lower())
        if pkg is None:
            errors.append(f"uv.lock is missing package `{pkg_name}`.")
            continue
        git_src = pkg.get("source", {}).get("git")
        if not isinstance(git_src, str) or "#" not in git_src:
            errors.append(
                f"uv.lock package `{pkg_name}` must resolve from Git (`source.git`), got {git_src!r}."
            )
            continue
        if "?rev=main#" not in git_src:
            errors.append(
                f"uv.lock package `{pkg_name}` must resolve `?rev=main#...`, got `{git_src}`."
            )
            continue
        repo_part, locked_sha = git_src.split("#", 1)
        repo_url = repo_part.split("?", 1)[0]
        if offline:
            continue
        proc = subprocess.run(
            ["git", "ls-remote", repo_url, "refs/heads/main"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            errors.append(f"Failed to query `{repo_url}` refs/heads/main: {proc.stderr.strip()}")
            continue
        remote_sha = proc.stdout.split()[0].strip()
        if locked_sha != remote_sha:
            errors.append(
                f"uv.lock pins `{pkg_name}` to {locked_sha[:12]}, "
                f"but `{repo_url}` main is {remote_sha[:12]}. "
                f"Refresh with: uv lock --upgrade-package {pkg_name}"
            )

    return errors


def _extract_string_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    items: list[str] = []
    for elt in node.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            return None
        items.append(elt.value)
    return items


def _is_type_checking_guard(test: ast.AST) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
        return test.value.id == "typing" and test.attr == "TYPE_CHECKING"
    return False


def _type_checking_node_ids(tree: ast.AST) -> set[int]:
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                guarded.add(id(child))
    return guarded


def _check_init_exports_parity(root: Path, init_path: Path) -> list[str]:
    rel = init_path.relative_to(root).as_posix()
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    dunder_all: list[str] | None = None
    exports_keys: set[str] | None = None

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    dunder_all = _extract_string_list(stmt.value)
                elif (
                    isinstance(target, ast.Name)
                    and target.id == "_EXPORTS"
                    and isinstance(stmt.value, ast.Dict)
                ):
                    exports_keys = {
                        key.value
                        for key in stmt.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "_EXPORTS"
            and isinstance(stmt.value, ast.Dict)
        ):
            exports_keys = {
                key.value
                for key in stmt.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }

    if exports_keys is not None:
        if dunder_all is None:
            return [f"{rel} defines `_EXPORTS` without a literal `__all__` list."]
        all_set = set(dunder_all)
        if exports_keys != all_set:
            missing = sorted(all_set - exports_keys)
            extra = sorted(exports_keys - all_set)
            return [
                f"{rel} `_EXPORTS` keys drifted from `__all__` "
                f"(missing={missing}, extra={extra})."
            ]
    return []


def _iter_source_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for source_root in SOURCE_ROOTS:
        base = root / source_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
                continue
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                continue
            files.append(path)
    return files


def _check_file_imports_resolve(
    root: Path, py_path: Path, checked_roots: frozenset[str]
) -> list[str]:
    errors: list[str] = []
    rel = py_path.relative_to(root)
    rel_posix = rel.as_posix()
    package = ".".join(rel.parts[:-1])
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    tc_ids = _type_checking_node_ids(tree)

    for node in ast.walk(tree):
        if id(node) in tc_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in checked_roots:
                    continue
                try:
                    importlib.import_module(alias.name)
                except Exception as exc:
                    errors.append(
                        f"{rel_posix}:{node.lineno} cannot import `{alias.name}` "
                        f"({type(exc).__name__}: {exc})"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if node.level:
                if not package:
                    continue
                try:
                    mod_name = importlib.util.resolve_name("." * node.level + mod_name, package)
                except Exception as exc:
                    errors.append(
                        f"{rel_posix}:{node.lineno} invalid relative import "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue
            if not mod_name or mod_name.split(".")[0] not in checked_roots:
                continue
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                errors.append(
                    f"{rel_posix}:{node.lineno} cannot import module `{mod_name}` "
                    f"({type(exc).__name__}: {exc})"
                )
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                try:
                    getattr(mod, alias.name)
                except AttributeError:
                    try:
                        importlib.import_module(f"{mod_name}.{alias.name}")
                    except Exception as exc:
                        errors.append(
                            f"{rel_posix}:{node.lineno} missing symbol "
                            f"`{mod_name}.{alias.name}` ({type(exc).__name__}: {exc})"
                        )
                except Exception as exc:
                    errors.append(
                        f"{rel_posix}:{node.lineno} failed resolving "
                        f"`{mod_name}.{alias.name}` ({type(exc).__name__}: {exc})"
                    )
    return errors


def _invoke_cli_help(label: str, target: str, argv_tail: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    mod_name, attr_name = target.split(":", 1)
    try:
        fn = getattr(importlib.import_module(mod_name), attr_name)
    except Exception as exc:
        return [f"Failed to load script `{label}` ({target}): {type(exc).__name__}: {exc}"]
    old_argv = sys.argv[:]
    buf = io.StringIO()
    try:
        sys.argv = [label, *argv_tail]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                fn()
            except SystemExit as exc:
                if exc.code not in (0, None):
                    errors.append(f"Script `{label} {' '.join(argv_tail)}` exited with {exc.code}.")
            else:
                errors.append(
                    f"Script `{label} {' '.join(argv_tail)}` returned without SystemExit(0)."
                )
        if not buf.getvalue().strip():
            errors.append(f"Script `{label} {' '.join(argv_tail)}` produced empty output.")
    except Exception as exc:
        errors.append(f"Script `{label} {' '.join(argv_tail)}` raised {type(exc).__name__}: {exc}")
    finally:
        sys.argv = old_argv
    return errors


def check_runtime_imports_and_cli(root: Path) -> list[str]:
    errors: list[str] = []
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

    source_files = _iter_source_py_files(root)
    init_files = [path for path in source_files if path.name == "__init__.py"]
    for init_path in init_files:
        errors.extend(_check_init_exports_parity(root, init_path))

    checked_roots = frozenset((*SOURCE_ROOTS, *EXPECTED_GIT_DEPS.keys()))
    root_str = str(root.resolve())
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True

    try:
        for init_path in init_files:
            rel = init_path.relative_to(root)
            mod_name = ".".join(rel.parts[:-1])
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                errors.append(f"Failed to import `{mod_name}`: {type(exc).__name__}: {exc}")
                continue
            for export_name in getattr(mod, "__all__", ()):
                try:
                    getattr(mod, export_name)
                except Exception as exc:
                    errors.append(
                        f"Failed to resolve `{mod_name}.{export_name}`: "
                        f"{type(exc).__name__}: {exc}"
                    )

        for py_path in source_files:
            errors.extend(_check_file_imports_resolve(root, py_path, checked_roots))

        manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        scripts: dict[str, str] = manifest.get("project", {}).get("scripts", {})
        for script_name, target in sorted(scripts.items()):
            errors.extend(_invoke_cli_help(script_name, target, ("--help",)))
        for target, argv_tail in EXTRA_CLI_HELP_INVOCATIONS:
            errors.extend(_invoke_cli_help("jaxtar", target, argv_tail))
    finally:
        if inserted and root_str in sys.path:
            sys.path.remove(root_str)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root directory",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network `git ls-remote` check against upstream main branches",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    errors.extend(check_pyproject_contracts(root))
    errors.extend(check_lockfile_main_commits(root, offline=args.offline))
    errors.extend(check_runtime_imports_and_cli(root))

    if errors:
        print(f"Runtime contract violations ({len(errors)}):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Runtime contracts OK ({root.name}: {len(_iter_source_py_files(root))} files verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
