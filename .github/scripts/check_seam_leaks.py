#!/usr/bin/env python3
"""AST-based seam-leakage guard for standalone JAxtar CI."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SOURCE_ROOTS = (
    "JAxtar",
    "cli",
    "config",
    "helpers",
    "heuristic",
    "qfunction",
    "train_util",
    "neural_util",
)

UPSTREAM_PACKAGES = {
    "xtructure": "Xtructure",
    "puxle": "PuXle",
}

SANCTIONED_SEAMS: dict[str, frozenset[str]] = {
    "xtructure": frozenset(
        {
            "xtructure",
            "xtructure.numpy",
            "xtructure.xtructure_numpy",
            "xtructure.io",
            "xtructure.core.layout",
            "xtructure.core.display",
            "xtructure.core.packing",
        }
    ),
    "puxle": frozenset(
        {
            "puxle",
            "puxle.benchmark",
            "puxle.world_model",
            "puxle.core.trajectory",
        }
    ),
}

UNAPPROVED_PACKAGE_CHILDREN: dict[str, frozenset[str]] = {
    "xtructure": frozenset({"core", "hashtable", "bgpq", "queue", "stack"}),
    "puxle": frozenset({"core", "puzzles", "render", "utils", "cli"}),
}

EXCLUDED_PARTS = {
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
    "scripts",
}


@dataclass(frozen=True)
class SeamViolation:
    path: str
    lineno: int
    kind: str
    raw_import: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "lineno": self.lineno,
            "kind": self.kind,
            "raw_import": self.raw_import,
            "reason": self.reason,
        }


def iter_source_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for top_pkg in SOURCE_ROOTS:
        base = root / top_pkg
        candidates: Iterable[Path] = base.rglob("*.py") if base.is_dir() else ()
        for path in candidates:
            rel = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
                continue
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                continue
            if path not in seen:
                files.append((top_pkg, path))
                seen.add(path)
    return sorted(files, key=lambda item: item[1].relative_to(root).as_posix())


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _has_private_segment(dotted_name: str) -> bool:
    return any(_is_private_name(part) for part in dotted_name.split(".") if part)


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def check_file_for_leaks(root: Path, path: Path, source_top: str) -> list[SeamViolation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    rel_path = _display_path(root, path)
    violations: list[SeamViolation] = []
    intra_pkgs = frozenset(SOURCE_ROOTS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw = (ast.get_source_segment(source, node) or "").strip()
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in UPSTREAM_PACKAGES:
                    target_repo = UPSTREAM_PACKAGES[top]
                    if _has_private_segment(alias.name):
                        violations.append(
                            SeamViolation(
                                path=rel_path,
                                lineno=node.lineno,
                                kind="private-cross-repo-import",
                                raw_import=raw,
                                reason=(
                                    f"Private cross-repo import `{alias.name}`; "
                                    f"expose or deepen a public seam in `{target_repo}`."
                                ),
                            )
                        )
                    elif alias.name not in SANCTIONED_SEAMS[top]:
                        violations.append(
                            SeamViolation(
                                path=rel_path,
                                lineno=node.lineno,
                                kind="internal-cross-repo-seam",
                                raw_import=raw,
                                reason=(
                                    f"Internal cross-repo import `{alias.name}` from "
                                    f"`{target_repo}`; use a sanctioned seam."
                                ),
                            )
                        )
                elif top in intra_pkgs and top != source_top and _has_private_segment(alias.name):
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="private-cross-package-import",
                            raw_import=raw,
                            reason=(
                                f"Private cross-package import `{alias.name}` from `{top}`; "
                                "promote the helper or keep it behind a public seam."
                            ),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            mod = node.module
            top = mod.split(".")[0]
            raw = (ast.get_source_segment(source, node) or "").strip()
            has_private_alias = any(
                _is_private_name(alias.name) for alias in node.names if alias.name != "*"
            )

            if top in UPSTREAM_PACKAGES:
                target_repo = UPSTREAM_PACKAGES[top]
                if _has_private_segment(mod) or has_private_alias:
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="private-cross-repo-import",
                            raw_import=raw,
                            reason=(
                                f"Private symbol or module imported from `{target_repo}`; "
                                f"fix the invariant or deepen the seam in `{target_repo}`."
                            ),
                        )
                    )
                    continue
                imports_internal_child = mod == top and any(
                    alias.name in UNAPPROVED_PACKAGE_CHILDREN[top] for alias in node.names
                )
                if mod not in SANCTIONED_SEAMS[top] or imports_internal_child:
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="internal-cross-repo-seam",
                            raw_import=raw,
                            reason=(
                                f"Unsanctioned internal seam `{mod}` imported from "
                                f"`{target_repo}`; use a sanctioned seam or update CONTEXT.md."
                            ),
                        )
                    )
            elif top in intra_pkgs and top != source_top:
                if _has_private_segment(mod) or has_private_alias:
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="private-cross-package-import",
                            raw_import=raw,
                            reason=(
                                f"Private symbol or module imported across packages from `{top}`; "
                                "promote the helper to a public seam."
                            ),
                        )
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument(
        "--check-file",
        type=Path,
        default=None,
        help="Check a single file path as part of `JAxtar`",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations: list[SeamViolation] = []
    if args.check_file is not None:
        files_checked = 1
        violations.extend(check_file_for_leaks(root, args.check_file.resolve(), "JAxtar"))
    else:
        files = iter_source_files(root)
        files_checked = len(files)
        for source_top, path in files:
            violations.extend(check_file_for_leaks(root, path, source_top))

    if args.json:
        print(
            json.dumps(
                {
                    "failed": bool(violations),
                    "files_checked": files_checked,
                    "violation_count": len(violations),
                    "violations": [v.to_dict() for v in violations],
                },
                indent=2,
            )
        )
    else:
        print(f"Files checked: {files_checked}, Seam violations: {len(violations)}")
        for v in violations:
            print(f"  {v.path}:{v.lineno} [{v.kind}] {v.raw_import}\n    -> {v.reason}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
