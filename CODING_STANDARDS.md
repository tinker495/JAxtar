# Coding Standards & Code Review Rules

Authoring and review criteria for `JAxtar`. Keep repository documentation in English; write review comments and findings in Korean while keeping code identifiers and architecture terms in English.

## Documentation and Commenting

- Record *why* non-obvious invariants, algorithmic choices, or edge cases exist; let names, types, and CLI `--help` express *what* the code does.
- Keep chronological refactor notes and benchmark logs in Git-excluded local storage (`runs/`), out of repository `.md` files and comments.
- Preserve valid rationale comments and `# type: ignore` / `# noqa` / `# pragma` directives; delete obsolete comments and dead code.

## Structural Review Criteria

CI owns formatting and linting (`black`, `ruff`, `check_seam_leaks.py`, `check_types.py`); omit mechanical findings from code review. Review suggestions do not authorize changes to protected tests (`AGENTS.md`).

- **Deep modules over shallow wrappers:** Flag duplicated concept ownership, shotgun surgery, delegation-only wrappers, and eroded stage boundaries. Prefer deleting code and narrowing interfaces over adding indirection.
- **Eliminate speculative generality:** Flag single-implementation interfaces, unused configuration options, and standard-library reimplementations for deletion.
- **Seam ownership:** Flag `_`-prefixed or unsanctioned internal imports across `JAxtar` packages or from `PuXle`/`Xtructure`, caller-side mirroring of upstream sentinel/hash/layout internals, and open-coded multi-pass container choreography. Move the invariant or deepened interface into the owning module or upstream repository.
- **Evidence-first ordering:** Present findings in severity order with concrete evidence, and re-review commits added after a reviewed SHA before merging.
