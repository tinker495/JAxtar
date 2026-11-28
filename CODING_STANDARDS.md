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
- **Evidence-first ordering:** Present findings in severity order with concrete evidence, and re-review commits added after a reviewed SHA before merging.

Use sanctioned public imports across packages and repositories. Run `.github/scripts/check_seam_leaks.py` for mechanical import violations.

## Cross-repository ownership

Trace each changed boundary operation from its caller to the documented public API, including helpers assembled entirely from public calls. Ask which repository must change if the provider changes its representation while preserving the public contract. A caller that must change with the representation has taken ownership of a provider invariant.

JAxtar selects search candidates, costs and priorities. PuXle owns puzzle transitions and state encodings. Xtructure owns hash probing, padding and duplicate-write resolution. Passing candidate slots, values and a mask to `xnp.update_on_condition` keeps these responsibilities separate. Constructing the table sentinel and electing a first writer in search code makes the caller depend on Xtructure's representation. Implement that operation in Xtructure, using its existing public API when it already expresses the intent.

In the review or completion summary, name the caller, the public API and the repository responsible for the invariant. For a violation, cite the offending call and the operation the owner should provide. Require that correction before approval, even when the import guard passes or a benchmark improves. For a diff with no affected boundary, state that once. Recheck changed calls on subsequent revisions.
