# Repository Guidelines

## Language and Review

Write repository documentation in English, including nested AGENTS.md files.
Use Korean for conversation, review comments, PR bodies, and local reports;
keep code, comments, identifiers, and commit messages in English.

CI owns formatting and linting; omit those findings from code review. Review
structural problems beyond the edited lines: duplicated ownership, shotgun
surgery, delegation-only wrappers, and eroded stage boundaries. Propose moving,
merging, or deleting modules when evidence supports it. Prefer deep modules with
narrow interfaces and fewer concepts. Flag single-implementation interfaces,
unused configuration, and standard-library replacements as deletion candidates.
Present evidence and consequences in severity order. Re-review code added after
the reviewed commit before merging. Review suggestions do not authorize changes
to tests or acceptance assets.

## Test Ownership and Change Permissions

Tests are executable product specifications owned by the user. Agents may read
and run them. Creating, modifying, deleting, moving, or regenerating tests or
acceptance assets requires explicit user authorization for the affected scope
and expected behavior. A feature, bug fix, refactor, or request to make tests pass
does not by itself grant that authorization.

Every test must name its author, who must be the user rather than an agent, and
include comments explaining in detail why the test was added.

Protected assets include assertions, cases, parameters, expected values,
fixtures, snapshots, golden data, reference results, independent validators and
oracles, evaluation data, acceptance thresholds and tolerances, test collection,
execution and exclusion settings, and CI validation steps. Internal unit tests,
regression tests, and test enhancements receive the same protection.

Agents may modify product implementations, propose test inputs and behavior
with a basis for expected results, and create temporary reproduction code or
observation logs outside formal test directories. Keep temporary validation
code out of commits, CI, and formal tests. Passing temporary checks does not
constitute user approval or satisfaction of acceptance criteria.

Do not derive expected values from current implementation output, weaken
assertions, widen tolerances, remove cases, add skip/xfail, change fixtures, or
bypass validation to make tests pass. Do not detect test environments or specific
inputs to hard-code answers or hide failures, violations, or omissions. Do not
relax these protections for convenience.

Investigate implementation defects first when tests fail. A failure alone does
not prove a test is wrong; a test's existence alone does not prove the user
approved it. If requirements conflict or expected behavior is uncertain, report
the reproduction input, actual result, conflicting requirements, and the user's
decision needed. Preserve the tests and pause only dependent changes. Do not
preserve incorrect product behavior to satisfy a conflicting test. Implement
new tests only after the user authorizes their scope and expected behavior.

## Repository Independence and Validation

This repository owns its dependency updates and CI configuration. Related
repositories' latest versions mean their main branches; explicitly select a
branch or commit when needed. A sibling editable checkout is not required.

Run the existing lint/format hooks, Pyrefly new-diagnostics comparison, and CPU
tests on each PR's latest commit as merge-required checks. Compare types against
the actual PR target revision, including a stacked PR's parent. Existing errors
remain visible: a passing new-diagnostics gate is distinct from a full type-check
pass. Product code is the initial typing scope; tests and documentation examples
are deferred.

Performance benchmarks and GPU/kernel/compiler validation run manually when the
user chooses, outside merge-required CI. For a performance experiment, agree on
inputs, settings, and quality/time/memory limits before measurement. Distinguish
main and PR-target baselines by commit SHA. Present search-policy changes and
their solution-quality tradeoffs for separate approval.

Keep reproduction scripts, independent validation tools, traces, diffs, and
reports in Git-excluded local storage; do not commit or upload them. Report tests
actually run, unverified scope, remaining failures, and decisions needed from
the user. Include a short normal CLI verification command and a local record.

## Source Navigation

Consult an available `graft/` graph first: `graft ask "<task>" --source` for
context, `graft grep "<literal>"` for all indexed occurrences, `graft skeleton
<file>` for an API, or `graft callers <symbol> --depth all` for refactor impact.
Read exact source spans when more detail is needed. Use Probe for unindexed
source relationships when available and `rg` as the fallback. Use `rg` directly
for filenames, configuration, documentation, and local output.
