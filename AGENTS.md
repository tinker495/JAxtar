<!-- Parent: ../AGENTS.md -->

## Documentation Language

Write all repository documentation in English, including `docs/`, README files,
design documents, glossaries, and `AGENTS.md`. This rule applies to every
subdirectory. Keep user-facing conversation and code-review comments in Korean.

## Code Review Rules

These rules apply to both human and AI reviewers. CI owns mechanical checks such
as formatting and linting; do not include them in code review.

The suggestions below about changing or removing tests and migrating them with
the implementation are review criteria, not permission for an agent to change
tests. Execution permissions are defined under "Test Ownership and Change
Permissions" in this document.

### Review Language and Terminology

- Write review comments and findings in Korean. Keep code identifiers and
  architecture terms in their original English form.

### Structural Change Proposals

- Go beyond minimal edits to diff lines. When a change exposes structural
  problems such as duplicated concept ownership, shotgun surgery, wrappers that
  only delegate, or eroded stage boundaries, explicitly propose deeper changes
  such as moving, merging, or deleting modules and reassigning ownership.
- Optimize for minimum context, not minimum diff: reduce the number of concepts a
  maintainer must understand through deep modules with narrow interfaces. Favor
  changes that solve problems by deleting code over adding it.
- Flag speculative generality immediately as a deletion candidate: interfaces
  with a single implementation, configuration nobody sets, and hand-written
  replacements for standard-library functionality.
- Support structural proposals with evidence of the current maintenance cost
  and present them in severity order.

## Test Ownership and Change Permissions

### Core Principle

Tests are executable product specifications owned by the user, not accessories
to the implementation. The agent implements the product; it does not decide what
qualifies as correct behavior.

Agents may read and run tests, but may not create, modify, or delete them.
Instructions to implement a feature, fix a bug, refactor, or make tests pass do
not authorize changes to tests or acceptance criteria.

Every test must name its author, who must be the user rather than an agent, and
include comments explaining in detail why the test was added.

### Protected Assets

Treat all repository tests and assets that determine product acceptance as
protected, regardless of their path or name. These include:

- Test code, assertions, test cases, and parameters.
- Expected values, fixtures, snapshots, golden data, and reference results.
- Independent validators and oracles, evaluation data, acceptance thresholds,
  and tolerances.
- Test collection, execution, and exclusion settings, and CI validation steps.

Agents may not exempt an asset by labeling it an internal unit test, regression
test, or test enhancement.

### Permitted Actions

- Read and run existing tests and validation assets to analyze requirements and
  failure causes.
- Modify product implementations to meet user requirements.
- Create temporary reproduction code, exploratory scripts, and observation logs
  for debugging.
- Propose needed test cases, counterexamples, and specification ambiguities to
  the user in text.

Temporary validation code belongs only in temporary locations outside formal
test directories. Do not commit it, register it in CI, or incorporate it into
formal tests. Passing temporary checks does not constitute user approval or
satisfaction of product acceptance criteria.

### Prohibited Actions

- Creating, modifying, deleting, moving, or automatically regenerating tests.
- Treating current implementation output as the correct answer when establishing
  expected values or snapshots.
- Weakening assertions, widening tolerances, removing cases, adding skip or xfail
  markers, changing fixtures, or bypassing validation steps to make tests pass.
- Detecting test environments or specific test inputs to hard-code answers, or
  hiding failures, violations, or omissions to pass validation.
- Relaxing this policy or related protections for convenience.

### Test Failures and Specification Conflicts

Investigate implementation defects first when a test fails. Failure alone does
not establish that the test is wrong.

Conversely, a test's existence alone does not establish that it is a correct
product specification approved by the user.

When tests conflict with user requirements, or evidence is insufficient to
determine expected behavior, do not change the tests or preserve incorrect
behavior in the implementation. Report the conflicting requirements,
reproduction input, actual result, and decision needed from the user. Pause only
the changes that depend on that decision.

Even when a new test is needed, do not add it yourself. Propose its input
conditions, behavior to verify, and the basis for its expected result to the
user.

### Completion Report

Distinguish the scope and results of tests actually run, validation that could
not be run, remaining failures, and decisions needed from the user.
