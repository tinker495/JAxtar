# JAxtar Terminal Design

## Overview and Layer Ownership

JAxtar's terminal output is dense, results-first, and semantically color-coded so "did it solve, how fast, how many states" reads at a glance. Every human-facing screen of `jaxtar`, the per-algorithm commands (`astar`, `qstar`, ...), `benchmark`, and `distance_train` renders with `rich`. Three layers compose each screen:

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Puzzle content | PuXle `State.str()` / `SolveConfig.__str__` | Boards, cubes, and grids as raw ANSI strings (heavy `┏━┓` frames, colored glyphs). |
| Batches | Xtructure `BatchedRenderer` | `Batched <Class>` + `shape: (...)` + items separated by ` \| ` (plain text). |
| Chrome | JAxtar | Panels, tables, trees, progress bars, and rules framing the content above. |

JAxtar owns only the chrome. Puzzle strings must enter `rich` via `Text.from_ansi(...)`, never raw ANSI inside markup, because markup interpolation causes `rich` to miscount escape-code widths and parse `[` as markup tags. Print external strings (exception messages, file paths, user input) with `markup=False` or `rich.markup.escape`. Matplotlib figures (`helpers/plots/`), GIF/PNG frames (`save_solution_animation_and_frames`), and log files are out of scope.

## Colors

Always name colors with `rich` style names (`blue`, `bold green`, ...), never hex or truecolor, so the user's terminal theme stays in control:

| Role | `rich` style | Meaning |
| --- | --- | --- |
| Primary | `blue` | Puzzle state views, path arrows (`→`/`↓`), step borders, heuristic distances, progress description |
| Success | `green` | Positive outcomes, solved borders/titles, config panel title, progress bar fill, `f:` step cost |
| Error | `red` | Failures and errors; `g:` step cost |
| Warning | `yellow` | Notices needing attention, early stopping, skipped runs, config-diff borders |
| Secondary | `cyan` | Metric/column names, evaluation titles, run rules, saved file paths |
| Tertiary | `magenta` | Config keys, table headers, Q-values |
| Guide | `bright_blue` | Config tree guides |
| Muted | `dim` | Low-priority borders/columns, `...` elisions |
| Neutral | `white` | Separators (` \| `, `: `) |

Semantic rules:
- Green and red indicate outcome; the sole exception is the step-cost triad on solution paths (`g` red, `h` blue, `f` green).
- Reuse an existing role above instead of introducing a new color to chrome.

## Typography, Shapes, and Layout

- **Typeface & markup**: Terminal monospace in regular (`body`) or bold (`title`) only; no underline, blink, background fills, shadows, or reverse video. Inline `rich` markup uses explicit closing tags (`[bold blue]Start State[/bold blue]`), never the short `[/]` form.
- **Casing & vocabulary**: Title Case for titles and metric labels (`Search Result`, `Avg. Path Cost`); sentence case for messages. Fixed status vocabulary: `"Solution Found"` / `"No Solution Found"`, `"Yes"` / `"No"`, `"Solved!"`, and `"N/A"` for absent values.
- **Content-hugging layout & contrast**: Panels and tables use `expand=False` (except the two-column config layout), nest at most two borders deep, and keep `rich`'s rounded box style (`╭─╮ ╰─╯`) so JAxtar chrome stays visually distinct from PuXle's heavy square frames (`┏━┓`).

## Progress and Log Capture Constraints

Use `helpers.rich_progress` (`tqdm`/`trange`) rather than the upstream `tqdm` package, and wrap any interleaved console output printed during a live progress bar inside `bar.pause()`. `_LogSanitizer` (`helpers/capture.py`) folds only `\r`, `\b`, erase-line (`ESC[nK`), and cursor-up (`ESC[nA`) into `console.log`; never use `Live(screen=True)` or absolute cursor positioning, which cannot be captured cleanly.

## Known Gaps

- Search tables use plain `[bold]` titles with default headers, whereas evaluation tables use `[bold cyan]` titles and `bold magenta` headers.
- Status lines interpolate run directories into markup without escaping (`cli/evaluation_runner.py`, `cli/benchmark_commands.py`, `cli/comparison_generator.py`).
