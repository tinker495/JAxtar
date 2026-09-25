---
version: alpha
name: JAxtar Terminal
description: Visual design of the rich-rendered terminal output of the JAxtar search, benchmark and training CLIs.
omitted:
  - section: rounded
    reason: Terminal output has no pixel radii; corner shape comes from rich box styles (see Shapes).
colors:
  primary: "#000080"
  secondary: "#008080"
  tertiary: "#800080"
  success: "#008000"
  error: "#800000"
  warning: "#808000"
  guide: "#0000ff"
  muted: "#808080"
  neutral: "#c0c0c0"
typography:
  title:
    fontFamily: monospace
    fontWeight: 700
  body:
    fontFamily: monospace
    fontWeight: 400
spacing:
  cell: 1
  panel-padding-x: 1
  grid-gap: 1
  arrow-gutter: 3
  border-allowance: 4
components:
  state-heading:
    textColor: "{colors.primary}"
    typography: "{typography.title}"
  path-arrow:
    textColor: "{colors.primary}"
    typography: "{typography.title}"
    width: "{spacing.arrow-gutter}"
  step-panel-border:
    textColor: "{colors.primary}"
    padding: "{spacing.panel-padding-x}"
  final-step-panel-border:
    textColor: "{colors.success}"
    padding: "{spacing.panel-padding-x}"
  cost-g:
    textColor: "{colors.error}"
    typography: "{typography.title}"
  cost-h:
    textColor: "{colors.primary}"
    typography: "{typography.title}"
  cost-f:
    textColor: "{colors.success}"
    typography: "{typography.title}"
  status-success:
    textColor: "{colors.success}"
    typography: "{typography.title}"
  status-failure:
    textColor: "{colors.error}"
    typography: "{typography.title}"
  status-warning:
    textColor: "{colors.warning}"
    typography: "{typography.title}"
  metric-label:
    textColor: "{colors.secondary}"
    typography: "{typography.body}"
  metric-value:
    typography: "{typography.body}"
  eval-table-title:
    textColor: "{colors.secondary}"
    typography: "{typography.title}"
  eval-table-header:
    textColor: "{colors.tertiary}"
    typography: "{typography.title}"
  eval-row-label:
    textColor: "{colors.muted}"
    typography: "{typography.body}"
  config-panel-title:
    textColor: "{colors.success}"
    typography: "{typography.title}"
  config-panel-border:
    textColor: "{colors.muted}"
  config-key:
    textColor: "{colors.tertiary}"
    typography: "{typography.title}"
  config-tree-guide:
    textColor: "{colors.guide}"
  summary-panel-border:
    textColor: "{colors.success}"
  diff-panel-border:
    textColor: "{colors.warning}"
  progress-description:
    textColor: "{colors.primary}"
    typography: "{typography.title}"
  progress-metrics:
    textColor: "{colors.secondary}"
  progress-bar:
    textColor: "{colors.success}"
  rule-run:
    textColor: "{colors.secondary}"
    typography: "{typography.title}"
  rule-complete:
    textColor: "{colors.success}"
    typography: "{typography.title}"
  heuristic-distance:
    textColor: "{colors.primary}"
  q-action:
    textColor: "{colors.success}"
  q-value:
    textColor: "{colors.tertiary}"
  q-separator:
    textColor: "{colors.neutral}"
  elision:
    textColor: "{colors.muted}"
---

# JAxtar Terminal Design

## Overview

JAxtar's terminal output reads like a lab instrument: dense, results-first, and color-coded so that
"did it solve, how fast, how many states" can be read at a glance. Color carries meaning, never
decoration. Every human-facing render of the `jaxtar`, per-algorithm (`astar`, `qstar`, ...),
`benchmark` and `distance_train` commands is built with [rich](https://github.com/Textualize/rich)
(`rich` in `pyproject.toml`; rich 15 in the workspace venv).

Three layers compose each screen:

| Layer | Owner | What it draws |
| --- | --- | --- |
| Puzzle content | PuXle `State.str()` / `SolveConfig.__str__` | Boards, cubes, grids as raw ANSI strings (heavy `┏━┓` frames, colored glyphs). |
| Batches | Xtructure `BatchedRenderer` | `Batched <Class>` + `shape: (...)` + items side by side, separated by ` \| ` (plain text). |
| Chrome | JAxtar (this repo) | Panels, tables, trees, progress bars and rules around the content above. |

JAxtar owns only the chrome. Puzzle strings enter rich through `Text.from_ansi(...)`, so their
colors survive and rich measures their width correctly.

Source of truth:

| Area | Code |
| --- | --- |
| Search screens | `helpers/visualization.py` (`build_seed_setup_panel`, `build_result_table`, `build_summary_table`, `build_solution_path_panel`, `build_vmapped_*`), `cli/search_runner.py` |
| Config panel | `helpers/config_printer.py` `print_config` |
| Evaluation / comparison | `helpers/summaries.py`, `cli/evaluation_runner.py`, `cli/comparison_generator.py`, `cli/benchmark_commands.py` |
| Progress | `helpers/rich_progress.py` `RichProgressBar`, `tqdm`, `trange` |
| Number and distance text | `helpers/formatting.py` |
| Error boundary | `cli/runtime.py` `run_cli` |
| Log capture of rich output | `helpers/capture.py` `_LogSanitizer`, `tee_console` |

Out of scope: matplotlib figures (`helpers/plots/`), solution GIF/PNG frames
(`save_solution_animation_and_frames`, which draws PuXle `img()` output) and log files.

## Colors

Code always names colors with rich style names (`blue`, `bold green`, ...), never hex or
truecolor, so that the user's terminal theme stays in control. The hex values in the tokens are
the colors rich itself uses for those names in its default export theme
(`rich.terminal_theme.DEFAULT_TERMINAL_THEME`); a real terminal remaps them.

| Token | rich style | Role | Used by |
| --- | --- | --- | --- |
| `primary` (#000080) | `blue` | The puzzle being searched | "Start State", "Goal", "Seed N", `→`/`↓` path arrows, intermediate step borders, `h:` values, heuristic `Dist`, progress description |
| `success` (#008000) | `green` | Positive outcome, completion | "Solution Found", "Solved!", "Yes", final step border, "Solution Path"/"Solution State" titles, config panel title, evaluation summary border, progress bar fill, `f:` values, "Sweep Complete" rule |
| `error` (#800000) | `red` | Failure | "No Solution Found", "No", `Error:` lines; also `g:` values |
| `warning` (#808000) | `yellow` | Needs attention | profiling notice, "Early stopping triggered!", invalid run directories, runs skipped by compare for missing files, config-differences panel border |
| `secondary` (#008080) | `cyan` | Names of things | "Metric"/"Seed" columns, progress metrics, evaluation table titles, sweep run rules, workload-signature title, saved file paths |
| `tertiary` (#800080) | `magenta` | Keys and headers | config keys, evaluation table headers, Q-values |
| `guide` (#0000ff) | `bright_blue` | Structure lines | config tree guides |
| `muted` (#808080) | `dim` | Low-priority chrome | config panel border, evaluation "Metric"/"Run" columns, ` ... ` elision, skipped-column notices (`dim red`) |
| `neutral` (#c0c0c0) | `white` | Separators | ` \| ` and `: ` in Q-value strips |

Semantic rules:

- Green and red mean outcome. The single exception is the cost triad on solution steps:
  `g` red, `h` blue, `f` green (`build_solution_path_panel`).
- Blue belongs to the puzzle view; cyan names things; magenta marks keys; yellow asks for
  attention. A new render picks one of these roles instead of adding a color.

Colors that come from rich defaults rather than from JAxtar code: table titles are italic
(`table.title`), table headers bold (`table.header`), rules draw a `bright_green` line
(`rule.line`), progress percentage is magenta, elapsed time yellow, remaining time cyan, the
spinner green, and `Pretty` values in the config tree use rich's repr highlighter (numbers bold
cyan, strings green, `True` italic bright green, `None` italic magenta).

## Typography

One face: the terminal's monospace font. Two weights:

- `title` (bold): panel and table titles, status words, config keys, cost labels, arrows, rules.
- `body` (regular): values, table cells, metrics.

No underline, blink or reverse video. Italic appears only through rich's default table-title
style.

Markup is inline rich markup with explicit closing tags that repeat the style name
(`[bold blue]Start State[/bold blue]`), not the short `[/]` form.

Casing: Title Case for titles and metric labels ("Search Result", "Search Time", "States/s",
"Avg. Path Cost"); sentence case for messages ("Profiling enabled, starting trace...").
Status vocabulary is fixed: "Solution Found" / "No Solution Found", "Yes" / "No", "Solved!",
"N/A" for values that do not exist.

Numerals:

| Quantity | Format | Example | Where |
| --- | --- | --- | --- |
| Counts, rates | `human_format`: round to 3 significant digits, then pick `K/M/B/T` (999,999 → `1M`) | `1.23M` | all tables, config panel, progress postfix |
| Search time | `{:.2f} s` | `0.41 s` | result table |
| Average time | `{:.3f} s` | `0.412 s` | evaluation summary |
| Cost | `{:.1f}` (search), `{:.2f}` (evaluation averages) | `20.0` | tables |
| Rates | `{:.1f}%` (search summary), `{:.2f}% (n/N)` (evaluation) | `95.00% (19/20)` | tables |
| Cost gap | signed `{:+.2f}` | `+0.35` | evaluation summary |
| R², CCC | `{:.3f}` | `0.981` | evaluation summary |
| g / h / f | `{:4.1f}` | `g:12.0` | step panel titles |
| Heuristic distance, Q-value | `{:.2f}`, `{:.1f}` | `17.35`, `'U': 12.1` | seed panel `Dist:` |
| Training metrics | `{:.4f}`, or `{:.2e}` when `\|v\| < 0.01` | `0.0123`, `3.10e-04` | progress bar |
| Speed-up | `x{:.1f}` | `x7.9` | vmapped tables |

## Layout

The unit is one character cell. Tokens under `spacing` are cell counts.

- Everything hugs its content: panels and tables use `expand=False`. The only full-width element
  is the two-column config layout (`Columns(..., equal=True, expand=True)`).
- Results come first. A search prints, in order: config panel → seed setup panel (single seed
  only) → result table → optional workload-signature panel → optional solution path → summary
  table (multi-seed only).
- Tables put labels left (cyan) and numbers right-justified. Summary tables separate per-seed rows
  from a bold `Average` row with `add_section()`.
- Seed setup panel: Start and Goal side by side separated by a bold blue ` → `, each headed by a
  centered bold blue label; a `Dist:` line underneath. Without goal data only Start is shown.
- Solution path: step panels flow left to right, joined by ` → ` (`arrow-gutter`, 3 cells), and
  wrap to the terminal width. Panels per row =
  `(console.width - border-allowance + arrow-gutter) // (widest panel + arrow-gutter)`, at least 1.
  Every row after the first starts with a bold blue `↓`. The last panel is the solved state.
- Config panel: a single tree when there are at most two top-level keys or the config is flat;
  otherwise two columns, with the largest item (by JSON length) alone on the left. Strings longer
  than 80 characters end in ` ...`; sequences longer than 10 items show the first 5 and last 5.
- Q-value strips list all actions when there are at most 6, otherwise the first 2, a dim ` ... `,
  and the last 2.
- Progress: a bold blue description line, optional `key : value` lines in cyan, then the bar row
  (spinner, bar, `M/N`, percent, elapsed, remaining, metrics). It refreshes at 10 Hz.
  `RichProgressBar.pause()` stops the live display and brackets interleaved output with rules.
- Batches of states are laid out by Xtructure: 2 head and 2 tail items once a batch exceeds 4.

## Elevation & Depth

Flat. There are no background fills, shadows or reverse video. Hierarchy comes from:

1. Containment: Panel ⊃ grid ⊃ Panel. At most two nested borders (the "Solution Path" panel
   around its step panels).
2. Border color: blue for intermediate steps, green for the solved step and summaries, dim for
   configuration, yellow for differences that need review.
3. Bold colored titles set into the top border of panels.
4. Rules (`console.rule`) as section breaks between sweep runs and around paused progress output.

## Shapes

- Panels: rich's default `box.ROUNDED` (`╭─╮ ╰─╯`).
- Tables: rich's default `box.HEAVY_HEAD` (heavy header rule, light body lines).
- Layout grids: `Table.grid(...)`, no borders.
- Rules: a single `─` line.
- Puzzle content inside panels keeps PuXle's heavy square frames (`┏━┓`). The contrast is
  intentional: rounded means JAxtar chrome, square means puzzle content.
- Glyphs: `→` and `↓` for path direction; the progress spinner is rich's `dots`.

## Components

| Component | Builder | Anatomy | Tokens |
| --- | --- | --- | --- |
| Seed setup panel | `helpers/visualization.py` `build_seed_setup_panel` | Panel titled `Seed N` (bold blue); grid of Start → Goal; `Dist:` line | `state-heading`, `path-arrow`, `heuristic-distance` |
| Result table | `build_result_table` | Title `Search Result for Seed N` (bold); columns Metric (cyan) / Value (right); rows Status, Cost, Search Time, Search States, States/s | `metric-label`, `metric-value`, `status-success`, `status-failure` |
| Summary table | `build_summary_table` | Title `Summary for <puzzle> (N seeds)`; Seed (cyan, right), Solved (Yes/No, centered), time, states, states/s, cost; section break; bold `Average` row | `metric-label`, `status-success`, `status-failure` |
| Solution path | `build_solution_path_panel` | Outer panel `Solution Path` (bold green); step panels titled `g:…\|h:…\|f:…`, board centered, `Action: <label>` under it; final panel says `Solved!` | `step-panel-border`, `final-step-panel-border`, `cost-g`, `cost-h`, `cost-f`, `path-arrow` |
| Solution state panel | `cli/search_runner.py` | Panel `Solution State` / `Vmapped Solution` (bold green) for puzzles without goal data | `status-success` |
| Vmapped setup and results | `build_vmapped_setup_panel`, `build_vmapped_results_table_multi` / `_single` | Panel `Vmapped Search Setup` (bold blue) with batch ↓ goals; Metric/Value tables with speed-ups | `state-heading`, `metric-label` |
| Workload signature | `cli/search_runner.py` | Panel `Xtructure Workload Signature` (bold cyan) around indented JSON | `metric-label` |
| Config panel | `helpers/config_printer.py` `print_config` | Dim-bordered panel, bold green title, tree with bright-blue guides, bold magenta keys | `config-panel-title`, `config-panel-border`, `config-key`, `config-tree-guide` |
| Evaluation summary | `helpers/summaries.py` `create_summary_panel` | Green-bordered panel around a table titled `Evaluation Summary` (bold cyan), bold magenta header, dim Metric column (width 30) | `summary-panel-border`, `eval-table-title`, `eval-table-header`, `eval-row-label` |
| Comparison summary | `create_comparison_summary_panel` | As above; one row per run label | same |
| Configuration differences | `cli/comparison_generator.py` | Yellow-bordered panel, table `Configuration Differences` | `diff-panel-border`, `eval-table-title`, `eval-table-header` |
| Progress bar | `helpers/rich_progress.py` `RichProgressBar` (`tqdm`, `trange`) | Description line, optional detail lines, bar row with green bar and cyan metrics | `progress-description`, `progress-metrics`, `progress-bar` |
| Sweep rules | `cli/evaluation_runner.py` | `Run i/N: pr=…, cw=…, bs=…` (bold cyan); `Sweep Complete…` (bold green) | `rule-run`, `rule-complete` |
| Status lines | `console.print` one-liners | One sentence; the key word or path in bold/color | `status-success`, `status-failure`, `status-warning` |
| Distance strips | `helpers/formatting.py` `heuristic_dist_format`, `qfunction_dist_format` | `17.35` (blue) or `'U': 12.1 \| 'D': 13.0 ...` | `heuristic-distance`, `q-action`, `q-value`, `q-separator`, `elision` |
| CLI error | `cli/runtime.py` `run_cli` | `Error: <message>` on stderr, `markup=False`, exit code 2 | none (plain) |

Component notes:

- Q-strip action labels keep their own ANSI color when PuXle provides one (Lights Out labels are
  bright yellow); uncolored labels become green.
- Border tokens (`*-border`) set rich's `border_style`: in a terminal a border is text, so its
  color is a text color.

## Do's and Don'ts

- Do build new screens from the existing builders; a new result table copies the two-column
  Metric (cyan) / Value (right-justified) shape.
- Do name colors by rich style and pick an existing role from Colors; don't add hex, truecolor
  or new roles to chrome.
- Do keep green and red for outcomes (the `g` cost label is the only exception).
- Do pass PuXle and Xtructure strings through `Text.from_ansi(...)`; don't print raw ANSI inside
  markup, where rich miscounts widths and may parse `[` as a tag.
- Do print external text (exception messages, file paths, user input) with `markup=False` or
  `rich.markup.escape`, as `run_cli` does.
- Do keep `expand=False`, and format counts with `human_format`.
- Do use `helpers.rich_progress` `tqdm`/`trange`, not the `tqdm` package, and wrap output
  printed during a live bar in `bar.pause()`.
- Don't use rich features that `helpers/capture.py` cannot fold into the run's `console.log`:
  it interprets only `\r`, `\b`, erase-line (`ESC[nK`) and cursor-up (`ESC[nA`). No alternate
  screen (`Live(screen=True)`), no absolute cursor positioning.
- Don't add background colors, reverse video or blinking text.

## Known Gaps

Current code that does not follow the rules above. Don't copy these patterns.

- Two table title styles: search tables use plain `[bold]` titles with default headers, while
  evaluation tables use `[bold cyan]` titles and `bold magenta` headers.
- Status lines interpolate run directories into markup without escaping
  (`cli/evaluation_runner.py`, `cli/benchmark_commands.py`, `cli/comparison_generator.py`).
