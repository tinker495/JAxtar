# Human Play Command

The `human_play` command provides an interactive mode to solve puzzles manually. This is a useful tool for understanding a puzzle's mechanics, testing specific scenarios, or simply having fun.

## Usage

The basic syntax for the `human_play` command is:

```bash
python main.py human_play -p <puzzle_name> [OPTIONS]
```

Once started, the command will display the initial state of the puzzle and a list of available actions. You can perform an action by pressing the corresponding key (e.g., number keys, 'w', 'a', 's', 'd', or arrow keys, depending on the puzzle). The game ends when you solve the puzzle or press `ESC` to exit.

Example:

```bash
python main.py human_play -p slidepuzzle
```

## Options

The `human_play` command primarily uses the puzzle options to set up the environment.

### Puzzle Options (`@puzzle_options`)

These options define the puzzle environment to be solved.

-   `-p, --puzzle`: **(Required)** Specifies the puzzle to play.
    -   Type: `Choice`
    -   Choices: `n-puzzle`, `rubikscube`, `slidepuzzle`, etc.
-   `-pargs, --puzzle_args`: JSON string for additional puzzle-specific arguments.
    -   Type: `String`
-   `-ps, --puzzle_size`: A simpler way to set the size for puzzles that support it.
    -   Type: `String`
    -   Default: `default`
-   `-h, --hard`: If available, use a "hard" version of the puzzle.
    -   Type: `Flag`
-   `-s, --seeds`: A single seed for generating the initial puzzle state. Note: Unlike other commands, `human_play` only accepts a single seed.
    -   Type: `String`
    -   Default: `"0"`
