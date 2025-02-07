import click
import jax
import numpy as np

from puzzle.sokoban import Sokoban


def convert_level(level):
    """
    Convert a single 10x10 level (list of 10 string lines) into
    two flattened arrays (of size 100) representing the init and target boards.

    Encoding:
      - '#' is a wall (1) for both boards.
      - '@' is the player's start (2) in the init board.
      - '$' is a box (3) in the init board.
      - '.' marks a goal position: set as 3 in the target board.
      - All other characters (e.g. space) are treated as empty (0).
    """
    init_board = np.zeros((100,), dtype=np.uint8)
    target_board = np.zeros((100,), dtype=np.uint8)
    # define encoding values
    WALL = 1
    PLAYER = 2
    BOX = 3

    for i, row in enumerate(level):
        for j, ch in enumerate(row):
            idx = i * 10 + j
            if ch == "#":
                init_board[idx] = WALL
                target_board[idx] = WALL
            elif ch == "@":
                init_board[idx] = PLAYER
            elif ch == "$":
                init_board[idx] = BOX
            elif ch == ".":
                target_board[idx] = BOX
            else:
                # assume space or unknown remains as EMPTY
                pass
    return init_board, target_board


def parse_file(file_path):
    """
    Read the level file where each level starts with a semicolon (e.g. "; 0")
    followed by 10 lines of 10 characters. It returns two lists of numpy arrays,
    one for init boards and one for target boards.
    """
    puzzles_init = []
    puzzles_target = []

    with open(file_path, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]

    i = 0
    while i < len(lines):
        # each level begins with a semicolon line.
        if lines[i].startswith(";"):
            i += 1  # skip the semicolon header line
            level_lines = []
            # collect exactly 10 lines for the level
            for _ in range(10):
                if i < len(lines):
                    level_lines.append(lines[i])
                    i += 1
                else:
                    break
            if len(level_lines) != 10:
                print("Warning: expected 10 lines for a level but got", len(level_lines))
                continue
            init_board, target_board = convert_level(level_lines)
            puzzles_init.append(init_board)
            puzzles_target.append(target_board)
        else:
            i += 1  # skip any lines not relevant
    return puzzles_init, puzzles_target


input_files = [
    "boxoban-levels/unfiltered/train/000.txt",
    "boxoban-levels/unfiltered/train/001.txt",
    "boxoban-levels/unfiltered/train/002.txt",
    "boxoban-levels/unfiltered/train/003.txt",
    "boxoban-levels/unfiltered/train/004.txt",
    "boxoban-levels/unfiltered/train/005.txt",
    "boxoban-levels/unfiltered/train/006.txt",
    "boxoban-levels/unfiltered/train/007.txt",
    "boxoban-levels/unfiltered/train/008.txt",
    "boxoban-levels/unfiltered/train/009.txt",
    "boxoban-levels/unfiltered/train/010.txt",
    "boxoban-levels/unfiltered/train/011.txt",
    "boxoban-levels/unfiltered/train/012.txt",
    "boxoban-levels/unfiltered/train/013.txt",
    "boxoban-levels/unfiltered/train/014.txt",
    "boxoban-levels/unfiltered/train/015.txt",
    "boxoban-levels/unfiltered/train/016.txt",
    "boxoban-levels/unfiltered/train/017.txt",
    "boxoban-levels/unfiltered/train/018.txt",
    "boxoban-levels/unfiltered/train/019.txt",
    "boxoban-levels/unfiltered/train/020.txt",
    "boxoban-levels/unfiltered/train/021.txt",
    "boxoban-levels/unfiltered/train/022.txt",
    "boxoban-levels/unfiltered/train/023.txt",
    "boxoban-levels/unfiltered/train/024.txt",
    "boxoban-levels/unfiltered/train/025.txt",
    "boxoban-levels/unfiltered/train/026.txt",
    "boxoban-levels/unfiltered/train/027.txt",
    "boxoban-levels/unfiltered/train/028.txt",
    "boxoban-levels/unfiltered/train/029.txt",
]


@click.command()
@click.option("--init-out", default="init.npy", help="Output file for init puzzles (.npy)")
@click.option("--target-out", default="target.npy", help="Output file for target puzzles (.npy)")
def cli(init_out, target_out):
    puzzles_init = []
    puzzles_target = []
    for file_path in input_files:
        sub_init, sub_target = parse_file(file_path)
        puzzles_init.extend(sub_init)
        puzzles_target.extend(sub_target)
    if puzzles_init:
        puzzles_init = np.array(puzzles_init)
        puzzles_target = np.array(puzzles_target)
        print(puzzles_init.shape, puzzles_target.shape)
        puzzles_init = jax.vmap(Sokoban.pack_board)(puzzles_init)
        puzzles_target = jax.vmap(Sokoban.pack_board)(puzzles_target)
        print(puzzles_init.shape, puzzles_target.shape)
        np.save(init_out, puzzles_init)
        np.save(target_out, puzzles_target)
        click.echo(f"Parsed and saved {len(puzzles_init)} puzzles to {init_out} and {target_out}")
    else:
        click.echo("No valid puzzles were found in the files.")


if __name__ == "__main__":
    cli()
