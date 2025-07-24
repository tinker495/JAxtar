import numpy as np
from rich.text import Text


def human_format_to_float(num_str):
    num_str = num_str.upper()  # convert to uppercase
    num_str = num_str.replace("K", "e3").replace("M", "e6").replace("B", "e9").replace("T", "e12")
    return float(num_str)


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def heuristic_dist_format(puzzle, dist) -> Text:
    """Formats the heuristic distance using Rich."""
    return Text(f"{dist:.2f}", style="blue")


def qfunction_dist_format(puzzle, qvalues) -> Text:
    """Formats the Q-function distribution using Rich."""
    action_len = qvalues.shape[0]

    def format_part(i):
        action_str = Text.from_ansi(f"'{puzzle.action_to_string(i)}'")
        if action_str.style == "":
            action_str.style = "green"
        q_value = f"{float(qvalues[i]):.1f}"
        return Text.assemble(action_str, (": ", "white"), (Text(q_value, "magenta")))

    if action_len <= 6:
        parts = [format_part(i) for i in range(action_len)]
        return Text(" | ", style="white").join(parts)
    else:
        first_parts = [format_part(i) for i in range(2)]
        last_parts = [format_part(i) for i in range(action_len - 2, action_len)]

        final_text = Text()
        final_text.append(Text(" | ", style="white").join(first_parts))
        final_text.append(" ... ", style="dim white")
        final_text.append(Text(" | ", style="white").join(last_parts))
        return final_text


def img_to_colored_str(img: np.ndarray) -> str:
    """
    Convert a numpy array to an ascii string.
    img size = (32, 32, 3)
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ascii_art_lines = []
    for row in img:
        line = ""
        for pixel in row:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            char = "██"
            # Append the character with ANSI escape codes to reflect its original color
            line += f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m"
        ascii_art_lines.append(line)
    return "\n".join(ascii_art_lines)
