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


def heuristic_dist_format(puzzle, dist):
    return f"{dist:.2f}"


def qfunction_dist_format(puzzle, qvalues):
    action_len = qvalues.shape[0]
    if action_len <= 6:
        return " | ".join(
            f"'{puzzle.action_to_string(i)}': {float(qvalues[i]):.1f}" for i in range(action_len)
        )
    else:
        str = " | ".join(
            f"'{puzzle.action_to_string(i)}': {float(qvalues[i]):.1f}" for i in range(2)
        )
        str += " ... "
        str += " | ".join(
            f"'{puzzle.action_to_string(i)}': {float(qvalues[i]):.1f}"
            for i in range(action_len - 2, action_len)
        )
        return str
