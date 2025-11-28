from .Objects import Path


def path_cost(path: Path) -> int:
    """Calculate the cost (length) of the given path."""
    return max(0, len(path) - 1)


def path_turns(path: Path) -> int:
    """Count number of turns in a 3D path.
    Via-only moves (layer changes without planar displacement) do not count as turns.
    """
    if len(path) < 3:
        return 0

    moves = []
    prev = path[0]
    for nxt in path[1:]:
        dr = nxt[0] - prev[0]
        dc = nxt[1] - prev[1]
        if dr != 0 or dc != 0:
            moves.append((dr, dc))
        prev = nxt
    if len(moves) < 2:
        return 0
    turns = 0
    for i in range(1, len(moves)):
        if moves[i] != moves[i - 1]:
            turns += 1
    return turns


def path_to_str(path: Path) -> str:
    """Convert a path to a human-readable string."""
    return "->".join(f"({r},{c},{z})" for r, c, z in path)
