from typing import Dict
from .utilities import path_cost, path_turns


def total_length(solution) -> int:
    if not solution:
        return 0
    # solution: dict[int, path]
    return sum(path_cost(solution[k]) for k in solution)


def total_turns(solution) -> int:
    if not solution:
        return 0
    return sum(path_turns(solution[k]) for k in solution)


def routed_count(solution) -> int:
    return len(solution) if solution else 0


def layers_used(solution) -> int:
    """Estimate number of PCB layers actually used by the solution.
    Counts distinct layer indices present in paths; falls back to 1 if none.
    """
    if not solution:
        return 0
    layers = set()
    for path in solution.values():
        for p in path:
            if isinstance(p, tuple) and len(p) == 3 and p[2] is not None:
                try:
                    layers.add(int(p[2]))
                except Exception:
                    pass
    return len(layers) if layers else 1
