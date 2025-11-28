import pytest

from pcb_routing import GridBoard, Net, Pin
from pcb_routing.candidates import REGISTRY


def is_valid_path(board: GridBoard, net: Net, path):
    if not path or path[0] != net.src or path[-1] != net.dst:
        return False
    layer = path[0][2]
    if any(pt[2] != layer for pt in path):
        return False
    seen = set()
    for a, b in zip(path, path[1:]):
        if a in seen:
            return False
        seen.add(a)
        if not board.in_bounds(a) or not board.in_bounds(b):
            return False
        if not board.passable(a) or not board.passable(b):
            return False
        if a[2] != b[2]:
            return False
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


@pytest.mark.parametrize("gen_key", [
    "astar",
])
def test_generators_basic(gen_key):
    B = GridBoard(5, 5)
    a = Pin("A", (0, 0, 0))
    b = Pin("B", (0, 4, 0))
    net = Net(a, b, "N1")

    Gcls = REGISTRY.get(gen_key)
    assert Gcls is not None, f"Generator {gen_key} not registered"
    try:
        G = Gcls(B, net, seed=123)
    except TypeError:
        G = Gcls(B, net)
    paths = list(G.generate(limit=3))
    assert isinstance(paths, list)
    for p in paths:
        assert all(len(pt) == 3 for pt in p), f"Path must be 3D: {p}"
        assert is_valid_path(B, net, p)
