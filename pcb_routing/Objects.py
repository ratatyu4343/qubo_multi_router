from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

Point3D = Tuple[int, int, int]
Path = List[Point3D]


def _as_point3d(value: Sequence[int]) -> Point3D:
    """Best-effort conversion of a coordinate-like sequence to Point3D."""
    if len(value) == 3:
        r, c, z = map(int, value)
        return (r, c, z)
    if len(value) == 2:
        r, c = map(int, value)
        return (r, c, 0)
    raise ValueError(f"Expected 2 or 3 components, got {len(value)}: {value}")


@dataclass
class Pin:
    name: str = "pin"
    coord: Point3D = (0, 0, 0)

    def __post_init__(self):
        self.coord = _as_point3d(self.coord)


@dataclass
class Net:
    src_point: Pin
    dst_point: Pin
    name: str = "net"

    def __post_init__(self):
        # Ensure associated pins have 3D coordinates
        self.src_point.coord = _as_point3d(self.src_point.coord)
        self.dst_point.coord = _as_point3d(self.dst_point.coord)

    @property
    def src(self) -> Point3D:
        return self.src_point.coord

    @property
    def dst(self) -> Point3D:
        return self.dst_point.coord


@dataclass
class GridBoard:
    rows: int
    cols: int
    blocked: Set[Point3D] = field(default_factory=set)

    def __post_init__(self):
        self.blocked = {_as_point3d(cell) for cell in self.blocked}

    def in_bounds(self, p: Sequence[int]) -> bool:
        r, c, z = _as_point3d(p)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return z >= 0

    def passable(self, p: Sequence[int]) -> bool:
        pt = _as_point3d(p)
        return self.in_bounds(pt) and pt not in self.blocked

    def neighbors(self, p: Sequence[int], include_vias: bool = False) -> List[Point3D]:
        r, c, z = _as_point3d(p)
        candidates: List[Point3D] = [
            (r - 1, c, z),
            (r + 1, c, z),
            (r, c - 1, z),
            (r, c + 1, z),
        ]
        if include_vias:
            candidates.extend([(r, c, z - 1), (r, c, z + 1)])
        return [q for q in candidates if self.passable(q)]

    def manhattan(self, a: Sequence[int], b: Sequence[int]) -> int:
        ra, ca, za = _as_point3d(a)
        rb, cb, zb = _as_point3d(b)
        return abs(ra - rb) + abs(ca - cb) + abs(za - zb)

    def add_blocked(self, cell: Sequence[int]) -> None:
        """Add a blocked cell, converting coordinates to 3D."""
        self.blocked.add(_as_point3d(cell))
