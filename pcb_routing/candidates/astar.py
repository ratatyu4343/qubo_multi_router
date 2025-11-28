import heapq
from typing import Iterable, List

from .base import CandidateGenerator
from ..Objects import GridBoard, Net, Path


class AStarGenerator(CandidateGenerator):
    """Generate shortest-path style candidates using A* with simple congestion penalties."""

    def __init__(self, board: GridBoard, net: Net, seed=None):
        super().__init__(board, net, seed=seed)

    def generate(self, limit: int = 5) -> Iterable[Path]:
        """Return up to `limit` paths, biasing against reusing cells via a simple penalty map."""
        found: List[Path] = []
        penalty = {}
        for attempt in range(limit):
            start = self.net.src
            goal = self.net.dst
            openq = []
            heapq.heappush(openq, (0 + self.board.manhattan(start, goal), 0, start, [start]))
            seen_cost = {start: 0}
            while openq:
                pri, g, node, path = heapq.heappop(openq)
                if node == goal:
                    found.append(path)
                    for q in path[1:-1]:
                        penalty[q] = penalty.get(q, 0) + 1
                    break
                for nb in self.board.neighbors(node):
                    newg = g + 1 + penalty.get(nb, 0) * 2
                    if seen_cost.get(nb, 1e9) > newg:
                        seen_cost[nb] = newg
                        h = self.board.manhattan(nb, goal)
                        # use local RNG for tie-breaking noise
                        jitter = self.rng.random() * 1e-3
                        heapq.heappush(openq, (newg + h + jitter, newg, nb, path + [nb]))
        return found
