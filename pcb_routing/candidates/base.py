from typing import Iterable, List, Optional
import random

from ..Objects import GridBoard, Net, Path


class CandidateGenerator:
    """Base class for per-net candidate path generators."""

    def __init__(self, board: GridBoard, net: Net, seed: Optional[int] = None):
        self.board = board
        self.net = net
        self.seed = seed
        # Local RNG for generators that choose to use it
        self.rng = random.Random(seed) if seed is not None else random

    def generate(self, limit: int = 50) -> Iterable[Path]:
        """Return an iterable of candidate paths (up to `limit`)."""
        raise NotImplementedError
