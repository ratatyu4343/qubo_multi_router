from collections import deque
import logging
from typing import Dict, List, Optional, Set, Tuple

from .Objects import GridBoard, Net, Path

logger = logging.getLogger(__name__)

Coord = Tuple[int, int, int]


class WaveRouter:
    def __init__(self, board: GridBoard, max_layers: int = 10):
        self.board = board
        self.max_layers = max(1, int(max_layers))

    def _neighbors(self, cell: Coord) -> List[Coord]:
        r, c, layer = cell
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(r + dr, c + dc, layer) for dr, dc in moves]

    def _reconstruct_path(
        self,
        meeting: Coord,
        parent_src: Dict[Coord, Optional[Coord]],
        parent_dst: Dict[Coord, Optional[Coord]],
    ) -> Path:
        path: Path = []
        # Walk back from meeting to source
        current: Optional[Coord] = meeting
        while current is not None:
            path.append(current)
            current = parent_src.get(current)
        path.reverse()

        # Walk from meeting towards destination, skipping meeting itself
        current = parent_dst.get(meeting)
        while current is not None:
            path.append(current)
            current = parent_dst.get(current)
        return path

    def _route_single_layer(
        self,
        net: Net,
        layer: int,
        blocked_layers: List[Set[Coord]],
    ) -> Path:
        src = (net.src[0], net.src[1], layer)
        dst = (net.dst[0], net.dst[1], layer)
        if src == dst:
            return [src]

        queue_src: deque[Coord] = deque([src])
        queue_dst: deque[Coord] = deque([dst])
        visited_src: Set[Coord] = {src}
        visited_dst: Set[Coord] = {dst}
        parent_src: Dict[Coord, Optional[Coord]] = {src: None}
        parent_dst: Dict[Coord, Optional[Coord]] = {dst: None}

        while queue_src and queue_dst:
            # Expand from source side
            for _ in range(len(queue_src)):
                cell = queue_src.popleft()
                for neighbor in self._neighbors(cell):
                    if not self.board.in_bounds(neighbor):
                        continue
                    if neighbor in blocked_layers[layer]:
                        continue
                    if neighbor in visited_src:
                        continue
                    visited_src.add(neighbor)
                    parent_src[neighbor] = cell
                    queue_src.append(neighbor)
                    if neighbor in visited_dst:
                        return self._reconstruct_path(neighbor, parent_src, parent_dst)

            # Expand from destination side
            for _ in range(len(queue_dst)):
                cell = queue_dst.popleft()
                for neighbor in self._neighbors(cell):
                    if not self.board.in_bounds(neighbor):
                        continue
                    if neighbor in blocked_layers[layer]:
                        continue
                    if neighbor in visited_dst:
                        continue
                    visited_dst.add(neighbor)
                    parent_dst[neighbor] = cell
                    queue_dst.append(neighbor)
                    if neighbor in visited_src:
                        return self._reconstruct_path(neighbor, parent_src, parent_dst)

        return []

    def route_net(self, net: Net, blocked_layers: List[Set[Coord]]) -> Tuple[Path, Optional[int]]:
        for layer in range(self.max_layers):
            path = self._route_single_layer(net, layer, blocked_layers)
            if path:
                return path, layer
        return [], None

    def route_all(self, nets: List[Net]) -> Dict[int, Path]:
        result: Dict[int, Path] = {}
        used_layers = 0

        base_blocked: List[Set[Coord]] = [set() for _ in range(self.max_layers)]
        for layer in range(self.max_layers):
            for cell in self.board.blocked:
                if cell[2] == layer:
                    base_blocked[layer].add((cell[0], cell[1], layer))

        all_pins = [tuple(pin) for net in nets for pin in (net.src, net.dst)]

        for idx, net in enumerate(nets):
            per_net_blocked = [set(layer_cells) for layer_cells in base_blocked]

            src_pin = (net.src[0], net.src[1])
            dst_pin = (net.dst[0], net.dst[1])

            for pin in all_pins:
                pin_rc = (pin[0], pin[1])
                if pin_rc == src_pin or pin_rc == dst_pin:
                    continue
                for layer_idx in range(self.max_layers):
                    per_net_blocked[layer_idx].add((pin[0], pin[1], layer_idx))

            # Ensure current pins are free on all layers
            for layer_idx in range(self.max_layers):
                per_net_blocked[layer_idx].discard((src_pin[0], src_pin[1], layer_idx))
                per_net_blocked[layer_idx].discard((dst_pin[0], dst_pin[1], layer_idx))

            path, layer = self.route_net(net, per_net_blocked)
            if path and layer is not None:
                result[idx] = path
                used_layers = max(used_layers, layer + 1)
                for cell in path:
                    base_blocked[cell[2]].add(cell)
            else:
                logger.warning(
                    "Net %s could not be routed even with %s layers.",
                    idx,
                    self.max_layers,
                )

        logger.info("Total layers used: %s", used_layers)
        return result
