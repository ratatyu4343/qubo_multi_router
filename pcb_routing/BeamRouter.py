from collections import deque
import logging
from typing import Dict, List, Set, Tuple

from .Objects import GridBoard, Net, Path

logger = logging.getLogger(__name__)

Direction = Tuple[int, int]
DirState = Tuple[Tuple[int, int, int], Direction]


class BeamRouter:
    """Bidirectional beam router with reflective turns when beams hit obstacles."""

    def __init__(self, board: GridBoard, max_layers: int = 10):
        self.board = board
        self.max_layers = max(1, int(max_layers))
        self.directions: List[Direction] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _turn_pairs(self, direction: Direction) -> Tuple[Direction, Direction]:
        dx, dy = direction
        return (dy, -dx), (-dy, dx)

    def _passable(self, cell: Tuple[int, int, int], blocked_layer: Set[Tuple[int, int, int]]) -> bool:
        if not self.board.in_bounds(cell):
            return False
        return cell not in blocked_layer

    def beam_bidirectional(
        self,
        net: Net,
        layer: int,
        blocked: List[Set[Tuple[int, int, int]]],
    ) -> Path:
        src = (net.src[0], net.src[1], layer)
        dst = (net.dst[0], net.dst[1], layer)
        if src == dst:
            return [src]

        blocked_layer = blocked[layer]

        def expand(
            queue: deque,
            seen: Set[DirState],
            paths: Dict[Tuple[int, int, int], Path],
            other_paths: Dict[Tuple[int, int, int], Path],
            from_src: bool,
        ) -> Path | None:
            for _ in range(len(queue)):
                cell, direction, path = queue.popleft()
                forward = (cell[0] + direction[0], cell[1] + direction[1], layer)
                forward_state = (forward, direction)
                advanced = False

                if self._passable(forward, blocked_layer) and forward_state not in seen:
                    seen.add(forward_state)
                    forward_path = path + [forward]
                    queue.append((forward, direction, forward_path))
                    paths.setdefault(forward, forward_path)
                    advanced = True
                    if forward in other_paths:
                        other_path = other_paths[forward]
                        if from_src:
                            return forward_path[:-1] + other_path[::-1]
                        return other_path[:-1] + forward_path[::-1]

                if not advanced:
                    for new_dir in self._turn_pairs(direction):
                        state = (cell, new_dir)
                        if state in seen:
                            continue
                        seen.add(state)
                        queue.append((cell, new_dir, path))

            return None

        beams_src = deque([(src, d, [src]) for d in self.directions])
        beams_dst = deque([(dst, d, [dst]) for d in self.directions])
        seen_src: Set[DirState] = {(src, d) for d in self.directions}
        seen_dst: Set[DirState] = {(dst, d) for d in self.directions}
        paths_src: Dict[Tuple[int, int, int], Path] = {src: [src]}
        paths_dst: Dict[Tuple[int, int, int], Path] = {dst: [dst]}

        while beams_src and beams_dst:
            meeting = expand(beams_src, seen_src, paths_src, paths_dst, True)
            if meeting:
                return self._simplify_path(meeting)
            meeting = expand(beams_dst, seen_dst, paths_dst, paths_src, False)
            if meeting:
                return self._simplify_path(meeting)

        return []

    def route_net(self, net: Net, blocked_layers: List[Set[Tuple[int, int, int]]]) -> Tuple[Path, int]:
        for layer in range(self.max_layers):
            path = self.beam_bidirectional(net, layer, blocked_layers)
            if path:
                return path, layer
        return [], None

    def route_all(self, nets: List[Net]) -> Dict[int, Path]:
        result: Dict[int, Path] = {}
        used_layers = 0

        base_blocked = [set() for _ in range(self.max_layers)]
        for cell in getattr(self.board, "blocked", []):
            layer = cell[2]
            if 0 <= layer < self.max_layers:
                base_blocked[layer].add(cell)

        all_pins = [tuple(pin) for net in nets for pin in (net.src, net.dst)]

        for idx, net in enumerate(nets):
            per_net_blocked = [set(layer_cells) for layer_cells in base_blocked]

            src_pin = tuple(net.src)
            dst_pin = tuple(net.dst)

            for pin in all_pins:
                if pin == src_pin or pin == dst_pin:
                    continue
                for layer_idx in range(self.max_layers):
                    per_net_blocked[layer_idx].add((pin[0], pin[1], layer_idx))

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
                    "Net %s could not be routed by Beam Algorithm even with %s layers.",
                    idx,
                    self.max_layers,
                )

        logger.info("Total layers used: %s", used_layers)
        return result

    @staticmethod
    def _simplify_path(path: Path) -> Path:
        stack: Path = []
        index_map: Dict[Tuple[int, int, int], int] = {}
        for cell in path:
            cell_key = (int(cell[0]), int(cell[1]), int(cell[2])) if len(cell) == 3 else tuple(cell)
            if cell_key in index_map:
                keep_upto = index_map[cell_key]
                for removed in stack[keep_upto + 1 :]:
                    index_map.pop(removed, None)
                stack = stack[: keep_upto + 1]
            else:
                stack.append(cell_key)  # type: ignore[arg-type]
                index_map[cell_key] = len(stack) - 1
        return stack
