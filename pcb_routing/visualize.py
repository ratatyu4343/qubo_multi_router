"""Single-entry visualization helper for PCB routing."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt

PointLike = Sequence[int]
PathEntry = Tuple[List[PointLike], str]


def _rc(pt: PointLike) -> Tuple[int, int]:
    return int(pt[0]), int(pt[1])


def _layer(pt: PointLike) -> int:
    return int(pt[2]) if len(pt) >= 3 else 0


def _blocked_xy_by_layer(board) -> Dict[int, Set[Tuple[int, int]]]:
    layers: Dict[int, Set[Tuple[int, int]]] = {}
    board_layers = getattr(board, "layers", None)
    if isinstance(board_layers, int) and board_layers >= 0:
        for layer in range(board_layers):
            layers[layer] = set()
    for cell in getattr(board, "blocked", []):
        layers.setdefault(_layer(cell), set()).add(_rc(cell))
    return layers


def _prepare_paths(paths: Union[None, Dict[int, Iterable[PointLike]], Iterable], nets) -> List[PathEntry]:
    if paths is None:
        return []

    if isinstance(paths, dict):
        normalized: List[PathEntry] = []
        for idx, path in sorted(paths.items()):
            label = getattr(nets[idx], "name", f"net_{idx}")
            normalized.append((list(path), label))
        return normalized

    normalized: List[PathEntry] = []
    for item in list(paths):
        if isinstance(item, tuple) and len(item) == 2:
            path, label = item
        else:
            path, label = item, ''
        normalized.append((list(path), label))
    return normalized


def _needs_multilayer(paths: List[PathEntry]) -> bool:
    layers = {_layer(pt) for path, _ in paths for pt in path}
    return len(layers) > 1


def _palette() -> List[str]:
    return [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#1b9e77",
        "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    ]


def _draw_single(board, nets, paths: List[PathEntry], show: bool, save_path: str | None) -> None:
    blocked_layers = _blocked_xy_by_layer(board)
    blocked_xy = set().union(*blocked_layers.values()) if blocked_layers else set()

    fig, ax = plt.subplots(figsize=(board.cols / 2, board.rows / 2))
    for r in range(board.rows):
        for c in range(board.cols):
            color = 'black' if (r, c) in blocked_xy else 'white'
            ax.add_patch(
                plt.Rectangle((c, board.rows - 1 - r), 1, 1, facecolor=color, edgecolor='black')
            )

    font_kwargs = dict(color='black', fontsize=8, ha='center', va='bottom', fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8))
    seen_pins = set()
    for net in nets:
        for coord, name in ((net.src, getattr(net.src_point, "name", "")),
                            (net.dst, getattr(net.dst_point, "name", ""))):
            sr, sc = _rc(coord)
            ax.plot(sc + 0.5, board.rows - 1 - sr + 0.5, 'ko', markersize=8)
            if name and (sr, sc) not in seen_pins:
                ax.text(sc + 0.5, board.rows - 1 - sr + 0.7, name, **font_kwargs)
                seen_pins.add((sr, sc))

    palette = _palette()
    for idx, (path, label) in enumerate(paths):
        coords = list(map(_rc, path))
        if len(coords) < 2:
            continue
        xs = [c + 0.5 for _, c in coords]
        ys = [board.rows - 1 - r + 0.5 for r, _ in coords]
        color = palette[idx % len(palette)]
        ax.plot(xs, ys, '-', color=color, linewidth=2)
        if label:
            mid = len(coords) // 2
            ax.text(xs[mid], ys[mid], label, color=color, fontsize=8,
                    ha='center', va='center', fontstyle='italic',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8))

    ax.set_xlim(0, board.cols)
    ax.set_ylim(0, board.rows)
    ax.set_xticks(range(board.cols + 1))
    ax.set_yticks(range(board.rows + 1))
    ax.set_aspect('equal')
    ax.grid(True, color='black', linestyle=':', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def _draw_layers(board, nets, paths: List[PathEntry], show: bool, save_path: str | None) -> None:
    blocked_layers = _blocked_xy_by_layer(board)
    used_layers = sorted({_layer(pt) for path, _ in paths for pt in path}) or [0]
    fig, axs = plt.subplots(1, len(used_layers),
                            figsize=(board.cols * max(1, len(used_layers)) / 3, board.rows / 2),
                            squeeze=False)
    axs = axs[0]
    palette = _palette()

    for idx_layer, layer in enumerate(used_layers):
        ax = axs[idx_layer]
        ax.set_title(f"Layer {layer + 1}")
        blocked_xy = blocked_layers.get(layer, set())
        for r in range(board.rows):
            for c in range(board.cols):
                color = 'black' if (r, c) in blocked_xy else 'white'
                ax.add_patch(
                    plt.Rectangle((c, board.rows - 1 - r), 1, 1, facecolor=color, edgecolor='black')
                )
        for net in nets:
            sr, sc = _rc(net.src)
            dr, dc = _rc(net.dst)
            ax.plot(sc + 0.5, board.rows - 1 - sr + 0.5, 'ko', markersize=6)
            ax.plot(dc + 0.5, board.rows - 1 - dr + 0.5, 'ko', markersize=6)
        for idx, (path, label) in enumerate(paths):
            layer_path = [pt for pt in path if _layer(pt) == layer]
            if len(layer_path) < 2:
                continue
            coords = list(map(_rc, layer_path))
            xs = [c + 0.5 for _, c in coords]
            ys = [board.rows - 1 - r + 0.5 for r, _ in coords]
            color = palette[idx % len(palette)]
            ax.plot(xs, ys, '-', color=color, linewidth=2)
            if label:
                mid = len(coords) // 2
                ax.text(xs[mid], ys[mid], label, color=color, fontsize=8,
                        ha='center', va='center', fontstyle='italic',
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8))
        ax.set_xlim(0, board.cols)
        ax.set_ylim(0, board.rows)
        ax.set_xticks(range(board.cols + 1))
        ax.set_yticks(range(board.rows + 1))
        ax.set_aspect('equal')
        ax.grid(True, color='black', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def visualize(board, nets, paths: Union[None, Dict[int, Iterable[PointLike]], Iterable] = None,
             show: bool = True, save_path: str | None = None) -> None:
    """Render routed paths on the board.

    Args:
        board: GridBoard.
        nets: iterable of Net objects (indices correspond to solution keys).
        paths: dict[int, path], iterable of paths, or iterable of (path, label).
        show: display via matplotlib.
        save_path: optional filepath to save PNG.
    """
    normalized = _prepare_paths(paths, nets)
    if normalized and _needs_multilayer(normalized):
        _draw_layers(board, nets, normalized, show, save_path)
    else:
        _draw_single(board, nets, normalized, show, save_path)
