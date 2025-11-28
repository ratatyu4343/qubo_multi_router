"""
Experiment utilities for running PCB trace routing scenarios.

The ``Experiment`` class coordinates routing algorithms from ``pcb_routing``,
records metrics, and optionally saves visualization artefacts.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
from uuid import uuid4

from pcb_routing import GridBoard, Net, Pin
from pcb_routing.metrics import layers_used, routed_count, total_length, total_turns
from pcb_routing.visualize import visualize

logger = logging.getLogger(__name__)

PathLike = Sequence[int]
PathType = Sequence[PathLike]
SolutionType = Dict[int, PathType]
RunnerType = Callable[[GridBoard, Sequence[Net], "RoutingConfiguration"], SolutionType]


@dataclass
class RoutingConfiguration:
    """Configuration that describes how a routing run should be executed."""

    name: str
    algorithm: str = "wave"
    router_kwargs: Dict[str, Any] = field(default_factory=dict)
    solve_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    runner: Optional[RunnerType] = None

    def normalized_algorithm(self) -> str:
        return self.algorithm.lower().strip()


@dataclass
class RunResult:
    """Summary of a single routing execution."""

    run_id: str
    config_name: str
    algorithm: str
    timestamp: str
    metrics: Dict[str, Any]
    metadata: Mapping[str, Any]
    config_snapshot: Mapping[str, Any]
    router_state: Mapping[str, Any]
    solution: SolutionType
    metrics_path: Path
    solution_path: Path
    image_path: Optional[Path]
    manifest_path: Path
    board_path: Path


class Experiment:
    """Orchestrates routing experiments and manages outputs on disk."""

    def __init__(
        self,
        name: str,
        board: GridBoard,
        nets: Sequence[Net],
        output_dir: Path | str,
    ) -> None:
        self.name = name
        self.board = board
        self.nets: List[Net] = list(nets)
        self.output_root = Path(output_dir).expanduser()
        self.experiment_dir = self.output_root / self.name
        self.metrics_dir = self.experiment_dir / "metrics"
        self.images_dir = self.experiment_dir / "images"
        self.solutions_dir = self.experiment_dir / "solutions"
        self.board_path = self.experiment_dir / "board.json"
        self.definition_path = self.experiment_dir / "experiment.json"
        self.manifest_path = self.experiment_dir / "metrics_log.csv"
        self._ensure_directories()

        self._configurations: Dict[str, RoutingConfiguration] = {}
        self.history: Dict[str, RunResult] = {}
        self._manifest_fieldnames: Optional[List[str]] = None
        self._net_names: List[str] = self._build_net_names()
        self._write_board_description()
        self._write_experiment_definition()

    def add_configuration(self, config: RoutingConfiguration) -> None:
        """Register a routing configuration by name."""
        if config.name in self._configurations:
            raise ValueError(f"Configuration {config.name!r} already exists.")
        self._configurations[config.name] = config
        logger.debug("Configuration %s registered.", config.name)
        self._write_experiment_definition()

    def add_configurations(self, configs: Iterable[RoutingConfiguration]) -> None:
        """Register multiple configurations in one call."""
        for config in configs:
            self.add_configuration(config)

    def remove_configuration(self, name: str) -> None:
        """Remove a previously registered configuration."""
        self._configurations.pop(name, None)
        self.history.pop(name, None)
        logger.debug("Configuration %s removed.", name)
        self._write_experiment_definition()

    def run_configuration(
        self,
        name: str,
        *,
        save_image: bool = False,
        show_image: bool = False,
    ) -> RunResult:
        """Execute a single configuration."""
        config = self._configurations.get(name)
        if config is None:
            raise KeyError(f"Configuration {name!r} is not registered.")

        logger.info("Running configuration %s using %s algorithm.", name, config.algorithm)
        solution, router_params = self._execute(config)
        conflict_count, conflict_map = self._conflict_stats(solution)
        metrics = self._compute_metrics(solution, conflict_count)
        timestamp = self._timestamp()
        run_id = self._generate_run_id()

        config_snapshot = self._config_snapshot(config)
        router_state = self._router_state(router_params, config.router_kwargs)

        metrics_path = self._write_metrics(
            config_snapshot,
            metrics,
            timestamp,
            run_id,
            router_state,
        )
        solution_path = self._write_solution(
            config_snapshot,
            solution,
            timestamp,
            run_id,
            conflict_count,
            conflict_map,
            router_state,
        )

        image_path: Optional[Path] = None
        if save_image:
            image_path = self.images_dir / f"{config.name}_{timestamp}_{run_id}.png"
            visualize(
                self.board,
                self.nets,
                solution,
                show=show_image,
                save_path=str(image_path),
            )
        elif show_image:
            visualize(self.board, self.nets, solution, show=True, save_path=None)

        self._append_manifest(
            run_id=run_id,
            config_snapshot=config_snapshot,
            router_state=router_state,
            timestamp=timestamp,
            metrics=metrics,
            metrics_path=metrics_path,
            solution_path=solution_path,
            image_path=image_path,
        )

        result = RunResult(
            run_id=run_id,
            config_name=config.name,
            algorithm=config.algorithm,
            timestamp=timestamp,
            metrics=metrics,
            metadata=config.metadata,
            config_snapshot=config_snapshot,
            router_state=router_state,
            solution=solution,
            metrics_path=metrics_path,
            solution_path=solution_path,
            image_path=image_path,
            manifest_path=self.manifest_path,
            board_path=self.board_path,
        )
        self.history[config.name] = result
        logger.info("Configuration %s completed.", name)
        return result

    def run_all(
        self,
        config_names: Iterable[str] | None = None,
        *,
        save_images: bool = False,
        show_images: bool = False,
    ) -> List[RunResult]:
        """Execute multiple configurations, returning the collected results."""
        names = list(config_names) if config_names is not None else list(self._configurations.keys())
        results: List[RunResult] = []
        for name in names:
            result = self.run_configuration(name, save_image=save_images, show_image=show_images)
            results.append(result)
        return results

    def _execute(self, config: RoutingConfiguration) -> tuple[SolutionType, Mapping[str, Any]]:
        if config.runner is not None:
            solution = config.runner(self.board, self.nets, config)
            return solution, dict(config.router_kwargs)

        algorithm = config.normalized_algorithm()

        if algorithm == "wave":
            from pcb_routing.WaveRouter import WaveRouter

            kwargs = dict(config.router_kwargs)
            router = WaveRouter(self.board, **kwargs)
            return router.route_all(self.nets), kwargs

        if algorithm == "beam":
            from pcb_routing.BeamRouter import BeamRouter

            kwargs = dict(config.router_kwargs)
            router = BeamRouter(self.board, **kwargs)
            return router.route_all(self.nets), kwargs

        if algorithm in {"quantum", "quantum_multilayer"}:
            from pcb_routing import MultiNetQuantumRouter

            kwargs = dict(config.router_kwargs)
            router = MultiNetQuantumRouter(self.board, **kwargs)

            solve_kwargs = dict(config.solve_kwargs)
            solution = router.solve_multilayer(self.nets, **solve_kwargs)
            params = router.get_parameters() if hasattr(router, "get_parameters") else kwargs
            return solution, params

        raise ValueError(f"Unsupported algorithm {config.algorithm!r}.")

    def _compute_metrics(self, solution: SolutionType, conflict_count: int) -> Dict[str, Any]:
        return {
            "total_nets": len(self.nets),
            "routed_nets": routed_count(solution),
            "total_length": total_length(solution),
            "total_turns": total_turns(solution),
            "layers_used": layers_used(solution),
            "cell_conflicts": conflict_count,
        }

    def _write_metrics(
        self,
        config_snapshot: Mapping[str, Any],
        metrics: Dict[str, Any],
        timestamp: str,
        run_id: str,
        router_state: Mapping[str, Any],
    ) -> Path:
        payload = {
            "experiment": self.name,
            "run_id": run_id,
            "name": config_snapshot["name"],
            "algorithm": config_snapshot["algorithm"],
            "config": config_snapshot,
            "router_state": router_state,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        path = self.metrics_dir / f"{config_snapshot['name']}_{timestamp}_{run_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def _write_solution(
        self,
        config_snapshot: Mapping[str, Any],
        solution: SolutionType,
        timestamp: str,
        run_id: str,
        conflict_count: int,
        conflict_map: Dict[tuple[int, int, int], List[str]],
        router_state: Mapping[str, Any],
    ) -> Path:
        path = self.solutions_dir / f"{config_snapshot['name']}_{timestamp}_{run_id}.json"
        routed_names: List[str] = []
        paths_payload: Dict[str, List[List[int]]] = {}

        for idx, net_name in enumerate(self._net_names):
            raw_path = solution.get(idx, [])
            serialized_path = [list(map(int, cell)) for cell in raw_path]
            if serialized_path:
                routed_names.append(net_name)
            paths_payload[net_name] = serialized_path

        payload = {
            "experiment": self.name,
            "run_id": run_id,
            "config": config_snapshot,
            "algorithm": config_snapshot["algorithm"],
            "timestamp": timestamp,
            "total_nets": len(self.nets),
            "routed_net_names": routed_names,
            "unrouted_net_names": [name for name in self._net_names if name not in routed_names],
            "net_pin_mapping": {
                net_name: {
                    "src_pin": net.src_point.name,
                    "dst_pin": net.dst_point.name,
                }
                for net_name, net in zip(self._net_names, self.nets)
            },
            "router_state": router_state,
            "paths": paths_payload,
            "cell_conflicts": conflict_count,
            "conflict_cells": [
                {"cell": [r, c, layer], "nets": net_names}
                for (r, c, layer), net_names in conflict_map.items()
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def _conflict_stats(
        self,
        solution: SolutionType,
    ) -> tuple[int, Dict[tuple[int, int, int], List[str]]]:
        cell_to_nets: Dict[tuple[int, int, int], set[int]] = {}
        pin_points: set[tuple[int, int, int]] = set()
        for net in self.nets:
            pin_points.add((int(net.src[0]), int(net.src[1]), 0))
            pin_points.add((int(net.dst[0]), int(net.dst[1]), 0))

        for net_idx, path in solution.items():
            if not path:
                continue
            for cell in path:
                try:
                    r = int(cell[0])
                    c = int(cell[1])
                    layer = int(cell[2]) if len(cell) >= 3 else 0
                except (TypeError, ValueError, IndexError):
                    continue
                key = (r, c, layer)
                cell_to_nets.setdefault(key, set()).add(net_idx)

        conflict_cells: Dict[tuple[int, int, int], List[str]] = {}
        for cell, net_indices in cell_to_nets.items():
            if len(net_indices) > 1:
                if cell in pin_points:
                    continue
                names = sorted(self._net_names[idx] for idx in net_indices if idx < len(self._net_names))
                conflict_cells[cell] = names

        return len(conflict_cells), conflict_cells

    def _ensure_directories(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        for directory in (self.metrics_dir, self.images_dir, self.solutions_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _append_manifest(
        self,
        *,
        run_id: str,
        config_snapshot: Mapping[str, Any],
        router_state: Mapping[str, Any],
        timestamp: str,
        metrics: Mapping[str, Any],
        metrics_path: Path,
        solution_path: Path,
        image_path: Optional[Path],
    ) -> None:
        config_json = json.dumps(config_snapshot, ensure_ascii=False)
        router_state_json = json.dumps(router_state, ensure_ascii=False)
        fieldnames = self._manifest_fieldnames or self._build_manifest_fieldnames(metrics)
        self._ensure_manifest_header(fieldnames)
        row: Dict[str, Any] = {
            "experiment_name": self.name,
            "config_name": config_snapshot["name"],
            "run_id": run_id,
            "timestamp": timestamp,
            "algorithm": config_snapshot["algorithm"],
            "metrics_path": str(metrics_path),
            "solution_path": str(solution_path),
            "image_path": str(image_path) if image_path else "",
            "config_json": config_json,
            "router_state_json": router_state_json,
        }
        row.update(metrics)
        with self.manifest_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

    def _write_experiment_definition(self) -> None:
        self.definition_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "output_dir": str(self.output_root),
            "board_file": self.board_path.name,
            "configs": [
                {
                    "name": cfg.name,
                    "algorithm": cfg.algorithm,
                    "router_kwargs": cfg.router_kwargs,
                    "solve_kwargs": cfg.solve_kwargs,
                    "metadata": cfg.metadata,
                }
                for cfg in self._configurations.values()
            ],
        }
        with self.definition_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _build_manifest_fieldnames(self, metrics: Mapping[str, Any]) -> List[str]:
        base = [
            "experiment_name",
            "config_name",
            "run_id",
            "timestamp",
            "algorithm",
            "metrics_path",
            "solution_path",
            "image_path",
            "config_json",
            "router_state_json",
        ]
        metric_keys = sorted(metrics.keys())
        fieldnames = base + metric_keys
        self._manifest_fieldnames = fieldnames
        return fieldnames

    def _ensure_manifest_header(self, fieldnames: List[str]) -> None:
        if not self.manifest_path.exists():
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()

    @staticmethod
    def _config_snapshot(config: RoutingConfiguration) -> Dict[str, Any]:
        return {
            "name": config.name,
            "algorithm": config.algorithm,
            "router_kwargs": dict(config.router_kwargs),
            "solve_kwargs": dict(config.solve_kwargs),
            "metadata": dict(config.metadata),
        }

    @staticmethod
    def _router_state(
        router_params: Mapping[str, Any],
        router_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for key, value in router_params.items():
            if key in router_kwargs and router_kwargs[key] == value:
                continue
            state[key] = value
        return state

    def _write_board_description(self) -> None:
        blocked_cells = sorted(self.board.blocked)
        if self.board_path.exists():
            return
        pins_payload: Dict[str, Dict[str, Any]] = {}
        for idx, net in enumerate(self.nets):
            src_pin = net.src_point
            dst_pin = net.dst_point
            pins_payload.setdefault(src_pin.name, {"coords": list(map(int, src_pin.coord))})
            pins_payload.setdefault(dst_pin.name, {"coords": list(map(int, dst_pin.coord))})

        payload = {
            "experiment": self.name,
            "rows": self.board.rows,
            "cols": self.board.cols,
            "blocked": [list(map(int, cell)) for cell in blocked_cells],
            "total_nets": len(self.nets),
            "pins": pins_payload,
            "nets": [
                {
                    "name": self._net_names[idx],
                    "src_pin": net.src_point.name,
                    "dst_pin": net.dst_point.name,
                    "src": list(map(int, net.src)),
                    "dst": list(map(int, net.dst)),
                }
                for idx, net in enumerate(self.nets)
            ],
            "created_at": self._timestamp(),
        }
        with self.board_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _build_net_names(self) -> List[str]:
        names: List[str] = []
        seen: Dict[str, int] = {}
        for idx, net in enumerate(self.nets):
            base = (net.name or f"net_{idx}").strip() or f"net_{idx}"
            count = seen.get(base, 0)
            if count:
                unique = f"{base}_{count}"
            else:
                unique = base
            seen[base] = count + 1
            names.append(unique)
        return names

    @classmethod
    def load_from_directory(cls, directory: Path | str) -> "Experiment":
        directory_path = Path(directory).expanduser()
        definition_file = directory_path / "experiment.json"
        if not definition_file.exists():
            raise FileNotFoundError(f"Experiment definition not found at {definition_file}")

        with definition_file.open("r", encoding="utf-8") as fh:
            definition = json.load(fh)

        name = definition.get("name") or directory_path.name
        board_file = definition.get("board_file", "board.json")
        board_path = directory_path / board_file
        if not board_path.exists():
            raise FileNotFoundError(f"Board description not found at {board_path}")

        with board_path.open("r", encoding="utf-8") as fh:
            board_data = json.load(fh)

        blocked_cells = {
            tuple(int(v) for v in cell)
            for cell in board_data.get("blocked", [])
        }
        board = GridBoard(
            rows=int(board_data["rows"]),
            cols=int(board_data["cols"]),
            blocked=blocked_cells,
        )

        pins_payload = board_data.get("pins", {})
        pins: Dict[str, Pin] = {}
        for pin_name, pin_info in pins_payload.items():
            coords_source = pin_info.get("coords") if isinstance(pin_info, dict) else pin_info
            if coords_source is None:
                raise ValueError(f"Missing coordinates for pin {pin_name}")
            coords_tuple = tuple(int(v) for v in coords_source)
            pins[pin_name] = Pin(pin_name, coords_tuple)

        nets_payload = board_data.get("nets", [])
        nets: List[Net] = []
        for idx, net_info in enumerate(nets_payload):
            net_name = net_info.get("name") or f"net_{idx}"
            src_pin_name = net_info.get("src_pin")
            dst_pin_name = net_info.get("dst_pin")

            if src_pin_name and src_pin_name in pins:
                src_pin = pins[src_pin_name]
            else:
                src_coords = tuple(int(v) for v in net_info.get("src", []))
                src_pin = Pin(src_pin_name or f"{net_name}_src", src_coords)
            if dst_pin_name and dst_pin_name in pins:
                dst_pin = pins[dst_pin_name]
            else:
                dst_coords = tuple(int(v) for v in net_info.get("dst", []))
                dst_pin = Pin(dst_pin_name or f"{net_name}_dst", dst_coords)

            nets.append(Net(src_pin, dst_pin, net_name))

        output_root = directory_path.parent
        experiment = cls(
            name=name,
            board=board,
            nets=nets,
            output_dir=output_root,
        )

        # Replace configurations using saved definitions
        experiment._configurations.clear()
        for cfg_data in definition.get("configs", []):
            config = RoutingConfiguration(
                name=cfg_data["name"],
                algorithm=cfg_data["algorithm"],
                router_kwargs=dict(cfg_data.get("router_kwargs", {})),
                solve_kwargs=dict(cfg_data.get("solve_kwargs", {})),
                metadata=dict(cfg_data.get("metadata", {})),
            )
            experiment._configurations[config.name] = config

        experiment._write_experiment_definition()
        return experiment

    @staticmethod
    def _generate_run_id() -> str:
        return uuid4().hex[:12]

    @staticmethod
    def _timestamp() -> str:
        # Use timezone-aware UTC to avoid deprecated utcnow()
        return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def generate_random_case(
        rows: int,
        cols: int,
        pin_count: int,
        net_count: int,
        *,
        seed: int | None = None,
    ) -> tuple[GridBoard, List[Net]]:
        """
        Build a random board and nets with spacing constraints.

        Pins are placed on layer 0 with at least one empty cell (Chebyshev distance >= 2)
        between any two pins. Nets are sampled by pairing pins (pins may participate in
        multiple nets).
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if pin_count < 2:
            raise ValueError("pin_count must be at least 2")
        if net_count <= 0:
            raise ValueError("net_count must be positive")

        rng = random.Random(seed)
        all_positions = [(r, c, 0) for r in range(rows) for c in range(cols)]
        rng.shuffle(all_positions)

        pins_coords: List[tuple[int, int, int]] = []
        for coord in all_positions:
            if all(
                max(abs(coord[0] - existing[0]), abs(coord[1] - existing[1])) >= 2
                for existing in pins_coords
            ):
                pins_coords.append(coord)
            if len(pins_coords) == pin_count:
                break

        if len(pins_coords) < pin_count:
            raise ValueError(
                "Unable to place requested number of pins with spacing constraints; "
                "consider reducing pin_count or increasing board size."
            )

        pins_list: List[Pin] = []
        pin_map: Dict[str, Pin] = {}
        for idx, coord in enumerate(pins_coords, start=1):
            name = f"P{idx}"
            pin = Pin(name, coord)
            pins_list.append(pin)
            pin_map[name] = pin

        nets: List[Net] = []
        for idx in range(net_count):
            src_pin, dst_pin = rng.sample(pins_list, 2)
            nets.append(Net(src_pin, dst_pin, f"Net_{idx + 1}"))

        board = GridBoard(rows, cols)
        return board, nets

    @staticmethod
    def load_all_metrics(root_dir: Path | str) -> "pd.DataFrame":
        """Load metrics_log.csv from all experiments under a root directory."""
        if pd is None:
            raise ImportError("pandas is required for load_all_metrics")
        root = Path(root_dir)
        frames: List["pd.DataFrame"] = []
        for csv_path in root.glob("*/metrics_log.csv"):
            df = pd.read_csv(csv_path)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
