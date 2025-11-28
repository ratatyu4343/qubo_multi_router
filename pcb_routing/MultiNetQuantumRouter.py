from __future__ import annotations

from .Objects import GridBoard, Net, Path, Pin
from .utilities import path_cost
from typing import Any, Dict, List
from .QuantumAnnealer import QuantumAnnealer
from .candidates import REGISTRY as CANDIDATE_REGISTRY
import logging

logger = logging.getLogger(__name__)

class MultiNetQuantumRouter:
    """ Multi-net router using candidate generation + QUBO solving.
    Parameters:
    - board: GridBoard instance representing the PCB grid.
    - conflict_penalty: Penalty for overlapping paths.
    - selection_reward: Reward for selecting a path (if None, set to max_cost + 1).
    - num_reads: Number of samples to draw.
    - turn_penalty: Penalty per turn in a path.
    - annealer: Optional QuantumAnnealer instance; if None, a default one is created.
    """

    def __init__(self, board: GridBoard,
                 num_reads=300,
                 annealer=None,
                 cost_weight=1.0,
                 turn_weight=0.5,
                 onepath_weight=10.0,
                 conflict_weight=1.0,
                 P_H=1000.0,
                 use_dwave=False,
                 seed=None):
        self.board = board
        self.annealer = annealer or QuantumAnnealer(num_reads=num_reads, cost_weight=cost_weight,
                                                    turn_weight=turn_weight, onepath_weight=onepath_weight,
                                                    conflict_weight=conflict_weight, P_H=P_H, use_dwave=use_dwave,
                                                    seed=seed)
        self.seed = seed
        # Initialize generator registry (can be extended at runtime)
        self.genmap = dict(CANDIDATE_REGISTRY)
        self._params = {
            "num_reads": num_reads,
            "cost_weight": cost_weight,
            "turn_weight": turn_weight,
            "onepath_weight": onepath_weight,
            "conflict_weight": conflict_weight,
            "P_H": P_H,
            "use_dwave": use_dwave,
            "seed": seed,
            "annealer": (
                annealer.__class__.__name__ if annealer is not None else self.annealer.__class__.__name__
            ),
            "default_candidate_method": "astar",
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of router parameters."""
        return dict(self._params)

    def generate_candidates(self, net: Net, limit=30, method: str | None = None) -> List[Path]:
        """Generate candidate paths for a given net using the registered generator (default astar)."""
        method = method or "astar"
        Gcls = self.genmap.get(method, None)
        if Gcls is None:
            # Fallback to astar from registry
            from .candidates.astar import AStarGenerator as Gcls  # type: ignore
        # Try to pass seed if generator supports it, otherwise fallback
        seed = getattr(self, 'seed', None)
        try:
            if seed is not None:
                G = Gcls(self.board, net, seed=seed)
            else:
                G = Gcls(self.board, net)
        except TypeError:
            try:
                G = Gcls(self.board, net, seed)
            except TypeError:
                G = Gcls(self.board, net)
                if seed is not None:
                    try:
                        import random
                        random.seed(int(seed))
                        setattr(G, 'seed', int(seed))
                    except Exception:
                        pass
        return list(G.generate(limit=limit))

    def register_generator(self, name: str, cls):
        """Register a candidate generator class under a name."""
        self.genmap[name] = cls

    def solve_all(self, nets: List[Net], per_net_limit=20, allow_conflicts=False,
                  allow_not_having_path=False, method: str | None = None):
        '''
        Solve routing for all nets using candidate generation + QUBO solving.

        This method:
        1. Generates a set of candidate paths for each net, temporarily blocking other net pins.
        2. Constructs a QUBO model to select exactly one path per net while minimizing total path length
        and penalizing cell conflicts between paths.
        3. Solves the QUBO to find the optimal set of non-conflicting paths.

        Args:
            nets (List[Net]): The list of nets (source/destination pairs) to be routed.
            per_net_limit (int): Maximum number of candidates to generate per net.
            allow_conflicts (bool): If True, allows conflicting paths in the solution.
            method (str | None): Optional override for candidate generation.

        Returns:
            dict: A dictionary mapping net indices (0 to len(nets)-1) to their chosen path (a list of cells),
                or an empty dict if no feasible solution (according to the annealer) is found.
        '''

        all_candidates = []  # Stores (global_candidate_index, net_index, path)
        global_cand_idx = 0
        generator_method = method or "astar"
        
        # Identify all pin cells to avoid routing through other nets' pins
        pin_cells = {net.src for net in nets} | {net.dst for net in nets}

        for net_idx, net in enumerate(nets):
            # 2) Temporarily block all other net's pins (except the current net's src/dst)
            pins_to_block = {(p[0], p[1], p[2]) for p in (pin_cells - {net.src, net.dst})}
            saved_blocked_cells = set(self.board.blocked)
            self.board.blocked |= pins_to_block

            # 3) Generate candidates for the current net
            cands = self.generate_candidates(net, limit=per_net_limit, method=generator_method)
            
            # 4) Restore the original blocked cells
            self.board.blocked = saved_blocked_cells

            # 5) Add all candidates to the global list
            for path in cands:
                # path is the list of cells (coords)
                all_candidates.append((global_cand_idx, net_idx, path))
                global_cand_idx += 1
                
        if not all_candidates:
            logger.warning("No candidate paths were generated for any net.")
            return {}

        # --- 6) QUBO Model Construction and Solving ---
        
        # The build_qubo method must correctly implement the objective function and constraints:
        # Objective: Minimize total path length (sum of path costs).
        # Constraint 1 (Hard): Exactly one path must be chosen for each net. (e.g., Penalty_A * (1 - sum(x_i))^2 for each net)
        # Constraint 2 (Hard): No conflicts (overlapping cells) between chosen paths. (e.g., Penalty_B * sum(x_i * x_j) for conflicting paths i, j)
        # Penalties A and B must be large enough to enforce the constraints (e.g., larger than the maximum possible path length * number of nets).
        
        # 6) Build the QUBO model
        build = getattr(self.annealer, 'build_qubo_fast', self.annealer.build_qubo)
        bqm = build(all_candidates)

        # 7) Solve the QUBO (using simulator or D-Wave)
        # The solver returns a collection of low-energy samples.
        samples = self.annealer.solve(bqm)
        if not samples:
            logger.warning("No samples returned from the annealer.")
            return {}
        
        # --- 8) Solution Evaluation and Selection ---
        best_sol_paths = {}
        best_energy = float('inf')

        for sample, energy in samples.data(['sample', 'energy']):
            current_net_paths = {}
            chosen_candidate_indices = {idx for idx, _, _ in all_candidates if sample.get(idx, 0) == 1}
            nets_with_path = set()
            used_cells = set()
            conflict = False

            for gidx in chosen_candidate_indices:
                try:
                    _, net_idx, path = next(t for t in all_candidates if t[0] == gidx)
                    if net_idx in current_net_paths:
                        conflict = True
                        break
                    # Check for cell overlap
                    if not allow_conflicts and used_cells & set(path):
                        conflict = True
                        break
                    used_cells |= set(path)
                    current_net_paths[net_idx] = path
                    nets_with_path.add(net_idx)
                except StopIteration:
                    logger.warning("Candidate index %s not found in all_candidates.", gidx)
                    conflict = True
                    break
                
            # Ensure all nets have a path
            if len(nets_with_path) < len(nets) and not allow_not_having_path:
                logger.warning("Not all nets have a path in this solution.")
                continue
            
            # If conflicts are not allowed, skip solutions with conflicts
            if conflict and not allow_conflicts:
                continue
        
            if energy < best_energy:
                best_energy = energy
                best_sol_paths = current_net_paths

        return best_sol_paths

    def solve_multilayer(self, nets, method: str | None = None, per_net_limit=30, max_layers=5,
                         allow_conflicts=True, allow_not_having_path=True):
        """Route nets iteratively across multiple layers by invoking solve_all per layer."""
        max_layers = max(1, int(max_layers))
        unrouted = set(range(len(nets)))
        all_paths = {}
        pin_usage = {}  # pin_coord -> set(net_idx)
        for idx, net in enumerate(nets):
            pin_usage.setdefault(net.src, set()).add(idx)
            pin_usage.setdefault(net.dst, set()).add(idx)

        initial_blocked = set(self.board.blocked)

        for layer in range(max_layers):
            # Stabilize ordering: map local indices to deterministic global indices
            idx_map = sorted(unrouted)
            if not idx_map:
                break
            nets_to_route = []
            for global_idx in idx_map:
                original = nets[global_idx]
                src_coord = original.src
                src_pin = Pin(
                    original.src_point.name,
                    (src_coord[0], src_coord[1], layer)
                )
                dst_coord = original.dst
                dst_pin = Pin(
                    original.dst_point.name,
                    (dst_coord[0], dst_coord[1], layer)
                )
                nets_to_route.append(Net(src_pin, dst_pin, original.name))
            if not nets_to_route:
                break

            # Формуємо список пінів, які можна блокувати (ті, для яких всі нети вже прокладені)
            pins_to_block = set()
            for pin, net_idxs in pin_usage.items():
                if net_idxs <= set(all_paths.keys()):
                    for layer_idx in range(max_layers):
                        pins_to_block.add((pin[0], pin[1], layer_idx))
            self.board.blocked = initial_blocked | pins_to_block

            sol = self.solve_all(
                nets_to_route, method=method, per_net_limit=per_net_limit,
                allow_conflicts=allow_conflicts, allow_not_having_path=allow_not_having_path
            )

            for local_idx, path in sol.items():
                global_idx = idx_map[local_idx]
                path3d = [(p[0], p[1], layer) for p in path]
                all_paths[global_idx] = path3d

            routed_global = {idx_map[i] for i in sol.keys()}
            unrouted -= routed_global
            if not unrouted:
                break

        self.board.blocked = initial_blocked
        return all_paths
