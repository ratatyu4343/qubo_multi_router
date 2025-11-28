from typing import List, Tuple, Dict
from .Objects import Path
from .utilities import path_cost, path_turns

from neal import SimulatedAnnealingSampler
import dimod


class QuantumAnnealer:
    """
    Клас QuantumAnnealer будує i розв’язує задачу трасування y вигляді QUBO-моделі.

    Гамільтоніан (функція енергії) має вигляд:
        H = w_cost * H_cost + w_turn * H_turn + w_onepath * H_onepath + w_conflict * H_conflict

    Де:
        - H_cost: мінімізує сумарну довжину шляхів.
        - H_turn: мінімізує кількість поворотів y вибраних шляхах.
        - H_onepath: штраф за вибір більше ніж одного шляху для одного нету.
        - H_conflict: штраф за перетин шляхів між різними нетами.

    Ваги w_* визначають відносну важливість кожного доданку.

    Параметри:
        conflict_weight  - коефіцієнт w_conflict.
        cost_weight      - коефіцієнт w_cost.
        turn_weight      - коефіцієнт w_turn.
        onepath_weight   - коефіцієнт w_onepath.
        P_H              - базовий штраф для жорстких обмежень.
        num_reads        - кількість семплів для відпалу.
        use_dwave        - якщо True, використовується реальний D-Wave.
    """

    def __init__(
        self,
        cost_weight=1.0,
        turn_weight=0.5,
        onepath_weight=10.0,
        conflict_weight=1.0,
        P_H=1000.0,
        num_reads=300,
        use_dwave=False,
        seed=None
    ):
        self.cost_weight = cost_weight
        self.turn_weight = turn_weight
        self.onepath_weight = onepath_weight
        self.conflict_weight = conflict_weight
        self.P_H = P_H
        self.num_reads = num_reads
        self.use_dwave = use_dwave
        self.seed = seed

        if use_dwave:
            from dwave.system import DWaveSampler, EmbeddingComposite
            self.sampler = EmbeddingComposite(DWaveSampler())
        else:
            self.sampler = SimulatedAnnealingSampler()

    def build_qubo(self, all_candidates: List[Tuple[int, int, Path]]) -> dimod.BinaryQuadraticModel:
        """
        Побудова QUBO для задачі трасування.

        Кожна змінна x_i відповідає кандидату-шляху:
            x_i = 1, якщо шлях вибрано; 0 — інакше.

        Гамільтоніан:
            H = w_cost * H_cost + w_turn * H_turn + w_onepath * H_onepath + w_conflict * H_conflict

        Де:
            H_cost     = Σ_i (length_i * x_i)
            H_turn     = Σ_i (turns_i * x_i)
            H_onepath  = Σ_net (Σ_i∈net x_i - 1)²
            H_conflict = Σ_(i<j, conflict(i,j)) x_i * x_j

        Повертає:
            dimod.BinaryQuadraticModel (BQM) модель, готову до відпалу.
        """

        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}
        net_map: Dict[int, List[int]] = {}

        # === H_cost + H_turn ===
        # Вартість та кількість поворотів додаються до лінійних коефіцієнтів.
        for idx, net_idx, path in all_candidates:
            L = float(path_cost(path))
            T = float(path_turns(path))
            linear[idx] = self.cost_weight * L + self.turn_weight * T
            net_map.setdefault(net_idx, []).append(idx)

        # === H_onepath ===
        # (Σ_i∈net x_i - 1)² = Σ_i x_i² - 2Σ_i x_i + 2Σ_{i<j} x_i x_j + 1
        # x_i² = x_i для бінарних змінних, тому константний доданок можна ігнорувати.
        for net_idx, indices in net_map.items():
            for i in indices:
                linear[i] = linear.get(i, 0.0) + self.onepath_weight * self.P_H  # Σ_i x_i²
                linear[i] = linear.get(i, 0.0) - 2 * self.onepath_weight * self.P_H  # -2Σ_i x_i
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    quadratic[(indices[i], indices[j])] = (
                        quadratic.get((indices[i], indices[j]), 0.0)
                        + 2 * self.onepath_weight * self.P_H
                    )

        # === H_conflict ===
        # Якщо два шляхи з різних нетів мають спільні координати — штраф,
        # але НЕ штрафуємо, якщо це спільний src або dst для обох нетів.
        num_candidates = len(all_candidates)
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                idx_i, net_i, path_i = all_candidates[i]
                idx_j, net_j, path_j = all_candidates[j]
                if net_i == net_j:
                    continue
                # Знаходимо спільні точки
                overlap = set(path_i) & set(path_j)
                if not overlap:
                    continue
                # Дістаємо початкові та кінцеві точки для обох шляхів
                src_i, dst_i = path_i[0], path_i[-1]
                src_j, dst_j = path_j[0], path_j[-1]
                # Якщо всі точки перетину — це спільний src або dst для обох нетів, НЕ штрафуємо
                only_pins = all(
                    (pt == src_i and pt == src_j) or (pt == dst_i and pt == dst_j)
                    for pt in overlap
                )
                if only_pins:
                    continue  # не штрафуємо за перетин у піннах
                # Інакше — штрафуємо
                quadratic[(idx_i, idx_j)] = quadratic.get((idx_i, idx_j), 0.0) + self.conflict_weight * self.P_H

        # === Нормалізація коефіцієнтів для стабільності ===
        all_vals = [abs(v) for v in linear.values()] + [abs(v) for v in quadratic.values()]
        max_val = max(all_vals) if all_vals else 1.0
        scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for k in linear:
            linear[k] *= scale
        for k in quadratic:
            quadratic[k] *= scale

        # === Побудова BQM ===
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset=0.0, vartype=dimod.BINARY)
        return bqm

    def solve(self, bqm: dimod.BinaryQuadraticModel):
        """Solve the provided BQM with the configured sampler and return a sampleset."""
        kwargs = {"num_reads": self.num_reads}
        if self.seed is not None and not self.use_dwave:
            try:
                kwargs["seed"] = int(self.seed)
            except Exception:
                pass
        return self.sampler.sample(bqm, **kwargs)

    # Optimized alternative to build_qubo using cell-indexed conflict construction
    def build_qubo_fast(self, all_candidates: List[Tuple[int, int, Path]]) -> dimod.BinaryQuadraticModel:
        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}
        net_map: Dict[int, List[int]] = {}

        # Cache metadata per candidate: (idx, net_idx, path, cells_set, src, dst, L, T)
        meta = []
        for idx, net_idx, path in all_candidates:
            L = float(path_cost(path))
            T = path_turns(path)
            linear[idx] = self.cost_weight * L + self.turn_weight * T
            net_map.setdefault(net_idx, []).append(idx)
            meta.append((idx, net_idx, path, set(path), path[0], path[-1], L, T))

        # One-path constraint per net
        for net_idx, indices in net_map.items():
            for i in indices:
                linear[i] = linear.get(i, 0.0) + self.onepath_weight * self.P_H
                linear[i] = linear.get(i, 0.0) - 2 * self.onepath_weight * self.P_H
            for a_i in range(len(indices)):
                for b_i in range(a_i + 1, len(indices)):
                    a, b = indices[a_i], indices[b_i]
                    if a > b:
                        a, b = b, a
                    quadratic[(a, b)] = quadratic.get((a, b), 0.0) + 2 * self.onepath_weight * self.P_H

        # Conflict penalties: index candidates by grid cell; add each pair once
        cell_to_ids: Dict[Tuple, List[int]] = {}
        for k, (idx, net_idx, path, cells, src, dst, _, _) in enumerate(meta):
            for cell in cells:
                cell_to_ids.setdefault(cell, []).append(k)

        seen_pairs = set()
        for cell, ids in cell_to_ids.items():
            if len(ids) < 2:
                continue
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i = ids[a]
                    j = ids[b]
                    idx_i, net_i, _, _, src_i, dst_i, _, _ = meta[i]
                    idx_j, net_j, _, _, src_j, dst_j, _, _ = meta[j]
                    if net_i == net_j:
                        continue
                    key = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
                    if key in seen_pairs:
                        continue
                    # Allow overlap at shared identical endpoints only
                    only_pins = ((cell == src_i and cell == src_j) or (cell == dst_i and cell == dst_j))
                    if only_pins:
                        continue
                    quadratic[key] = quadratic.get(key, 0.0) + self.conflict_weight * self.P_H
                    seen_pairs.add(key)

        # Normalize magnitudes
        all_vals = [abs(v) for v in linear.values()] + [abs(v) for v in quadratic.values()]
        max_val = max(all_vals) if all_vals else 1.0
        scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for k in linear:
            linear[k] *= scale
        for k in quadratic:
            quadratic[k] *= scale

        # Build BQM
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset=0.0, vartype=dimod.BINARY)
        return bqm
