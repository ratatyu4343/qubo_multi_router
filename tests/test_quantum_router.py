import pytest

from pcb_routing import GridBoard, Net, Pin
from pcb_routing.MultiNetQuantumRouter import MultiNetQuantumRouter


def _build_two_nets_board():
    B = GridBoard(4, 4)
    p1 = Pin("A", (0, 0, 0))
    p2 = Pin("B", (0, 3, 0))
    p3 = Pin("C", (3, 0, 0))
    p4 = Pin("D", (3, 3, 0))
    return B, [Net(p1, p2, "N1"), Net(p3, p4, "N2")]


def test_quantum_router_no_conflicts_prefers_disjoint_paths():
    B, nets = _build_two_nets_board()
    router = MultiNetQuantumRouter(
        B,
        num_reads=30,
        turn_weight=0.0,
        conflict_weight=100.0,
        onepath_weight=10.0,
        P_H=10.0,
        use_dwave=False,
        seed=42,
    )
    sol = router.solve_all(
        nets,
        per_net_limit=4,
        method="astar",
        allow_conflicts=False,
        allow_not_having_path=False,
    )
    assert set(sol.keys()) == {0, 1}
    cells_n1 = set(sol[0])
    cells_n2 = set(sol[1])
    assert cells_n1.isdisjoint(cells_n2)


def test_quantum_router_respects_per_net_limit_zero_paths_when_too_low():
    B, nets = _build_two_nets_board()
    router = MultiNetQuantumRouter(B, num_reads=10, use_dwave=False, seed=7)
    sol = router.solve_all(
        nets,
        per_net_limit=0,
        method="astar",
        allow_conflicts=True,
        allow_not_having_path=True,
    )
    # With zero candidates per net, solution is empty
    assert sol == {}


def test_quantum_router_deterministic_with_seed():
    B, nets = _build_two_nets_board()
    kwargs = dict(
        num_reads=20,
        use_dwave=False,
        seed=99,
        turn_weight=0.0,
        conflict_weight=10.0,
        onepath_weight=10.0,
        P_H=10.0,
    )
    sol1 = MultiNetQuantumRouter(B, **kwargs).solve_all(
        nets, per_net_limit=3, method="astar", allow_conflicts=True, allow_not_having_path=True
    )
    sol2 = MultiNetQuantumRouter(B, **kwargs).solve_all(
        nets, per_net_limit=3, method="astar", allow_conflicts=True, allow_not_having_path=True
    )
    assert sol1 == sol2
