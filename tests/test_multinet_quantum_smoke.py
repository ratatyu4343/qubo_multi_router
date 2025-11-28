import pytest

from pcb_routing import GridBoard, Net, Pin


def test_multinet_quantum_smoke():
    try:
        from pcb_routing import MultiNetQuantumRouter
    except Exception:
        pytest.skip("Optional quantum dependencies not installed")

    B = GridBoard(5, 5)
    pins = [
        Pin("A", (0, 0, 0)),
        Pin("B", (0, 4, 0)),
        Pin("C", (4, 0, 0)),
        Pin("D", (4, 4, 0)),
    ]
    nets = [Net(pins[0], pins[1], "N1"), Net(pins[2], pins[3], "N2")]

    qr = MultiNetQuantumRouter(B, num_reads=20, cost_weight=1.0, turn_weight=0.0,
                               conflict_weight=10.0, onepath_weight=10.0, P_H=10.0,
                               use_dwave=False, seed=123)
    sol = qr.solve_all(nets, per_net_limit=3, method="astar",
                       allow_conflicts=True, allow_not_having_path=True)
    assert isinstance(sol, dict)
    for path in sol.values():
        assert all(len(p) == 3 for p in path)
