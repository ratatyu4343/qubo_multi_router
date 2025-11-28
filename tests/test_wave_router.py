from pcb_routing import GridBoard, Net, Pin
from pcb_routing.WaveRouter import WaveRouter
from pcb_routing.BeamRouter import BeamRouter


def test_wave_router_simple():
    B = GridBoard(5, 5)
    a = Pin("A", (0, 0, 0))
    b = Pin("B", (0, 3, 0))
    net = Net(a, b, "N1")
    wr = WaveRouter(B, max_layers=2)
    sol = wr.route_all([net])
    assert 0 in sol
    p = sol[0]
    assert all(len(pt) == 3 for pt in p)
    # 3 steps minimum in straight line
    assert len(p) >= 4


def test_beam_router_simplifies_path():
    B = GridBoard(3, 3)
    router = BeamRouter(B, max_layers=1)
    raw_path = [(0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 1, 0), (0, 2, 0)]
    simplified = router._simplify_path(raw_path)
    assert simplified == [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
