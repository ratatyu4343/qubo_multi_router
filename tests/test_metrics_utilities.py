from pcb_routing import GridBoard, Net, Pin
from pcb_routing.metrics import total_length, total_turns, routed_count, layers_used
from pcb_routing.utilities import path_cost, path_turns


def test_path_cost_and_turns_multilayer():
    path = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
        (1, 2, 1),  # layer change should not count as turn in 2D projection
        (2, 2, 1),
    ]
    assert path_cost(path) == 5
    assert path_turns(path) == 1


def test_metric_helpers_basic():
    board = GridBoard(4, 4)
    net1 = Net(Pin("A", (0, 0, 0)), Pin("B", (0, 3, 0)), "N1")
    net2 = Net(Pin("C", (3, 0, 0)), Pin("D", (3, 3, 0)), "N2")

    sol = {
        0: [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)],
        1: [(3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 3, 0)],
    }

    assert routed_count(sol) == 2
    assert total_length(sol) == 6
    assert total_turns(sol) == 0
    assert layers_used(sol) == 1
