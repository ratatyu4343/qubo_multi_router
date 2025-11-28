from pcb_routing.utilities import path_cost, path_turns


def test_path_cost_and_turns():
    p_straight = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]
    assert path_cost(p_straight) == 3
    assert path_turns(p_straight) == 0

    p_turn = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]
    assert path_cost(p_turn) == 3
    assert path_turns(p_turn) == 1
