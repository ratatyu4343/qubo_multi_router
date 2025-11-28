from pcb_routing import GridBoard


def test_grid_neighbors_in_bounds():
    b = GridBoard(3, 3)
    nbs = set(b.neighbors((1, 1, 0)))
    assert nbs == {(0, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0)}

    # corners
    n00 = set(b.neighbors((0, 0, 1)))
    assert n00 == {(1, 0, 1), (0, 1, 1)}
