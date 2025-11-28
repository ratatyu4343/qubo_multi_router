Usage
=====

Quick start
-----------

1. Install deps::

     pip install -r requirements.txt

2. Run tests to verify::

     pytest

3. Default analysis on existing metrics (``main3``)::

     python main.py

   Reads ``saved_experiments2`` and prints Beam/Wave/Quantum A* comparisons.

Scenarios from ``main.py``
--------------------------

Generate 1000 random 20x20 cases and run all configs::

   python -c "import main; main.main()"

Compare configs using metrics in ``saved_experiments``::

   python -c "import main; main.main2()"

Available configurations
------------------------

- ``classical_beam`` — Beam Router (1 layer).
- ``classical_wave`` — Wave Router (1 layer).
- ``quantum_astar`` — MultiNet Quantum Router with A* candidates.

Quantum router modes
--------------------

- ``use_dwave=False`` — simulated annealer (`neal`) heuristic, no quantum hardware.
- ``use_dwave=True`` — placeholder; real D-Wave integration is not implemented here.
