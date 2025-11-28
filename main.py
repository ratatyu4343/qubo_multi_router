from __future__ import annotations
from pathlib import Path
import random
from experiments import Experiment, RoutingConfiguration
from pcb_routing import GridBoard, Net, Pin
import logging
logging.basicConfig(level=logging.ERROR)

def register_configurations(experiment: Experiment) -> None:
    """Register baseline routing configurations (beam, wave, quantum A*) for an experiment."""
    experiment.add_configuration(
        RoutingConfiguration(
            name="classical_beam",
            algorithm="beam",
            router_kwargs={'max_layers': 1},
        )
    )
    experiment.add_configuration(
        RoutingConfiguration(
            name="classical_wave",
            algorithm="wave",
            router_kwargs={'max_layers': 1},
        )
    )
    base_quantum_kwargs = {
        "num_reads": 250,
        "cost_weight": 1.0,
        "turn_weight": 0.001,
        "onepath_weight": 10.0,
        "conflict_weight": 100.0,
        "P_H": 100.0,
        "use_dwave": False,
    }
    experiment.add_configuration(
        RoutingConfiguration(
            name=f"quantum_astar",
            algorithm="quantum",
            router_kwargs=dict(base_quantum_kwargs),
            solve_kwargs={"method": "astar", "per_net_limit": 30, 'max_layers': 1},
            metadata={
                "description": "Quantum annealing sweep over per-net candidate count",
                "num_reads": base_quantum_kwargs["num_reads"],
                "candidate_method": "astar",
            },
        )
    )

def compare_two_configs_by_routed_count(config_a: str, config_b: str, df) -> float:
    """Compute average % delta in routed share between two configs across experiments."""
    subset = df[df["config_name"].isin([config_a, config_b])]
    pivot = (
        subset.pivot_table(
            index="experiment_name",
            columns="config_name",
            values=["routed_nets", "total_nets"],
            aggfunc="mean",
        )
    )
    pivot[("share", config_a)] = pivot[("routed_nets", config_a)] / pivot[("total_nets", config_a)]
    pivot[("share", config_b)] = pivot[("routed_nets", config_b)] / pivot[("total_nets", config_b)]
    pivot["share_ratio_pct"] = (pivot[("share", config_b)] / pivot[("share", config_a)] - 1) * 100
    return pivot['share_ratio_pct'].mean()

def compare_two_configs_by_length(config_a: str, config_b: str, df) -> float:
    """Compute average % delta in total length, comparing only rows where routed counts match."""
    subset = df[df["config_name"].isin([config_a, config_b])]
    pivot = (
        subset.pivot_table(
            index="experiment_name",
            columns="config_name",
            values=["routed_nets", "total_length"],
            aggfunc="mean",
        )
    )
    pivot = pivot[pivot[("routed_nets", config_a)] == pivot[("routed_nets", config_b)]]
    pivot["length_ratio_pct"] = (pivot[("total_length", config_b)] / pivot[("total_length", config_a)] - 1) * 100
    return pivot["length_ratio_pct"].mean()

def main() -> None:
    """Generate random 20x20 boards and run all configs, storing outputs under saved_experiments2/."""
    expeinets_boards = []
    for i in range(1000):
        width = 20
        height = 20
        num_pins = random.randint(2, 20)
        num_nets = random.randint(2, 30)
        expeinets_boards.append(Experiment.generate_random_case(width, height, num_pins, num_nets))
    
    for experiment_id, board_nets in enumerate(expeinets_boards):
        experiment = Experiment(
            name=f"compare_routers_20x20_{experiment_id}",
            board=board_nets[0],
            nets=board_nets[1],
            output_dir=Path("saved_experiments2"),
        )
        register_configurations(experiment)
        experiment.run_all(save_images=True)
        print(f"Completed experiment {experiment_id}")

def main2() -> None:
    """Compare Beam/Wave/Quantum A* using metrics already stored in saved_experiments."""
    df = Experiment.load_all_metrics('saved_experiments')
    print("Beam vs Wave by Routed Count (%):", compare_two_configs_by_routed_count("classical_beam", "classical_wave", df))
    print("Quantum A* vs Beam by Routed Count (%):", compare_two_configs_by_routed_count("classical_beam", "quantum_astar", df))
    print("Quantum A* vs Wave by Routed Count (%):", compare_two_configs_by_routed_count("classical_wave", "quantum_astar", df))

    print("Beam vs Wave by Length (%):", compare_two_configs_by_length("classical_beam", "classical_wave", df))
    print("Quantum A* vs Beam by Length (%):", compare_two_configs_by_length("classical_beam", "quantum_astar", df))
    print("Quantum A* vs Wave by Length (%):", compare_two_configs_by_length("classical_wave", "quantum_astar", df))

if __name__ == "__main__":
    main2()
