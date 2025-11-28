from pathlib import Path

from experiments import Experiment, RoutingConfiguration
from pcb_routing import GridBoard, Net, Pin


def test_experiment_run_all_creates_outputs(tmp_path: Path):
    board = GridBoard(3, 3)
    net = Net(Pin("A", (0, 0, 0)), Pin("B", (0, 2, 0)), "N1")
    exp = Experiment(
        name="tiny_exp",
        board=board,
        nets=[net],
        output_dir=tmp_path,
    )
    exp.add_configuration(
        RoutingConfiguration(
            name="wave",
            algorithm="wave",
            router_kwargs={"max_layers": 1},
        )
    )

    result = exp.run_all(save_images=False)
    assert result and result[0].config_name == "wave"

    manifest = tmp_path / "tiny_exp" / "metrics_log.csv"
    metrics_dir = tmp_path / "tiny_exp" / "metrics"
    solutions_dir = tmp_path / "tiny_exp" / "solutions"

    assert manifest.exists()
    assert any(metrics_dir.glob("*.json"))
    assert any(solutions_dir.glob("*.json"))

    df = Experiment.load_all_metrics(tmp_path)
    assert not df.empty
    assert set(df["config_name"]) == {"wave"}
