from pathlib import Path

import yaml

from neurips_grid.common import CONFIG, ROOT


def test_project_paths_are_checkout_relative():
    assert ROOT == Path(__file__).resolve().parents[1]
    assert CONFIG == ROOT / "configs/experiment.yaml"


def test_configured_inputs_are_relative_to_the_checkout():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = [spec["path"] for spec in config["panels"].values()]
    paths.extend(
        [
            config["ood"]["source_events"],
            config["ood"]["source_grids"],
            config["ood"]["source_series"],
            config["ood"]["observation_model"],
        ]
    )
    assert all(not Path(value).is_absolute() for value in paths)


def test_runtime_output_is_not_part_of_the_source_tree():
    assert not (ROOT / "runs").exists()
    assert not (ROOT / "outputs").exists()
    assert not list(ROOT.rglob("*.log"))
