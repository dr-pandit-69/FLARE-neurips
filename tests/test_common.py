from pathlib import Path

from neurips_grid.common import CONFIG, load_config, stable_hash
from neurips_grid.worker import cells_from_config


def test_configuration_is_complete():
    config = load_config(CONFIG)
    assert len([spec for spec in config["models"].values() if not spec.get("conditional")]) == 6
    assert set(config["tiers"]) == {"clean", "sparse", "noisy", "stale", "conflicting"}
    cells = {cell.name: cell for cell in cells_from_config(config)}
    assert {"primary_A0", "topology_ood_A0", "true_state", "schema_only"} <= set(cells)
    assert cells["no_memory"].use_memory is False
    assert "simulate_switch" not in cells["no_simulator"].allowed_verbs


def test_stable_hash_ignores_mapping_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

