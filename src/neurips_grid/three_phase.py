from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandapower as pp

from .common import REPO, ROOT, atomic_json, markdown_report, now_utc


def evaluate(path: Path) -> dict:
    net = pp.from_json(str(path))
    required = {
        "ext_grid": ("s_sc_max_mva", "rx_max", "x0x_max", "r0x0_max"),
        "line": ("r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km"),
        "trafo": ("vk0_percent", "vkr0_percent", "mag0_percent", "mag0_rx", "si0_hv_partial"),
    }
    missing = {}
    for table_name, columns in required.items():
        table = getattr(net, table_name, None)
        if table is None or not len(table):
            continue
        absent = [column for column in columns if column not in table.columns or table[column].isna().any()]
        if absent:
            missing[table_name] = absent
    try:
        trial = copy.deepcopy(net)
        pp.runpp_3ph(trial, numba=True)
        vm_columns = [name for name in trial.res_bus_3ph.columns if name.startswith("vm_")]
        values = trial.res_bus_3ph[vm_columns].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return {
            "grid_id": path.stem,
            "success": bool(trial.converged),
            "missing_native_parameters": missing,
            "min_vm_pu": float(finite.min()) if finite.size else None,
            "max_vm_pu": float(finite.max()) if finite.size else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "grid_id": path.stem,
            "success": False,
            "missing_native_parameters": missing,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grids",
        type=Path,
        default=REPO / "data/processed/substrate_v1/distribution_grids",
    )
    args = parser.parse_args(argv)
    grid_ids = json.loads((args.grids / "grid_manifest.json").read_text())["grid_ids"]
    rows = [evaluate(args.grids / f"{grid_id}.json") for grid_id in grid_ids]
    result = {
        "schema_version": "neurips-grid-three-phase-feasibility-v1",
        "created_utc": now_utc(),
        "n_grids": len(rows),
        "n_success": sum(row["success"] for row in rows),
        "eligible_for_three_phase_subset": sum(row["success"] for row in rows) > 0,
        "rows": rows,
        "policy": "No zero-sequence or short-circuit parameters are imputed.",
    }
    output = ROOT / "artifacts/physics/three_phase_feasibility.json"
    atomic_json(output, result)
    report = markdown_report(
        "three_phase",
        "native_parameters",
        [
            "# Native three-phase feasibility",
            "",
            f"- Grids attempted: {result['n_grids']}",
            f"- Successful without imputation: {result['n_success']}",
            f"- Three-phase subset eligible: {result['eligible_for_three_phase_subset']}",
            "- Policy: no missing electrical parameters are fabricated.",
        ],
    )
    print(json.dumps({"result": result, "output": str(output), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
