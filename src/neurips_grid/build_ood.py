from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp

from .common import (
    CONFIG,
    REPO,
    ROOT,
    atomic_json,
    atomic_text,
    load_config,
    markdown_report,
    now_utc,
    resolve_repo_path,
    sha256_file,
    stable_hash,
)

sys.path.insert(0, str(REPO / "src"))

from gridagent.oracle.restoration_oracle import (  # noqa: E402
    build_oracle_labels,
    calibrate_operating_point,
    hourly_load_profile,
    load_bin_factors,
)
from gridagent.oracle.reconfig_search import isolate_fault  # noqa: E402
from gridagent.simulation.grid_bank import (  # noqa: E402
    _switch_bus_pair,
    collapsed_graph,
    n_independent_loops,
    unsupplied,
)
from gridagent.simulation.pf_feasibility import pf_feasible  # noqa: E402

PANEL_NAMES = ("topology_ood", "generation_ood", "load_ood")


def _scale_joint(net, factor: float) -> None:
    for column in ("p_mw", "q_mvar"):
        if column in net.load.columns:
            net.load[column] *= float(factor)
        if len(net.sgen) and column in net.sgen.columns:
            net.sgen[column] *= float(factor)


def _scale_generation(net, factor: float) -> None:
    for column in ("p_mw", "q_mvar"):
        if len(net.sgen) and column in net.sgen.columns:
            net.sgen[column] *= float(factor)


def _load_only_factor(net, target_pct: float) -> float:
    lo, hi = 0.02, 4.0
    for _ in range(22):
        mid = (lo + hi) / 2.0
        trial = copy.deepcopy(net)
        for column in ("p_mw", "q_mvar"):
            if column in trial.load.columns:
                trial.load[column] *= mid
        ok, metrics = pf_feasible(trial)
        loading = max(
            float(metrics.get("max_line_loading_pct") or 0.0),
            float(metrics.get("max_trafo_loading_pct") or 0.0),
        )
        if ok and loading <= target_pct:
            lo = mid
        else:
            hi = mid
    return round(lo, 5)


def _cycle_edges(graph: nx.Graph) -> set[frozenset[int]]:
    cycles = nx.cycle_basis(graph)
    if len(cycles) != 1:
        return set()
    cycle = cycles[0]
    return {
        frozenset((int(left), int(right)))
        for left, right in zip(cycle, cycle[1:] + cycle[:1])
    }


def relocate_open_point(net, switch_rows: pd.DataFrame, token: str) -> tuple[object, pd.DataFrame, dict] | None:
    """Return one deterministic, validated alternative radial operating state."""
    tie_rows = switch_rows[
        (switch_rows["type"] == "tie") & (~switch_rows["closed_normal"].astype(bool))
    ].copy()
    tie_ids = sorted(int(value) for value in tie_rows["switch_id"])
    tie_ids.sort(key=lambda value: stable_hash([token, "tie", value]))
    for tie_id in tie_ids:
        tied = copy.deepcopy(net)
        tied.switch.at[tie_id, "closed"] = True
        if n_independent_loops(tied) != 1:
            continue
        edges = _cycle_edges(collapsed_graph(tied))
        candidates = []
        for row in switch_rows.itertuples():
            switch_id = int(row.switch_id)
            if switch_id == tie_id or not bool(tied.switch.at[switch_id, "closed"]):
                continue
            if str(row.type) not in {"sectionalizer", "RMU/LBS"}:
                continue
            pair = _switch_bus_pair(tied, switch_id)
            if pair is not None and frozenset(map(int, pair)) in edges:
                candidates.append(switch_id)
        candidates.sort(key=lambda value: stable_hash([token, "open", value]))
        for open_id in candidates:
            trial = copy.deepcopy(tied)
            trial.switch.at[open_id, "closed"] = False
            if n_independent_loops(trial) != 0 or unsupplied(trial):
                continue
            ok, metrics = pf_feasible(trial)
            if not ok:
                continue
            rows = switch_rows.copy()
            rows.loc[rows.switch_id == tie_id, "type"] = "sectionalizer"
            rows.loc[rows.switch_id == open_id, "type"] = "tie"
            rows.loc[rows.switch_id == tie_id, "closed_normal"] = True
            rows.loc[rows.switch_id == open_id, "closed_normal"] = False
            return trial, rows, {
                "closed_original_tie": tie_id,
                "opened_cycle_switch": open_id,
                "normal_pf": metrics,
            }
    return None


def _source_events(path: Path, excluded: set[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not event.get("interrupts") or not event.get("event_id"):
            continue
        if str(event["event_id"]) in excluded or event.get("parent_event_id") is not None:
            continue
        if any(key in event for key in ("storm_cluster_id", "storm_window")):
            continue
        result[str(event["grid_id"])].append(event)
    for grid_id in result:
        result[grid_id].sort(key=lambda row: stable_hash([grid_id, row["event_id"]]))
    return result


def _excluded_event_ids(config: dict) -> set[str]:
    excluded = set()
    for name in (
        "primary800",
        "robustness150",
        "natural200",
        "morphology_rural",
        "morphology_semiurban",
        "morphology_urban",
        "morphology_commercial",
    ):
        manifest = json.loads(resolve_repo_path(config["panels"][name]["path"]).read_text())
        excluded.update(str(row.get("event_id")) for row in manifest["episodes"] if row.get("event_id"))
    return excluded


def _prepare_series(source_series: Path, output_series: Path, variant_to_base: dict[str, str]) -> None:
    output_series.mkdir(parents=True, exist_ok=True)
    allocation = pd.read_parquet(source_series / "bus_allocation.parquet")
    rows = []
    for variant, base in variant_to_base.items():
        part = allocation[allocation.grid_id == base].copy()
        part["grid_id"] = variant
        rows.append(part)
    if not rows:
        raise RuntimeError("no bus-allocation rows were generated")
    pd.concat(rows, ignore_index=True).to_parquet(output_series / "bus_allocation.parquet", index=False)
    # The measured feeder series is immutable and panel-independent.  Reference it instead of
    # copying a 1.4 GB parquet into every generated panel.
    feeder_target = output_series / "real_feeders.parquet"
    if not feeder_target.exists():
        feeder_target.symlink_to((source_series / "real_feeders.parquet").resolve())


def _materialize_panel(name: str, config: dict, *, candidate_limit: int) -> dict:
    ood = config["ood"]
    source_grids = resolve_repo_path(ood["source_grids"])
    source_series = resolve_repo_path(ood["source_series"])
    output = ROOT / "artifacts/panels" / name
    grids_out = output / "distribution_grids"
    series_out = output / "series"
    grids_out.mkdir(parents=True, exist_ok=True)
    source_switches = pd.read_csv(source_grids / "switch_manifest.csv")
    variant_switches = []
    variant_to_base: dict[str, str] = {}
    variant_meta = []
    grid_paths = [
        source_grids / f"{grid_id}.json"
        for grid_id in sorted(str(value) for value in source_switches.grid_id.unique())
    ]
    for grid_path in grid_paths:
        base_id = grid_path.stem
        if name == "generation_ood" and not base_id.endswith("_2"):
            continue
        net = pp.from_json(str(grid_path))
        rows = source_switches[source_switches.grid_id == base_id].copy()
        if rows.empty:
            continue
        detail: dict = {"base_grid_id": base_id}
        if name == "topology_ood":
            relocated = relocate_open_point(net, rows, f"{config['seed']}:{base_id}")
            if relocated is None:
                continue
            net, rows, relocation = relocated
            detail.update(relocation)
            scale = calibrate_operating_point(net, target_pct=85.0)
            _scale_joint(net, scale)
            detail["joint_operating_scale"] = scale
        elif name == "generation_ood":
            multiplier = float(ood["generation_multiplier"])
            trial = copy.deepcopy(net)
            _scale_generation(trial, multiplier)
            scale = calibrate_operating_point(trial, target_pct=85.0)
            _scale_joint(trial, scale)
            ok, metrics = pf_feasible(trial)
            if not ok:
                multiplier = float(ood["generation_fallback_multiplier"])
                trial = copy.deepcopy(net)
                _scale_generation(trial, multiplier)
                scale = calibrate_operating_point(trial, target_pct=85.0)
                _scale_joint(trial, scale)
                ok, metrics = pf_feasible(trial)
            if not ok:
                continue
            net = trial
            detail.update({"generation_multiplier": multiplier, "joint_operating_scale": scale, "normal_pf": metrics})
        else:
            load_factor = _load_only_factor(net, float(ood["load_target_pct"]))
            for column in ("p_mw", "q_mvar"):
                if column in net.load.columns:
                    net.load[column] *= load_factor
            ok, metrics = pf_feasible(net)
            if not ok:
                continue
            detail.update({"load_multiplier": load_factor, "normal_pf": metrics})
        variant_id = f"{name}__{base_id}"
        rows["grid_id"] = variant_id
        rows["archetype"] = rows["archetype"].astype(str)
        for switch_id in net.switch.index:
            rows.loc[rows.switch_id == int(switch_id), "closed_normal"] = bool(net.switch.at[switch_id, "closed"])
        pp.to_json(net, str(grids_out / f"{variant_id}.json"))
        variant_switches.append(rows)
        variant_to_base[variant_id] = base_id
        variant_meta.append({"variant_grid_id": variant_id, **detail})
    if not variant_to_base:
        raise RuntimeError(f"no valid grid variants for {name}")
    switches = pd.concat(variant_switches, ignore_index=True)
    switches.to_csv(grids_out / "switch_manifest.csv", index=False)
    _prepare_series(source_series, series_out, variant_to_base)

    events_by_grid = _source_events(resolve_repo_path(ood["source_events"]), _excluded_event_ids(config))
    candidates = []
    for variant, base in variant_to_base.items():
        for event in events_by_grid.get(base, []):
            cloned = dict(event)
            cloned["grid_id"] = variant
            cloned["source_grid_id"] = base
            candidates.append(cloned)
    candidates.sort(key=lambda row: stable_hash([name, config["seed"], row["event_id"]]))
    candidates = candidates[:candidate_limit]
    if not candidates:
        raise RuntimeError(f"no unused events for {name}")
    events_path = output / "events.jsonl"
    atomic_text(events_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates))

    profile = hourly_load_profile(str(source_series))
    bin_of_hour, bin_factor = load_bin_factors(profile)
    atomic_json(
        output / "oracle/operating_point.json",
        {
            "base_scale": {variant: 1.0 for variant in variant_to_base},
            "hourly_profile": profile,
            "bin_of_hour": bin_of_hour,
            "bin_factor": bin_factor,
            "target_prefault_loading_pct": 85.0 if name != "load_ood" else float(ood["load_target_pct"]),
        },
    )
    atomic_json(
        output / "variant_manifest.json",
        {
            "schema_version": "neurips-grid-variants-v1",
            "created_utc": now_utc(),
            "panel": name,
            "variants": variant_meta,
        },
    )
    return {
        "name": name,
        "output": output,
        "n_variants": len(variant_to_base),
        "n_candidate_events": len(candidates),
    }


def _label_panel(
    name: str,
    config: dict,
    *,
    n_jobs: int,
    max_nodes: int,
    target_cases: int,
) -> dict:
    output = ROOT / "artifacts/panels" / name
    labels = output / "oracle/oracle_labels_v2.jsonl"
    meta = build_oracle_labels(
        grids_dir=str(output / "distribution_grids"),
        events_path=str(output / "events.jsonl"),
        obs_model_path=str(resolve_repo_path(config["ood"]["observation_model"])),
        out_path=str(labels),
        seed=int(config["seed"]),
        tiers=list(config["tiers"]),
        n_jobs=n_jobs,
        max_multifault_per_grid=0,
        initial_max_nodes=max_nodes,
        retry_max_nodes=max(20_000, max_nodes),
        convergence_max_nodes=(50_000,),
        max_opens=3,
        series_dir=str(output / "series"),
        checkpoint_dir=str(
            output / "oracle" / ("checkpoints_smoke" if target_cases == 8 else "checkpoints_full")
        ),
    )
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    eligible = [
        row for row in rows
        if int(row.get("controllable_customers", 0)) > 0
        and bool(row.get("initial_pf_feasible"))
        and bool(row.get("proven_optimal"))
    ]
    eligible.sort(key=lambda row: stable_hash([name, config["seed"], row["case_id"]]))
    selected = eligible[:target_cases]
    if len(selected) != target_cases:
        raise RuntimeError(
            f"{name} produced {len(selected)} proven-optimal eligible cases; "
            f"the preregistered target is {target_cases}"
        )
    selected_path = output / "oracle/selected_labels.jsonl"
    atomic_text(selected_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    dependence_hashes = {row.get("dependence_definition_sha256") for row in selected}
    manifest = {
        "schema_version": "neurips-grid-panel-v1",
        "created_utc": now_utc(),
        "name": name,
        "seed": int(config["seed"]),
        "grids_dir": str((output / "distribution_grids").resolve()),
        "series_dir": str((output / "series").resolve()),
        "labels_path": str(selected_path.resolve()),
        "labels_sha256": sha256_file(selected_path),
        "source_labels_path": str(labels.resolve()),
        "source_labels_sha256": sha256_file(labels),
        "dependence_definition_sha256": next(iter(dependence_hashes)) if len(dependence_hashes) == 1 else None,
        "episodes": [
            {
                "case_id": row["case_id"],
                "grid_id": row["grid_id"],
                "source_grid_id": row["grid_id"].split("__", 1)[-1],
                "event_id": row.get("event_id"),
                "dependence_block_id": row.get("dependence_block_id"),
                "difficulty_tier": row["by_tier"]["noisy"]["difficulty_tier"],
            }
            for row in selected
        ],
        "n_source_labels": len(rows),
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "oracle_meta": meta,
    }
    manifest_path = output / "panel_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_text(output / "panel_manifest.sha256", sha256_file(manifest_path) + "\n")
    return {"name": name, "n_labels": len(rows), "n_eligible": len(eligible), "n_selected": len(selected)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--panels", default="all")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--label-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.materialize_only and args.label_only:
        raise SystemExit("choose at most one of --materialize-only and --label-only")
    config = load_config(args.config)
    names = list(PANEL_NAMES) if args.panels == "all" else [value.strip() for value in args.panels.split(",")]
    unknown = set(names) - set(PANEL_NAMES)
    if unknown:
        raise SystemExit(f"unknown panels: {sorted(unknown)}")
    results = []
    # The smoke gate showed that roughly half of otherwise valid faults have no
    # restorable customer mass after isolation.  Oversample before the expensive
    # oracle pass so the fixed 200-case panel is achievable without selecting on
    # model outcomes.
    candidate_limit = 8 if args.smoke else int(config["ood"]["target_cases"]) * 3
    target_cases = 8 if args.smoke else int(config["ood"]["target_cases"])
    for name in names:
        if not args.label_only:
            results.append(_materialize_panel(name, config, candidate_limit=candidate_limit))
        if not args.materialize_only:
            results.append(
                _label_panel(
                    name,
                    config,
                    n_jobs=args.n_jobs,
                    max_nodes=args.max_nodes,
                    target_cases=target_cases,
                )
            )
    report = markdown_report(
        "ood_panels",
        "smoke" if args.smoke else "full",
        ["# FLARE OOD panels", "", "```json", json.dumps(results, indent=2, default=str), "```"],
    )
    print(json.dumps({"results": results, "report": str(report)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
