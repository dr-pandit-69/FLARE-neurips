from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_ood import PANEL_NAMES, _excluded_event_ids
from .common import CONFIG, ROOT, atomic_json, load_config, markdown_report, now_utc, sha256_file


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_panel(name: str, config: dict, excluded: set[str]) -> tuple[dict, list[str]]:
    root = ROOT / "artifacts/panels" / name
    errors = []
    required = (
        root / "panel_manifest.json",
        root / "variant_manifest.json",
        root / "events.jsonl",
        root / "oracle/selected_labels.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"panel": name, "missing": missing}, [f"{name}: missing {path}" for path in missing]
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    variants = json.loads(required[1].read_text(encoding="utf-8"))["variants"]
    events = _rows(required[2])
    labels = _rows(required[3])
    target = int(config["ood"]["target_cases"])
    if len(labels) != target or int(manifest.get("n_selected", -1)) != target:
        errors.append(f"{name}: selected {len(labels)} cases, expected {target}")
    case_ids = [str(row.get("case_id")) for row in labels]
    if len(case_ids) != len(set(case_ids)):
        errors.append(f"{name}: duplicate selected case ids")
    bad_labels = [
        row.get("case_id") for row in labels
        if not row.get("proven_optimal")
        or not row.get("initial_pf_feasible")
        or int(row.get("controllable_customers", 0)) <= 0
    ]
    if bad_labels:
        errors.append(f"{name}: {len(bad_labels)} selected labels fail eligibility")
    source_event_ids = {str(row.get("event_id")) for row in events}
    leaked = sorted(source_event_ids & excluded)
    if leaked:
        errors.append(f"{name}: {len(leaked)} source events overlap frozen evaluation panels")
    if manifest.get("labels_sha256") != sha256_file(required[3]):
        errors.append(f"{name}: selected-label hash mismatch")
    normal_pf_failures = []
    operating_values = []
    for row in variants:
        metrics = row.get("normal_pf") or {}
        if not metrics.get("converged") or not metrics.get("thermal_ok") or not metrics.get("voltage_ok"):
            normal_pf_failures.append(row.get("variant_grid_id"))
        if name == "generation_ood":
            operating_values.append(float(row["generation_multiplier"]))
        elif name == "load_ood":
            operating_values.append(float(row["load_multiplier"]))
        else:
            if "closed_original_tie" not in row or "opened_cycle_switch" not in row:
                errors.append(f"{name}: topology variant lacks a relocated open point")
            operating_values.append(float(row["joint_operating_scale"]))
    if normal_pf_failures:
        errors.append(f"{name}: {len(normal_pf_failures)} normal variants fail AC feasibility")
    summary = {
        "panel": name,
        "passed": not errors,
        "n_variants": len(variants),
        "n_source_events": len(events),
        "n_unique_source_event_ids": len(source_event_ids),
        "n_selected": len(labels),
        "n_proven_optimal": sum(bool(row.get("proven_optimal")) for row in labels),
        "n_initial_pf_feasible": sum(bool(row.get("initial_pf_feasible")) for row in labels),
        "operating_parameter_min": min(operating_values) if operating_values else None,
        "operating_parameter_max": max(operating_values) if operating_values else None,
        "errors": errors,
    }
    return summary, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    excluded = _excluded_event_ids(config)
    panels = []
    errors = []
    for name in PANEL_NAMES:
        summary, panel_errors = audit_panel(name, config, excluded)
        panels.append(summary)
        errors.extend(panel_errors)
    selected_event_sets = {
        name: {
            str(row.get("event_id"))
            for row in _rows(ROOT / "artifacts/panels" / name / "oracle/selected_labels.jsonl")
        }
        for name in PANEL_NAMES
        if (ROOT / "artifacts/panels" / name / "oracle/selected_labels.jsonl").is_file()
    }
    cross_panel_overlap = {
        f"{left}__{right}": len(selected_event_sets[left] & selected_event_sets[right])
        for index, left in enumerate(PANEL_NAMES)
        for right in PANEL_NAMES[index + 1:]
        if left in selected_event_sets and right in selected_event_sets
    }
    result = {
        "schema_version": "neurips-grid-panel-audit-v1",
        "created_utc": now_utc(),
        "passed": not errors,
        "panels": panels,
        "cross_panel_selected_event_overlap": cross_panel_overlap,
        "errors": errors,
    }
    output = ROOT / "artifacts/verification/panel_audit.json"
    atomic_json(output, result)
    report = markdown_report(
        "panel_audit",
        "ood3",
        [
            "# FLARE OOD panel audit",
            "",
            f"- Status: {'PASS' if result['passed'] else 'FAIL'}",
            f"- Errors: {len(errors)}",
            "",
            "| Panel | Variants | Candidates | Selected | Proven | Operating parameter range |",
            "|---|---:|---:|---:|---:|---:|",
            *(
                f"| {row['panel']} | {row.get('n_variants', 0)} | {row.get('n_source_events', 0)} | "
                f"{row.get('n_selected', 0)} | {row.get('n_proven_optimal', 0)} | "
                f"{row.get('operating_parameter_min')}--{row.get('operating_parameter_max')} |"
                for row in panels
            ),
            "",
            *(f"- {error}" for error in errors),
        ],
    )
    print(json.dumps({"result": result, "artifact": str(output), "report": str(report)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
