from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import CONFIG, REPO, ROOT, atomic_json, load_config, markdown_report, now_utc, resolve_repo_path, sha256_file
from .worker import cells_from_config


def verify_artifact(path: Path, config: dict, *, allow_subset: bool = False) -> list[str]:
    errors = []
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"{path}: unreadable ({type(exc).__name__})"]
    meta = artifact.get("meta", {})
    records = artifact.get("records", [])
    cell_by_name = {cell.name: cell for cell in cells_from_config(config)}
    cell = cell_by_name.get(str(meta.get("cell")))
    if cell is None:
        errors.append(f"{path}: unknown configured cell {meta.get('cell')!r}")
        expected_keys: set[tuple[str, str]] = set()
    else:
        manifest_path = resolve_repo_path(config["panels"][cell.panel]["path"])
        if not manifest_path.exists():
            errors.append(f"{path}: configured panel manifest is missing: {manifest_path}")
            expected_keys = set()
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_keys = {
                (str(row["case_id"]), tier)
                for row in manifest.get("episodes", [])
                for tier in cell.tiers
            }
            if meta.get("panel_manifest_sha256") != sha256_file(manifest_path):
                errors.append(f"{path}: panel manifest hash mismatch")
            if Path(str(meta.get("panel_manifest_path", ""))).resolve() != manifest_path.resolve():
                errors.append(f"{path}: panel manifest path mismatch")
    model_spec = config["models"].get(str(meta.get("model_slug")))
    if model_spec is None:
        errors.append(f"{path}: unknown model slug {meta.get('model_slug')!r}")
    else:
        if meta.get("model_id") != model_spec.get("model_id"):
            errors.append(f"{path}: configured model id mismatch")
        if meta.get("model_revision") != model_spec.get("revision"):
            errors.append(f"{path}: configured model revision mismatch")
    labels_path = Path(str(meta.get("labels_path", "")))
    if not labels_path.is_file() or sha256_file(labels_path) != meta.get("labels_sha256"):
        errors.append(f"{path}: oracle-label file/hash mismatch")
    if not meta.get("completed"):
        errors.append(f"{path}: incomplete")
    if len(records) != int(meta.get("n_expected_episodes", -1)):
        errors.append(f"{path}: row count {len(records)} != {meta.get('n_expected_episodes')}")
    keys = [(row.get("case_id"), row.get("tier")) for row in records]
    if len(keys) != len(set(keys)):
        errors.append(f"{path}: duplicate case/tier records")
    if expected_keys:
        found_keys = set(keys)
        missing = len(expected_keys - found_keys)
        extra = len(found_keys - expected_keys)
        if extra or (missing and not allow_subset):
            errors.append(f"{path}: configured case/tier mismatch (missing={missing}, extra={extra})")
    for index, row in enumerate(records):
        for field in (
            "case_id", "tier", "model", "model_slug", "model_revision", "cell",
            "artifact_generation_id", "regret_omni", "flisr_hardfail",
            "blocked_total", "shadow_attempts", "shadow_unsafe_attempts",
        ):
            if field not in row:
                errors.append(f"{path}: row {index} missing {field}")
        for field in ("regret_omni", "restored_customers", "flisr_hardfail", "blocked_total"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{path}: row {index} non-finite {field}")
        if float(row.get("regret_raw", 0.0)) < -1e-8:
            errors.append(f"{path}: row {index} beats oracle under shared accounting")
        if row.get("model_revision") != meta.get("model_revision"):
            errors.append(f"{path}: row {index} model revision mismatch")
        if row.get("artifact_generation_id") != meta.get("artifact_generation_id"):
            errors.append(f"{path}: row {index} generation mismatch")
    for event_index, event in enumerate(artifact.get("safety_events", [])):
        if not event.get("live_state_unchanged"):
            errors.append(f"{path}: safety event {event_index} changed live state")
        if event.get("unsafe") and not event.get("classes"):
            errors.append(f"{path}: safety event {event_index} lacks classification")
    safety_events = artifact.get("safety_events", [])
    for index, row in enumerate(records):
        start = int(row.get("shadow_event_start", -1))
        count = int(row.get("shadow_event_count", -1))
        if start < 0 or count < 0 or start + count > len(safety_events):
            errors.append(f"{path}: row {index} has invalid safety-event slice")
            continue
        linked = safety_events[start:start + count]
        if any(
            event.get("case_id") != row.get("case_id")
            or event.get("tier") != row.get("tier")
            for event in linked
        ):
            errors.append(f"{path}: row {index} safety-event linkage mismatch")
        if count != int(row.get("shadow_attempts", -1)):
            errors.append(f"{path}: row {index} shadow-attempt count mismatch")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/full")
    parser.add_argument("--models", default="all")
    parser.add_argument("--cells", default="all")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    models = [name for name, spec in config["models"].items() if not spec.get("conditional")]
    if args.models != "all":
        models = [value.strip() for value in args.models.split(",")]
    configured_cells = cells_from_config(config)
    # The time-constrained study uses one predeclared stochastic-decoding seed.
    # Keep the other configured seeds available as optional supplementary cells,
    # but do not require them for the frozen one-seed matrix.
    cells = [
        cell.name
        for cell in configured_cells
        if not cell.name.startswith("reliability_seed") or cell.name == "reliability_seed0"
    ]
    if args.cells == "core":
        cells = [name for name in cells if name.endswith("_A0")]
    elif args.cells != "all":
        cells = [value.strip() for value in args.cells.split(",")]
    errors = []
    checked = []
    runner_hashes = set()
    for model in models:
        for cell in cells:
            path = args.root / "models" / model / f"{cell}.json"
            if not path.exists():
                if not args.allow_partial:
                    errors.append(f"missing {path}")
                continue
            checked.append(path)
            errors.extend(verify_artifact(path, config, allow_subset=args.allow_partial))
            artifact = json.loads(path.read_text(encoding="utf-8"))
            runner_hashes.add(artifact.get("meta", {}).get("runner_code_sha256"))
    if len(runner_hashes) > 1:
        errors.append(f"mixed runner generations: {sorted(runner_hashes)}")
    if not args.allow_partial:
        panel_audit_path = ROOT / "artifacts/verification/panel_audit.json"
        if not panel_audit_path.is_file():
            errors.append("missing OOD panel audit")
        else:
            panel_audit = json.loads(panel_audit_path.read_text(encoding="utf-8"))
            if not panel_audit.get("passed"):
                errors.append("OOD panel audit did not pass")
    result = {
        "schema_version": "neurips-grid-verification-v1",
        "created_utc": now_utc(),
        "passed": not errors,
        "allow_partial": bool(args.allow_partial),
        "model_scope": models,
        "cell_scope": cells,
        "n_artifacts_checked": len(checked),
        "artifact_sha256": {str(path): sha256_file(path) for path in checked},
        "errors": errors,
    }
    output = ROOT / "artifacts/verification/verification.json"
    atomic_json(output, result)
    report = markdown_report(
        "verification",
        "partial" if args.allow_partial else "full",
        [
            "# FLARE verification",
            "",
            f"- Status: {'PASS' if result['passed'] else 'FAIL'}",
            f"- Artifacts checked: {len(checked)}",
            f"- Errors: {len(errors)}",
            "",
            *(f"- {error}" for error in errors[:100]),
        ],
    )
    print(json.dumps({"result": result, "manifest": str(output), "report": str(report)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
