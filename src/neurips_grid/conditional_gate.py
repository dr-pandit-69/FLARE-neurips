from __future__ import annotations

import json
from pathlib import Path

from .common import CONFIG, ROOT, atomic_json, load_config, markdown_report, now_utc
from .inventory import model_entry


def main() -> int:
    config = load_config(CONFIG)
    slug = "llama33_70b_4bit"
    spec = config["models"][slug]
    candidate = model_entry(slug, spec)
    full_cache = (
        Path.home()
        / ".cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots"
    )
    full_snapshot_bytes = 0
    for snapshot in full_cache.glob("*") if full_cache.exists() else ():
        size = 0
        for path in snapshot.rglob("*"):
            if path.is_file():
                try:
                    size += path.resolve().stat().st_size
                except FileNotFoundError:
                    pass
        full_snapshot_bytes = max(full_snapshot_bytes, size)
    reasons = []
    if spec["revision"] == "resolve-on-download":
        reasons.append("the declared 4-bit checkpoint has no resolved, frozen revision")
    if not candidate["cached"]:
        reasons.append("the declared 4-bit checkpoint is not cached")
    if full_snapshot_bytes > 40 * 1024**3:
        reasons.append("the available full-precision checkpoint exceeds the 40GB device capacity")
    passed = not reasons
    result = {
        "schema_version": "neurips-grid-conditional-model-gate-v1",
        "created_utc": now_utc(),
        "model_slug": slug,
        "model_id": spec["model_id"],
        "a100_capacity_gib": 40,
        "declared_quantized_snapshot_cached": candidate["cached"],
        "declared_revision": spec["revision"],
        "available_full_snapshot_bytes": full_snapshot_bytes,
        "passed_prelaunch_gate": passed,
        "reasons": reasons,
        "decision": "launch smoke" if passed else "omit conditional result",
    }
    output = ROOT / "artifacts/provenance/conditional_70b_gate.json"
    atomic_json(output, result)
    report = markdown_report(
        "conditional_gate",
        "llama33_70b",
        [
            "# Conditional 70B model gate",
            "",
            f"- Status: {'PASS' if passed else 'NOT LAUNCHED'}",
            f"- Declared model: `{spec['model_id']}`",
            f"- Declared revision: `{spec['revision']}`",
            f"- Cached declared 4-bit snapshot: {candidate['cached']}",
            f"- Cached full-precision bytes: {full_snapshot_bytes}",
            *(f"- Reason: {reason}" for reason in reasons),
        ],
    )
    print(json.dumps({"result": result, "artifact": str(output), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
