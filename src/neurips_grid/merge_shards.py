from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import CONFIG, atomic_json, load_config, markdown_report, now_utc, resolve_repo_path
from .worker import cells_from_config


IDENTITY_FIELDS = (
    "artifact_generation_id",
    "cell",
    "panel_manifest_sha256",
    "labels_sha256",
    "model_slug",
    "model_id",
    "model_revision",
    "runner_code_sha256",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sources = [json.loads(args.canonical.read_text(encoding="utf-8"))]
    sources.extend(json.loads(path.read_text(encoding="utf-8")) for path in args.shards)
    base_meta = sources[0]["meta"]
    for source in sources[1:]:
        for field in IDENTITY_FIELDS:
            if source["meta"].get(field) != base_meta.get(field):
                raise RuntimeError(f"shard identity mismatch for {field}")
        if not source["meta"].get("completed"):
            raise RuntimeError("cannot merge an incomplete shard")

    cell_name = str(base_meta["cell"])
    cell = {item.name: item for item in cells_from_config(config)}[cell_name]
    manifest_path = resolve_repo_path(config["panels"][cell.panel]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [
        (str(row["case_id"]), tier)
        for tier in cell.tiers
        for row in manifest["episodes"]
    ]

    by_pair: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for source in sources:
        source_events = source.get("safety_events", [])
        for record in source.get("records", []):
            pair = (str(record["case_id"]), str(record["tier"]))
            start = int(record["shadow_event_start"])
            count = int(record["shadow_event_count"])
            linked = source_events[start:start + count]
            candidate = (record, linked)
            if pair in by_pair:
                previous_record, previous_events = by_pair[pair]
                previous_comparable = dict(previous_record)
                candidate_comparable = dict(record)
                previous_comparable.pop("shadow_event_start", None)
                candidate_comparable.pop("shadow_event_start", None)
                if previous_comparable != candidate_comparable or previous_events != linked:
                    raise RuntimeError(f"conflicting duplicate record for {pair}")
                continue
            by_pair[pair] = candidate

    expected_set = set(expected)
    if set(by_pair) != expected_set:
        missing = len(expected_set - set(by_pair))
        extra = len(set(by_pair) - expected_set)
        raise RuntimeError(f"incomplete shard union: missing={missing} extra={extra}")

    records: list[dict] = []
    events: list[dict] = []
    for pair in expected:
        source_record, linked = by_pair[pair]
        record = dict(source_record)
        record["shadow_event_start"] = len(events)
        record["shadow_event_count"] = len(linked)
        records.append(record)
        events.extend(linked)

    transcripts_by_key: dict[tuple[str, str], dict] = {}
    for source in sources:
        for transcript in source.get("transcripts", []):
            key = (str(transcript.get("case_id")), str(transcript.get("tier")))
            transcripts_by_key.setdefault(key, transcript)

    meta = dict(base_meta)
    meta.update(
        {
            "completed": True,
            "n_expected_episodes": len(expected),
            "n_episodes": len(records),
            "updated_utc": now_utc(),
        }
    )
    result = {
        "meta": meta,
        "records": records,
        "safety_events": events,
        "transcripts": list(transcripts_by_key.values()),
    }
    atomic_json(args.canonical, result)
    report = markdown_report(
        "model_cell",
        f"{meta['model_slug']}_{cell_name}_merged2shards",
        [
            "# FLARE model cell",
            "",
            f"- Model: `{meta['model_id']}`",
            f"- Revision: `{meta['model_revision']}`",
            f"- Cell: `{cell_name}`",
            f"- Episodes: {len(records)}",
            f"- Shadow guard events: {len(events)}",
            f"- Artifact: `{args.canonical.resolve()}`",
            f"- Parallel shards merged: {len(args.shards)}",
            "- Status: PASS",
        ],
    )
    print(f"MERGED {len(records)} records into {args.canonical.resolve()} report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
