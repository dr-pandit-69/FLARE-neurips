from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .common import CONFIG, atomic_json, load_config, now_utc, resolve_repo_path
from .worker import (
    LocalLLM,
    Panel,
    cells_from_config,
    package_versions,
    run_cell,
    runner_hash,
)


def _seed_shard(canonical_path: Path, shard_path: Path, allowed_cases: set[str], expected_count: int) -> None:
    if shard_path.exists():
        return
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    source_events = canonical.get("safety_events", [])
    records: list[dict] = []
    events: list[dict] = []
    for source_record in canonical.get("records", []):
        if str(source_record.get("case_id")) not in allowed_cases:
            continue
        record = dict(source_record)
        start = int(record.get("shadow_event_start", 0))
        count = int(record.get("shadow_event_count", 0))
        linked = source_events[start:start + count]
        record["shadow_event_start"] = len(events)
        record["shadow_event_count"] = len(linked)
        records.append(record)
        events.extend(linked)
    transcripts = [
        row for row in canonical.get("transcripts", [])
        if str(row.get("case_id")) in allowed_cases
    ]
    meta = dict(canonical["meta"])
    meta.update(
        {
            "completed": False,
            "n_expected_episodes": expected_count,
            "n_episodes": len(records),
            "updated_utc": now_utc(),
        }
    )
    atomic_json(
        shard_path,
        {"meta": meta, "records": records, "safety_events": events, "transcripts": transcripts},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.num_shards < 2 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index/count")

    config = load_config(args.config)
    model_spec = dict(config["models"][args.model])
    cells = {cell.name: cell for cell in cells_from_config(config)}
    cell = cells[args.cell]
    panel = Panel(cell.panel, resolve_repo_path(config["panels"][cell.panel]["path"]))
    panel.rows = [row for index, row in enumerate(panel.rows) if index % args.num_shards == args.shard_index]
    allowed_cases = {str(row["case_id"]) for row in panel.rows}
    expected_count = len(panel.rows) * len(cell.tiers)
    model_dir = args.output_root.resolve() / "models" / args.model
    shard_path = model_dir / f"{cell.name}.json"
    _seed_shard(args.canonical.resolve(), shard_path, allowed_cases, expected_count)

    code_hash = runner_hash(args.config)
    torch.cuda.reset_peak_memory_stats()
    llm = LocalLLM(
        model_spec["model_id"],
        device="cuda:0",
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        decoding_seed=int(config["decoding_seed"]),
        revision=model_spec["revision"],
    )
    if llm.resolved_revision != model_spec["revision"]:
        raise RuntimeError(f"model revision mismatch: {llm.resolved_revision} != {model_spec['revision']}")
    runtime = {
        "model_id": model_spec["model_id"],
        "resolved_revision": llm.resolved_revision,
        "model_type": llm.model_type,
        "loader_class": llm.loader_class,
        "tokenizer_class": type(llm.tok).__name__,
        "chat_template_sha256": llm.chat_template_sha256,
        "quantization_config": str(getattr(getattr(llm.model, "config", None), "quantization_config", None)),
        "package_versions": package_versions(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peak_cuda_memory_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        "created_utc": now_utc(),
        "parallel_shard": {"index": args.shard_index, "count": args.num_shards},
    }
    atomic_json(model_dir / "runtime.json", runtime)
    run_cell(
        llm=llm,
        runtime=runtime,
        config=config,
        model_slug=args.model,
        model_spec=model_spec,
        cell=cell,
        panel=panel,
        output_dir=model_dir,
        code_hash=code_hash,
        max_cases=None,
        force=False,
    )
    atomic_json(
        model_dir / "shard_complete.json",
        {
            "schema_version": "neurips-grid-shard-complete-v1",
            "model_slug": args.model,
            "cell": args.cell,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "n_expected_episodes": expected_count,
            "completed_utc": now_utc(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
