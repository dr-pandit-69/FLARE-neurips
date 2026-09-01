from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandapower as pp
import torch

from .common import (
    CONFIG,
    REPO,
    ROOT,
    atomic_json,
    load_config,
    markdown_report,
    now_utc,
    resolve_repo_path,
    sha256_file,
    stable_hash,
)
from .shadow_safety import audit_trace, summarize_events

sys.path.insert(0, str(REPO / "src"))

import gridagent.agent_interface.scripts.run_episodes as base  # noqa: E402
from gridagent.agent_interface.llm_agent import ALL_VERBS, run_llm_episode  # noqa: E402
from gridagent.agent_interface.restoration_env import Fault  # noqa: E402
from gridagent.oracle.restoration_oracle import CLASS_WEIGHTS  # noqa: E402
from gridagent.simulation.grid_bank import stable_seed  # noqa: E402
from gridagent.system.llm_proposer import LocalLLM  # noqa: E402


@dataclass(frozen=True)
class Cell:
    name: str
    panel: str
    tiers: tuple[str, ...]
    temperature: float
    top_p: float
    decoding_seed: int
    true_state: bool
    rank_ties: bool
    briefing: str
    use_memory: bool
    allowed_verbs: frozenset[str]


class Panel:
    def __init__(self, name: str, manifest_path: Path):
        self.name = name
        self.manifest_path = manifest_path.resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.net_cache: dict[str, Any] = {}
        if self.manifest.get("schema_version") == "neurips-grid-panel-v1":
            self._load_neurips_panel()
        else:
            self._load_existing_panel()

    def _load_existing_panel(self) -> None:
        slate, counts, pool_counts, info = base._load_panel_slate(str(self.manifest_path))
        self.rows = slate
        self.difficulty_counts = counts
        self.pool_counts = pool_counts
        self.labels_path = Path(info["labels_path"])
        self.labels_sha256 = info["labels_sha256"]
        self.dependence_definition_sha256 = info["dependence_definition_sha256"]
        self.cust, self.weights = base._load_cust_and_weights()
        self.switches = base._load_sw_meta()
        self.grids_dir = Path(base.GRIDS)

    def _load_neurips_panel(self) -> None:
        labels = Path(self.manifest["labels_path"])
        if sha256_file(labels) != self.manifest["labels_sha256"]:
            raise ValueError(f"labels hash mismatch: {labels}")
        by_id = {
            row["case_id"]: row
            for line in labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        ids = [row["case_id"] for row in self.manifest["episodes"]]
        if len(ids) != len(set(ids)) or any(case_id not in by_id for case_id in ids):
            raise ValueError(f"invalid case index in {self.manifest_path}")
        self.rows = [by_id[case_id] for case_id in ids]
        self.difficulty_counts = dict(Counter(row["by_tier"]["noisy"]["difficulty_tier"] for row in self.rows))
        self.pool_counts = self.difficulty_counts.copy()
        self.labels_path = labels
        self.labels_sha256 = self.manifest["labels_sha256"]
        self.dependence_definition_sha256 = self.manifest.get("dependence_definition_sha256")
        self.grids_dir = Path(self.manifest["grids_dir"])
        series_dir = Path(self.manifest["series_dir"])
        allocation = pd.read_parquet(series_dir / "bus_allocation.parquet")
        self.cust = {
            str(grid): {int(bus): int(customers) for bus, customers in zip(rows.bus, rows.customers)}
            for grid, rows in allocation.groupby("grid_id")
        }
        self.weights = {
            str(grid): {
                int(bus): float(CLASS_WEIGHTS.get(str(kind), 1.0))
                for bus, kind in zip(rows.bus, rows.customer_class)
            }
            for grid, rows in allocation.groupby("grid_id")
        }
        switches = pd.read_csv(self.grids_dir / "switch_manifest.csv")
        self.switches = {
            str(grid): {
                int(row.switch_id): {
                    "remote": bool(row.remote),
                    "travel_time_min": float(row.travel_time_min),
                    "type": str(row.type),
                }
                for row in rows.itertuples()
            }
            for grid, rows in switches.groupby("grid_id")
        }

    def net(self, grid_id: str):
        if grid_id not in self.net_cache:
            self.net_cache[grid_id] = pp.from_json(str(self.grids_dir / f"{grid_id}.json"))
        return self.net_cache[grid_id]


def cells_from_config(config: dict) -> list[Cell]:
    default_generation = config["generation"]
    result = []
    for name, spec in config["cells"].items():
        tiers = tuple(config["tiers"] if spec.get("tiers") == "all" else spec["tiers"])
        result.append(
            Cell(
                name=name,
                panel=str(spec["panel"]),
                tiers=tiers,
                temperature=float(spec.get("temperature", default_generation["temperature"])),
                top_p=float(spec.get("top_p", default_generation["top_p"])),
                decoding_seed=int(spec.get("decoding_seed", config["decoding_seed"])),
                true_state=bool(spec.get("true_state", False)),
                rank_ties=bool(spec.get("rank_ties", True)),
                briefing=str(spec.get("briefing", "full")),
                use_memory=bool(spec.get("use_memory", True)),
                allowed_verbs=frozenset(spec.get("allowed_verbs", ALL_VERBS)),
            )
        )
    return result


def package_versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "tokenizers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def runner_hash(config_path: Path) -> str:
    paths = [
        Path(__file__),
        Path(audit_trace.__code__.co_filename),
        config_path,
        REPO / "src/gridagent/agent_interface/llm_agent.py",
        REPO / "src/gridagent/agent_interface/restoration_env.py",
        REPO / "src/gridagent/agent_interface/scripts/run_episodes.py",
        REPO / "src/gridagent/system/llm_proposer.py",
    ]
    return stable_hash({str(path): sha256_file(path) for path in paths})


def _metadata(
    *,
    config: dict,
    cell: Cell,
    panel: Panel,
    model_slug: str,
    model_spec: dict,
    runtime: dict,
    code_hash: str,
    generation_id: str,
    n_expected: int,
    n_records: int,
    started: str,
    completed: bool,
) -> dict:
    return {
        "schema_version": "neurips-grid-model-cell-v1",
        "artifact_generation_id": generation_id,
        "cell": cell.name,
        "panel": panel.name,
        "panel_manifest_path": str(panel.manifest_path),
        "panel_manifest_sha256": panel.manifest_sha256,
        "labels_path": str(panel.labels_path),
        "labels_sha256": panel.labels_sha256,
        "dependence_definition_sha256": panel.dependence_definition_sha256,
        "model_slug": model_slug,
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "quantization": model_spec.get("quantization"),
        "tiers": list(cell.tiers),
        "temperature": cell.temperature,
        "top_p": cell.top_p,
        "true_state": cell.true_state,
        "rank_ties": cell.rank_ties,
        "briefing": cell.briefing,
        "use_memory": cell.use_memory,
        "allowed_verbs": sorted(cell.allowed_verbs),
        "rollout_seed": int(config["rollout_seed"]),
        "runner_code_sha256": code_hash,
        "runtime": runtime,
        "n_expected_episodes": n_expected,
        "n_episodes": n_records,
        "difficulty_counts": panel.difficulty_counts,
        "started_utc": started,
        "updated_utc": now_utc(),
        "completed": completed,
    }


def _generation_spec(config: dict, cell: Cell, panel: Panel, model_slug: str, model_spec: dict, code_hash: str, max_cases: int | None) -> dict:
    cell_spec = {
        **cell.__dict__,
        "tiers": list(cell.tiers),
        "allowed_verbs": sorted(cell.allowed_verbs),
    }
    return {
        "schema_version": "neurips-grid-generation-v1",
        "model_slug": model_slug,
        "model_id": model_spec["model_id"],
        "revision": model_spec["revision"],
        "cell": cell_spec,
        "panel_sha256": panel.manifest_sha256,
        "labels_sha256": panel.labels_sha256,
        "runner_sha256": code_hash,
        "rollout_seed": config["rollout_seed"],
        "max_new_tokens": config["generation"]["max_new_tokens"],
        "max_cases": max_cases,
    }


def run_cell(
    *,
    llm,
    runtime: dict,
    config: dict,
    model_slug: str,
    model_spec: dict,
    cell: Cell,
    panel: Panel,
    output_dir: Path,
    code_hash: str,
    max_cases: int | None,
    force: bool,
) -> Path:
    rows = panel.rows[:max_cases] if max_cases is not None else panel.rows
    expected = [(row["case_id"], tier) for tier in cell.tiers for row in rows]
    expected_set = set(expected)
    spec = _generation_spec(config, cell, panel, model_slug, model_spec, code_hash, max_cases)
    generation_id = stable_hash(spec)[:20]
    path = output_dir / f"{cell.name}.json"
    records: list[dict] = []
    safety_events: list[dict] = []
    transcripts: list[dict] = []
    started = now_utc()
    if path.exists() and not force:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("meta", {}).get("artifact_generation_id") != generation_id:
            raise SystemExit(f"stale artifact has a different generation: {path}; use --force only after reviewing it")
        records = list(artifact.get("records", []))
        safety_events = list(artifact.get("safety_events", []))
        transcripts = list(artifact.get("transcripts", []))
        started = artifact.get("meta", {}).get("started_utc", started)
        if artifact.get("meta", {}).get("completed"):
            found = {(row.get("case_id"), row.get("tier")) for row in records}
            if found == expected_set and len(records) == len(expected_set):
                print(f"SKIP {model_slug} {cell.name}: already complete", flush=True)
                return path
    completed_pairs = {(row.get("case_id"), row.get("tier")) for row in records}
    if len(completed_pairs) != len(records) or not completed_pairs <= expected_set:
        raise RuntimeError(f"invalid resumable records in {path}")

    def checkpoint(done: bool = False) -> None:
        meta = _metadata(
            config=config,
            cell=cell,
            panel=panel,
            model_slug=model_slug,
            model_spec=model_spec,
            runtime=runtime,
            code_hash=code_hash,
            generation_id=generation_id,
            n_expected=len(expected_set),
            n_records=len(records),
            started=started,
            completed=done,
        )
        atomic_json(path, {"meta": meta, "records": records, "safety_events": safety_events, "transcripts": transcripts})

    llm.temperature = cell.temperature
    llm.top_p = cell.top_p
    transcript_limit = int(config["analysis"]["transcript_cases_per_tier"])
    checkpoint_every = int(config["generation"]["checkpoint_every"])
    for tier in cell.tiers:
        for row_index, oracle_row in enumerate(rows):
            pair = (oracle_row["case_id"], tier)
            if pair in completed_pairs:
                continue
            grid_id = str(oracle_row["grid_id"])
            rollout_seed = base._rollout_seed(oracle_row["case_id"], "llm", int(config["rollout_seed"]), tier)
            episode_decoding_seed = stable_seed((
                "neurips-grid-decoding",
                oracle_row["case_id"],
                tier,
                cell.name,
                cell.decoding_seed,
            ))
            llm.set_decoding_seed(episode_decoding_seed)
            fault = Fault(
                faulted_lines=[],
                faulted_buses=list(oracle_row["faulted_buses"]),
                earthing_regime=oracle_row.get("earthing_regime") or "resonant/compensated",
                load_scale=float(oracle_row.get("load_scale", 1.0)),
            )
            trace: list[dict] = []
            result = run_llm_episode(
                llm,
                panel.net(grid_id),
                fault,
                base._tier_label(oracle_row, tier),
                panel.cust.get(grid_id, {}),
                tier,
                sw_meta=panel.switches.get(grid_id, {}),
                weights=panel.weights.get(grid_id, {}),
                seed=rollout_seed,
                rank_ties=cell.rank_ties,
                true_state=cell.true_state,
                briefing=cell.briefing,
                use_memory=cell.use_memory,
                allowed_verbs=set(cell.allowed_verbs),
                capture_trace=trace,
            )
            episode_events = audit_trace(panel.net(grid_id), fault, trace)
            event_start = len(safety_events)
            for event in episode_events:
                safety_events.append(
                    {
                        "model_slug": model_slug,
                        "cell": cell.name,
                        "case_id": oracle_row["case_id"],
                        "tier": tier,
                        **event,
                    }
                )
            record = {
                **base._record_common(oracle_row, tier),
                **result,
                **summarize_events(episode_events),
                "model": model_spec["model_id"],
                "model_slug": model_slug,
                "model_revision": model_spec["revision"],
                "quantization": model_spec.get("quantization"),
                "cell": cell.name,
                "artifact_generation_id": generation_id,
                "episode_rollout_seed": rollout_seed,
                "episode_decoding_seed": episode_decoding_seed,
                "shadow_event_start": event_start,
                "shadow_event_count": len(episode_events),
                "uses_hidden_state": bool(cell.true_state),
                "post_isolation_stage": "RESTORE",
            }
            records.append(record)
            completed_pairs.add(pair)
            if cell.name == "primary_A0" and row_index < transcript_limit:
                transcripts.append(
                    {
                        "case_id": oracle_row["case_id"],
                        "tier": tier,
                        "model_slug": model_slug,
                        "trace": trace,
                    }
                )
            if len(records) % checkpoint_every == 0:
                checkpoint()
                print(f"{model_slug} {cell.name}: {len(records)}/{len(expected_set)}", flush=True)
    if completed_pairs != expected_set or len(records) != len(expected_set):
        checkpoint()
        raise RuntimeError(f"incomplete cell {cell.name}: {len(records)}/{len(expected_set)}")
    checkpoint(done=True)
    report = markdown_report(
        "model_cell",
        f"{model_slug}_{cell.name}",
        [
            "# FLARE model cell",
            "",
            f"- Model: `{model_spec['model_id']}`",
            f"- Revision: `{model_spec['revision']}`",
            f"- Cell: `{cell.name}`",
            f"- Episodes: {len(records)}",
            f"- Shadow guard events: {len(safety_events)}",
            f"- Artifact: `{path}`",
            "- Status: PASS",
        ],
    )
    print(f"DONE {model_slug} {cell.name}: {path} report={report}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cells", default="all", help="all, core, or comma-separated cell names")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.model not in config["models"]:
        raise SystemExit(f"unknown model: {args.model}")
    model_spec = dict(config["models"][args.model])
    if model_spec["revision"] == "resolve-on-download":
        raise SystemExit("conditional model revision must be resolved and frozen before inference")
    available_cells = cells_from_config(config)
    by_name = {cell.name: cell for cell in available_cells}
    if args.smoke:
        selected = [by_name["primary_A0"]]
        args.max_cases = args.max_cases or 5
    elif args.cells == "all":
        selected = available_cells
    elif args.cells == "core":
        selected = [cell for cell in available_cells if cell.name.endswith("_A0")]
    else:
        wanted = [value.strip() for value in args.cells.split(",")]
        unknown = set(wanted) - set(by_name)
        if unknown:
            raise SystemExit(f"unknown cells: {sorted(unknown)}")
        selected = [by_name[name] for name in wanted]
    output_root = args.output_root or (ROOT / "runs/smoke" if args.smoke else ROOT / "runs/full")
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
    }
    model_dir = output_root.resolve() / "models" / args.model
    atomic_json(model_dir / "runtime.json", runtime)
    panel_cache: dict[str, Panel] = {}
    for cell in selected:
        if cell.panel not in panel_cache:
            panel_cache[cell.panel] = Panel(cell.panel, resolve_repo_path(config["panels"][cell.panel]["path"]))
        run_cell(
            llm=llm,
            runtime=runtime,
            config=config,
            model_slug=args.model,
            model_spec=model_spec,
            cell=cell,
            panel=panel_cache[cell.panel],
            output_dir=model_dir,
            code_hash=code_hash,
            max_cases=args.max_cases,
            force=args.force,
        )
    atomic_json(
        model_dir / "worker_complete.json",
        {
            "schema_version": "neurips-grid-worker-complete-v1",
            "model_slug": args.model,
            "cells": [cell.name for cell in selected],
            "runner_code_sha256": code_hash,
            "completed_utc": now_utc(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
