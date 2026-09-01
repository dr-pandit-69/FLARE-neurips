"""Load only run artifacts produced by the current labels and rollout code."""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SUB = REPO / "data" / "processed" / "substrate_v1"
EXCLUDED_RUN_DIRS = frozenset({
    "hidden", "results", "repro", "smoke", "stale_pre_audit", "stale_pre_resolve",
})
RUNNER_SOURCE_PATHS = (
    "src/gridagent/agent_interface/scripts/run_episodes.py",
    "src/gridagent/agent_interface/llm_agent.py",
    "src/gridagent/agent_interface/baselines/restoration_policies.py",
    "src/gridagent/agent_interface/restoration_env.py",
    "src/gridagent/agent_interface/restoration_tools.py",
    "src/gridagent/agent_interface/cost_model.py",
    "src/gridagent/system/llm_proposer.py",
    "src/gridagent/simulation/grid_bank.py",
    "src/gridagent/simulation/pf_feasibility.py",
)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def runner_code_sha256(repo: str | os.PathLike = REPO) -> str:
    root = Path(repo)
    h = hashlib.sha256()
    for rel in RUNNER_SOURCE_PATHS:
        path = root / rel
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def current_label_sha256(sub_dir: str | os.PathLike = DEFAULT_SUB) -> str:
    oracle = Path(sub_dir) / "oracle"
    labels_path = oracle / "oracle_labels_v2.jsonl"
    meta_path = oracle / "oracle_meta.json"
    actual = sha256_file(labels_path)
    with meta_path.open() as fh:
        declared = json.load(fh).get("labels_sha256")
    if not declared:
        raise RuntimeError(f"{meta_path} has no labels_sha256; rebuild the oracle labels")
    if declared != actual:
        raise RuntimeError(
            f"oracle label hash mismatch: metadata={declared}, actual={actual}; "
            "the label generation is incomplete"
        )
    return actual


def load_current_runs(
    run_dir: str | os.PathLike,
    *,
    sub_dir: str | os.PathLike = DEFAULT_SUB,
) -> tuple[list[dict], dict]:
    """Return current runs and an audit of files excluded as stale or auxiliary."""
    run_dir = os.path.abspath(run_dir)
    label_hash = current_label_sha256(sub_dir)
    runner_hash = runner_code_sha256()
    runs: list[dict] = []
    audit = {"accepted": [], "stale": [], "auxiliary": [], "unreadable": []}

    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(path, run_dir)
        if EXCLUDED_RUN_DIRS.intersection(Path(rel).parts):
            audit["auxiliary"].append(rel)
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            audit["unreadable"].append(rel)
            continue
        if not isinstance(data, dict) or "meta" not in data or "records" not in data:
            continue
        meta = data["meta"]
        if (meta.get("labels_sha256") != label_hash
                or meta.get("runner_code_sha256") != runner_hash):
            audit["stale"].append(rel)
            continue
        data["_path"] = path
        data["_name"] = os.path.basename(path)
        runs.append(data)
        audit["accepted"].append(rel)

    audit["labels_sha256"] = label_hash
    audit["runner_code_sha256"] = runner_hash
    return runs, audit
