from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from .common import CONFIG, REPO, ROOT, atomic_json, load_config, markdown_report, now_utc, sha256_file

CORE_SOURCES = (
    "src/gridagent/agent_interface/llm_agent.py",
    "src/gridagent/agent_interface/restoration_env.py",
    "src/gridagent/agent_interface/scripts/run_episodes.py",
    "src/gridagent/system/llm_proposer.py",
    "src/gridagent/oracle/reconfig_search.py",
    "src/gridagent/oracle/restoration_oracle.py",
    "src/gridagent/oracle/obs_ceiling.py",
    "src/gridagent/simulation/pf_feasibility.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, str | None]:
    names = (
        "torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub",
        "tokenizers", "pandapower", "pandas", "numpy", "scipy", "PyYAML",
    )
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def snapshot_path(model_id: str, revision: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    model_dir = cache / ("models--" + model_id.replace("/", "--"))
    candidate = model_dir / "snapshots" / revision
    return candidate


def model_entry(slug: str, spec: dict) -> dict:
    revision = str(spec["revision"])
    candidate = snapshot_path(spec["model_id"], revision)
    cached = revision != "resolve-on-download" and candidate.is_dir()
    files = []
    total = 0
    if cached:
        for path in candidate.rglob("*"):
            if path.is_file():
                try:
                    size = path.resolve().stat().st_size
                except FileNotFoundError:
                    continue
                total += size
                files.append(str(path.relative_to(candidate)))
    return {
        "slug": slug,
        **spec,
        "cached": bool(cached),
        "snapshot_path": str(candidate) if cached else None,
        "snapshot_bytes": total,
        "snapshot_file_count": len(files),
        "has_config": cached and (candidate / "config.json").exists(),
        "has_tokenizer": cached and any((candidate / name).exists() for name in ("tokenizer.json", "tokenizer.model")),
    }


def discover_cached_repositories(config: dict) -> list[dict]:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    selected = {spec["model_id"]: slug for slug, spec in config["models"].items()}
    repositories = []
    for model_dir in sorted(hub.glob("models--*")):
        if not model_dir.is_dir():
            continue
        encoded = model_dir.name.removeprefix("models--")
        model_id = encoded.replace("--", "/", 1)
        snapshots = []
        for snapshot in sorted((model_dir / "snapshots").glob("*")):
            if not snapshot.is_dir():
                continue
            total = 0
            has_weight_file = False
            for path in snapshot.rglob("*"):
                if path.is_file():
                    has_weight_file = has_weight_file or path.name.endswith(
                        (".safetensors", ".bin", ".gguf")
                    )
                    try:
                        total += path.resolve().stat().st_size
                    except FileNotFoundError:
                        pass
            config_path = snapshot / "config.json"
            model_type = None
            architectures = []
            if config_path.exists():
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                model_type = payload.get("model_type")
                architectures = list(payload.get("architectures") or [])
            snapshots.append(
                {
                    "revision": snapshot.name,
                    "snapshot_bytes": total,
                    "model_type": model_type,
                    "architectures": architectures,
                    "has_weight_file": has_weight_file,
                }
            )
        max_bytes = max((row["snapshot_bytes"] for row in snapshots), default=0)
        registered_slug = selected.get(model_id)
        generative = any(
            "CausalLM" in architecture or "ConditionalGeneration" in architecture
            for row in snapshots
            for architecture in row["architectures"]
        )
        has_weights = any(row["has_weight_file"] for row in snapshots)
        if registered_slug:
            disposition = f"registered as {registered_slug}"
        elif generative and not has_weights:
            disposition = "excluded: incomplete metadata-only cache"
        elif generative and max_bytes > 38 * 1024**3:
            disposition = "excluded: cached precision exceeds a single 40GB A100 inference budget"
        elif generative:
            disposition = "excluded: not part of the frozen unified cohort"
        else:
            disposition = "excluded: non-generative encoder/backbone"
        repositories.append(
            {
                "model_id": model_id,
                "registered_slug": registered_slug,
                "generative": generative,
                "disposition": disposition,
                "snapshots": snapshots,
            }
        )
    return repositories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dirty_patch = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    manifest = {
        "schema_version": "neurips-grid-provenance-v1",
        "created_utc": now_utc(),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty_patch_sha256": __import__("hashlib").sha256(dirty_patch.encode()).hexdigest(),
        "untracked_paths": untracked,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "core_source_sha256": {
            path: sha256_file(ROOT / path) for path in CORE_SOURCES
        },
        "experiment_source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "src").rglob("*.py"))
        },
        "packages": package_versions(),
        "models": [model_entry(slug, spec) for slug, spec in config["models"].items()],
        "all_cached_model_repositories": discover_cached_repositories(config),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    output = ROOT / "artifacts/provenance/provenance.json"
    atomic_json(output, manifest)
    missing = [entry["slug"] for entry in manifest["models"] if not entry["cached"] and not entry.get("conditional")]
    report = markdown_report(
        "inventory",
        f"models{len(manifest['models'])}",
        [
            "# FLARE inventory",
            "",
            f"- Git HEAD: `{manifest['git_head']}`",
            f"- Configuration: `{manifest['config_sha256']}`",
            f"- Registered models: {len(manifest['models'])}",
            f"- All cached Hugging Face model repositories inspected: {len(manifest['all_cached_model_repositories'])}",
            f"- Missing required cached models: {', '.join(missing) if missing else 'none'}",
            f"- Machine-readable manifest: `{output}`",
            "",
            "| Cached repository | Disposition |",
            "|---|---|",
            *(
                f"| `{entry['model_id']}` | {entry['disposition']} |"
                for entry in manifest["all_cached_model_repositories"]
            ),
        ],
    )
    print(json.dumps({"manifest": str(output), "report": str(report), "missing": missing}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
