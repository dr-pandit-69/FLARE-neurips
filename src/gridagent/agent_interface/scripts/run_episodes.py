"""GridAgent-Bench v2 — Section-4 episode runner (REAL rollouts; frozen CLI contract).

Every arm is driven through the REAL ``RestorationEpisode`` and scored by the ONE shared cost model:
  * non-LLM arms -> ``baselines.restoration_policies.rollout_arm``
  * llm arm      -> ``agent_interface.llm_agent.run_llm_episode``
Cost/regret come from the ACTUAL restored bus-set. Every record carries provenance="rollout".

v2 changes
----------
* Reads the v2 oracle labels (multi-step plan, belief ceiling per observation tier, measured
  difficulty features) and merges the PER-TIER label block into each record.
* Only ``noop``/``random``/``greedy``/``flisr_ideal``/``oracle`` are tier-invariant (they act on the
  true topology). ``flisr_operator`` and ``obs_ceiling`` consume the DEGRADED observations, so they
  are rolled out separately per tier — that is the whole point of the fair incumbent (M10).
* ``_load_slate`` stratifies across the measured D0-D4 difficulty ladder and reports the counts.
* ORACLE_BEATEN guard (M1): the oracle arm must score exactly 0, and no arm may have negative
  UNCLIPPED regret. Either condition is recorded as a build-breaking flag on the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pandapower as pp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
SUB = os.environ.get("GA_SUBSTRATE", os.path.join(REPO, "data", "processed", "substrate_v1"))
GRIDS = os.path.join(SUB, "distribution_grids")
LABELS = os.path.join(SUB, "oracle", "oracle_labels_v2.jsonl")
ALL_TIERS = ["clean", "sparse", "noisy", "stale", "conflicting"]
TIER_INVARIANT = ["noop", "random", "greedy", "flisr_ideal", "oracle"]
TIER_DEPENDENT = [
    "flisr_operator",
    "belief_planner",
    "belief_planner_no_sim",
    "obs_ceiling",
]
NONLLM_ARMS = TIER_INVARIANT + TIER_DEPENDENT
DIFFICULTIES = ["D0", "D1", "D2", "D3", "D4"]
REF_TIER = "noisy"          # the shipped tier the slate is stratified on

from gridagent.agent_interface.baselines.restoration_policies import rollout_arm  # noqa: E402
from gridagent.agent_interface.restoration_env import (  # noqa: E402
    DEFAULT_MAX_COMMITTED_OPS,
    DEFAULT_MAX_SECTIONALIZER_OPENS,
    Fault,
)
from gridagent.analysis.run_artifacts import runner_code_sha256, sha256_file  # noqa: E402
from gridagent.oracle.restoration_oracle import CLASS_WEIGHTS  # noqa: E402
from gridagent.simulation.grid_bank import stable_seed  # noqa: E402


def _load_cust_and_weights():
    df = pd.read_parquet(os.path.join(SUB, "series", "bus_allocation.parquet"))
    cust = {g: {int(b): int(c) for b, c in zip(v["bus"], v["customers"])}
            for g, v in df.groupby("grid_id")}
    wts = {g: {int(b): float(CLASS_WEIGHTS.get(str(k), 1.0))
               for b, k in zip(v["bus"], v["customer_class"])}
           for g, v in df.groupby("grid_id")}
    return cust, wts


def _load_sw_meta():
    sm = pd.read_csv(os.path.join(GRIDS, "switch_manifest.csv"))
    return {g: {int(r.switch_id): {"remote": bool(r.remote), "travel_time_min": float(r.travel_time_min),
                                   "type": str(r.type)} for r in v.itertuples()}
            for g, v in sm.groupby("grid_id")}


def _load_slate(n_per_stratum: int, seed: int, labels_path: str = LABELS):
    """Stratified slate across the MEASURED D0-D4 ladder (at the reference observation tier).

    Only SCORABLE episodes are eligible: those where some feasible reconfiguration restores at
    least one customer. Episodes with nothing restorable are degenerate — every arm scores
    identically — and including them would dilute every contrast.
    """
    rng = np.random.default_rng(seed)
    strata: dict = {}
    for line in open(labels_path):
        if not line.strip():
            continue
        r = json.loads(line)
        if (int(r.get("controllable_customers", 0)) <= 0
                or not bool(r.get("initial_pf_feasible", False))):
            continue
        d = r["by_tier"][REF_TIER]["difficulty_tier"]
        strata.setdefault(d, []).append(r)
    excluded_ids = set()
    if seed == 1:
        tuned, _counts, _pool = _load_slate(n_per_stratum, 0, labels_path)
        excluded_ids = {r["case_id"] for r in tuned}

    slate, counts = [], {}
    for d in DIFFICULTIES:
        items = [r for r in strata.get(d, []) if r["case_id"] not in excluded_ids]
        if not items:
            counts[d] = 0
            continue
        k = min(n_per_stratum, len(items))
        idx = rng.choice(len(items), size=k, replace=False)
        for i in sorted(idx):
            slate.append(items[i])
        counts[d] = k
    return slate, counts, {d: len(v) for d, v in strata.items()}


def _resolve_manifest_path(value: str, manifest_path: str) -> str:
    path = value if os.path.isabs(value) else os.path.join(
        os.path.dirname(os.path.abspath(manifest_path)), value
    )
    return os.path.abspath(path)


def _load_panel_slate(panel_manifest_path: str, labels_path: str | None = None):
    """Load exact manifest IDs in order and verify every frozen input hash."""
    manifest_path = os.path.abspath(panel_manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    sidecar = os.path.splitext(manifest_path)[0] + ".sha256"
    if os.path.exists(sidecar):
        expected_manifest_hash = open(sidecar).read().strip().split()[0]
        if expected_manifest_hash != manifest_sha256:
            raise ValueError(
                f"panel manifest hash mismatch: sidecar={expected_manifest_hash}, "
                f"actual={manifest_sha256}"
            )
    manifest = json.load(open(manifest_path))
    declared_labels_path = _resolve_manifest_path(
        manifest["filtered_labels_path"], manifest_path
    )
    chosen_labels_path = os.path.abspath(labels_path or declared_labels_path)
    chosen_labels_hash = sha256_file(chosen_labels_path)
    if chosen_labels_hash != manifest["filtered_labels_sha256"]:
        raise ValueError(
            f"panel labels hash mismatch: manifest={manifest['filtered_labels_sha256']}, "
            f"actual={chosen_labels_hash}"
        )
    source_labels_path = _resolve_manifest_path(
        manifest["source_labels_path"], manifest_path
    )
    if sha256_file(source_labels_path) != manifest["source_labels_sha256"]:
        raise ValueError("panel source-label hash mismatch")
    split_path = _resolve_manifest_path(
        manifest["split_manifest_path"], manifest_path
    )
    if sha256_file(split_path) != manifest["split_manifest_sha256"]:
        raise ValueError("panel split-manifest hash mismatch")

    rows = {
        row["case_id"]: row
        for line in open(chosen_labels_path)
        if line.strip()
        for row in [json.loads(line)]
    }
    ordered_ids = [episode["case_id"] for episode in manifest["episodes"]]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("panel manifest contains duplicate case IDs")
    missing = [case_id for case_id in ordered_ids if case_id not in rows]
    if missing:
        raise ValueError(f"panel labels missing manifest IDs: {missing[:5]}")
    slate = [rows[case_id] for case_id in ordered_ids]
    counts = dict(Counter(
        row["by_tier"][REF_TIER]["difficulty_tier"] for row in slate
    ))
    info = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "labels_path": chosen_labels_path,
        "labels_sha256": chosen_labels_hash,
        "split_manifest_path": split_path,
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "dependence_definition_sha256": manifest[
            "dependence_definition_sha256"
        ],
    }
    return slate, counts, counts.copy(), info


_NET_CACHE: dict = {}


def _net(gid):
    if gid not in _NET_CACHE:
        _NET_CACHE[gid] = pp.from_json(os.path.join(GRIDS, f"{gid}.json"))
    return _NET_CACHE[gid]


def _tier_label(orc: dict, tier: str) -> dict:
    """Flatten the structural label + the tier-specific belief-ceiling block into one dict."""
    lab = {k: v for k, v in orc.items() if k != "by_tier"}
    lab.update(orc["by_tier"][tier])
    return lab


def _record_common(orc, tier):
    record = {
        "case_id": orc["case_id"],
        "grid_id": orc["grid_id"],
        "feeder_year": orc["feeder_year"],
        "variant_group_id": orc["variant_group_id"],
        "is_multifault": orc.get("is_multifault", False),
        "tier": tier,
    }
    for field in (
        "event_id",
        "event_burst_id",
        "weather_episode_id",
        "dependence_block_id",
        "dependence_block_type",
        "dependence_definition_sha256",
    ):
        if field in orc:
            record[field] = orc.get(field)
    if "storm_cluster_id" in orc:
        record["storm_cluster_id"] = orc["storm_cluster_id"]
        record["storm_window"] = orc["storm_cluster_id"]
    return record


def _rollout_seed(case_id: str, arm: str, seed: int, tier: str = "") -> int:
    """Return the matched public-observation seed for a paired comparison cell.

    ``arm`` remains in the signature for compatibility with existing callers, but
    it is intentionally excluded from the seed. Competing arms must observe the
    same degraded evidence for a fixed case, observation tier, and rollout seed.
    """
    del arm
    return stable_seed(("rollout", case_id, tier, int(seed)))


def _artifact_generation_id(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _atomic_write_artifact(path: str, artifact: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(artifact, handle)
    os.replace(temporary, path)


def _resume_records(path: str, generation_id: str, *, force: bool) -> list[dict]:
    if force or not os.path.exists(path):
        return []
    try:
        artifact = json.load(open(path))
    except Exception as exc:
        raise ValueError(f"cannot read partial artifact {path}: {exc}") from exc
    meta = artifact.get("meta", {})
    if meta.get("completed"):
        return list(artifact.get("records", []))
    if meta.get("artifact_generation_id") != generation_id:
        raise ValueError(
            "partial artifact generation does not match the current frozen inputs"
        )
    return list(artifact.get("records", []))


def _completed_artifact_is_current(
    path: str, generation_id: str, *, force: bool
) -> bool:
    """Return whether a completed artifact matches the exact requested generation."""
    if force or not os.path.exists(path):
        return False
    try:
        artifact = json.load(open(path))
    except Exception as exc:
        raise ValueError(f"cannot read existing artifact {path}: {exc}") from exc
    meta = artifact.get("meta", {})
    if not meta.get("completed"):
        return False
    if not artifact.get("records"):
        raise ValueError(f"completed artifact has no records: {path}")
    if meta.get("artifact_generation_id") != generation_id:
        raise ValueError(
            f"completed artifact is stale for the requested generation: {path}"
        )
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_episodes")
    ap.add_argument("--arm", required=True, choices=NONLLM_ARMS + ["llm", "unsafe_probe"])
    ap.add_argument("--tiers", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="backward-compatible v3 rollout-seed alias")
    ap.add_argument("--rollout-seed", type=int, default=None)
    ap.add_argument("--decoding-seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--true-state", action="store_true")
    ap.add_argument(
        "--briefing",
        choices=("full", "minimal", "schema_only"),
        default="full",
    )
    ap.add_argument(
        "--rank-ties",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--ablation", default="A0")
    ap.add_argument("--slate-n", type=int, default=int(os.environ.get("GA_SLATE_N", "8")))
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--panel-manifest", default=None)
    ap.add_argument("--belief-planner-config", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.panel_manifest and args.seed is not None:
        raise SystemExit(
            "ABORT: --seed is a v3 alias; panel-based v4 runs require --rollout-seed"
        )
    if (
        args.seed is not None
        and args.rollout_seed is not None
        and args.seed != args.rollout_seed
    ):
        raise SystemExit("ABORT: --seed and --rollout-seed disagree")
    rollout_seed = (
        args.rollout_seed
        if args.rollout_seed is not None
        else (args.seed if args.seed is not None else 0)
    )
    if args.temperature < 0.0:
        raise SystemExit("ABORT: --temperature must be nonnegative")
    if not 0.0 < args.top_p <= 1.0:
        raise SystemExit("ABORT: --top-p must lie in (0, 1]")
    if args.checkpoint_every <= 0:
        raise SystemExit("ABORT: --checkpoint-every must be positive")

    tiers = ALL_TIERS if args.tiers == "all" else args.tiers.split(",")
    if not tiers or any(tier not in ALL_TIERS for tier in tiers):
        raise SystemExit(f"ABORT: invalid tier list {tiers}")
    panel_info = None
    if args.panel_manifest:
        try:
            slate, slate_counts, pool_counts, panel_info = _load_panel_slate(
                args.panel_manifest, args.labels
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"ABORT: invalid panel manifest: {exc}") from exc
        labels_path = panel_info["labels_path"]
    else:
        labels_path = args.labels or LABELS
        slate, slate_counts, pool_counts = _load_slate(
            args.slate_n, rollout_seed, labels_path
        )
    labels_sha256 = sha256_file(labels_path)
    belief_planner_config = None
    belief_planner_config_sha256 = None
    if args.belief_planner_config:
        planner_config_path = os.path.abspath(args.belief_planner_config)
        belief_planner_config_sha256 = sha256_file(planner_config_path)
        planner_config_artifact = json.load(open(planner_config_path))
        belief_planner_config = planner_config_artifact.get(
            "selected_config", planner_config_artifact
        )
    elif args.arm in ("belief_planner", "belief_planner_no_sim") and panel_info:
        raise SystemExit(
            "ABORT: v4 belief-planner runs require --belief-planner-config"
        )
    labels_meta_path = os.path.join(os.path.dirname(labels_path), "oracle_meta.json")
    if not os.path.exists(labels_meta_path):
        raise SystemExit(f"ABORT: label metadata not found: {labels_meta_path}")
    labels_meta = json.load(open(labels_meta_path))
    declared = labels_meta.get("labels_sha256")
    if not declared:
        raise SystemExit(
            f"ABORT: {labels_meta_path} has no labels_sha256; rebuild the oracle labels")
    if declared != labels_sha256:
        raise SystemExit(
            f"ABORT: label hash mismatch: metadata={declared}, actual={labels_sha256}")
    runner_sha256 = runner_code_sha256(REPO)
    generation_spec = {
        "labels_sha256": labels_sha256,
        "runner_code_sha256": runner_sha256,
        "panel_manifest_sha256": (
            panel_info["manifest_sha256"] if panel_info else None
        ),
        "arm": args.arm,
        "model": args.model,
        "tiers": tiers,
        "rollout_seed": rollout_seed,
        "decoding_seed": args.decoding_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "true_state": args.true_state,
        "briefing": args.briefing,
        "rank_ties": args.rank_ties,
        "ablation": args.ablation,
        "max_new_tokens": args.max_new_tokens,
        "max_committed_ops": DEFAULT_MAX_COMMITTED_OPS,
        "max_sectionalizer_opens": DEFAULT_MAX_SECTIONALIZER_OPENS,
        "slate_n": args.slate_n if panel_info is None else None,
        "belief_planner_config_sha256": belief_planner_config_sha256,
    }
    generation_id = _artifact_generation_id(generation_spec)
    try:
        if _completed_artifact_is_current(
            args.out, generation_id, force=args.force
        ):
            print(f"SKIP {args.out} (completed generation matches)")
            return 0
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    cust_by, wts_by = _load_cust_and_weights()
    sw_meta_by = _load_sw_meta()
    if not slate:
        raise SystemExit(f"ABORT: empty slate from {labels_path}")
    dependence_blocks = sorted({
        str(row.get("dependence_block_id", row.get("storm_cluster_id")))
        for row in slate
    })
    records = []
    llm_runtime_meta = None

    def _fault(orc):
        return Fault(faulted_lines=[], faulted_buses=list(orc["faulted_buses"]),
                     earthing_regime=orc.get("earthing_regime") or "resonant/compensated",
                     load_scale=float(orc.get("load_scale", 1.0)))

    if args.arm in TIER_INVARIANT or args.arm == "unsafe_probe":
        for orc in slate:
            gid = orc["grid_id"]
            episode_seed = _rollout_seed(
                orc["case_id"], args.arm, rollout_seed
            )
            rng = np.random.default_rng(
                episode_seed
            )
            r = rollout_arm(args.arm, _net(gid), _fault(orc), _tier_label(orc, REF_TIER),
                            cust_by.get(gid, {}), rng, sw_meta=sw_meta_by.get(gid),
                            weights=wts_by.get(gid), tier=REF_TIER, seed=episode_seed)
            for tier in tiers:
                rec = {**_record_common(orc, tier), **r}
                rec.update({k: v for k, v in orc["by_tier"][tier].items()
                            if k in ("difficulty_tier", "belief_entropy", "regret_obs_gap",
                                     "obs_ceiling_proven_bound", "cost_obs_ceiling_kwh")})
                rec["regret_reasoning_gap"] = rec["regret_omni"] - rec.get("regret_obs_gap", 0.0)
                records.append(rec)
    elif args.arm in TIER_DEPENDENT:
        for tier in tiers:
            for orc in slate:
                gid = orc["grid_id"]
                episode_seed = _rollout_seed(
                    orc["case_id"], args.arm, rollout_seed, tier
                )
                rng = np.random.default_rng(
                    episode_seed
                )
                r = rollout_arm(args.arm, _net(gid), _fault(orc), _tier_label(orc, tier),
                                cust_by.get(gid, {}), rng, sw_meta=sw_meta_by.get(gid),
                                weights=wts_by.get(gid), tier=tier, seed=episode_seed,
                                belief_planner_config=belief_planner_config)
                records.append({**_record_common(orc, tier), **r})
            print(f"tier {tier}: {sum(1 for x in records if x['tier'] == tier)} episodes done", flush=True)
    else:  # llm
        import importlib.metadata
        import platform
        import torch
        assert torch.cuda.is_available(), "ABORT: LLM arm needs a GPU (run inside ga_gpu0/ga_gpu1)"
        if not args.model:
            raise SystemExit("ABORT: --model is required for the llm arm")
        print("GPU:", torch.cuda.get_device_name(0), "| model:", args.model, flush=True)
        from gridagent.agent_interface.llm_agent import run_llm_episode, ALL_VERBS
        from gridagent.system.llm_proposer import LocalLLM
        ABL = {
            "A0": {},
            "AB1_no_sense": {"allowed_verbs": ALL_VERBS - {"crew_inspect", "read_fault_indicator",
                                                           "test_switch"}},
            "AB2_no_simulate": {"allowed_verbs": ALL_VERBS - {"simulate_switch"}},
            "AB5_no_memory": {"use_memory": False},
            "AB6_no_tie_ranking": {"rank_ties": False},
            "AB7_minimal_briefing": {"briefing": "minimal"},
            "AB8_schema_only_briefing": {"briefing": "schema_only"},
            "A_TRUE_STATE": {"true_state": True},
        }
        if args.ablation not in ABL:
            raise SystemExit(f"unknown --ablation {args.ablation}; choose {sorted(ABL)}")
        abl_kw = {
            "rank_ties": bool(args.rank_ties),
            "true_state": bool(args.true_state),
            "briefing": args.briefing,
            **ABL[args.ablation],
        }
        torch.cuda.reset_peak_memory_stats()
        llm = LocalLLM(
            args.model,
            device="cuda:0",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            decoding_seed=args.decoding_seed,
        )
        quantization_config = getattr(
            getattr(llm.model, "config", None), "quantization_config", None
        )
        if hasattr(quantization_config, "to_dict"):
            quantization_config = quantization_config.to_dict()
        elif quantization_config is not None:
            quantization_config = str(quantization_config)
        package_versions = {"python": platform.python_version()}
        for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        llm_runtime_meta = {
            "model_type": llm.model_type,
            "loader_class": llm.loader_class,
            "tokenizer_class": type(llm.tok).__name__,
            "used_set_submodule_compat": llm.used_set_submodule_compat,
            "text_only_chat_inputs": True,
            "image_input_required": False,
            "quantization_config": quantization_config,
            "package_versions": package_versions,
        }
        try:
            records = _resume_records(
                args.out, generation_id, force=bool(args.force)
            )
        except ValueError as exc:
            raise SystemExit(f"ABORT: {exc}") from exc
        completed_pairs = {
            (row["case_id"], row["tier"]) for row in records
        }
        expected_pairs = {
            (row["case_id"], tier) for tier in tiers for row in slate
        }
        if not completed_pairs <= expected_pairs:
            raise SystemExit(
                "ABORT: partial artifact contains rows outside the frozen cell"
            )
        if len(completed_pairs) != len(records):
            raise SystemExit("ABORT: partial artifact contains duplicate rows")
        new_since_checkpoint = 0

        def checkpoint() -> None:
            llm_runtime_meta["peak_cuda_memory_allocated_gib"] = float(
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
            partial_meta = {
                "arm": "llm",
                "model": args.model,
                "ablation": args.ablation,
                "tiers": tiers,
                "rollout_seed": rollout_seed,
                "decoding_seed": args.decoding_seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_committed_ops": DEFAULT_MAX_COMMITTED_OPS,
                "max_sectionalizer_opens": (
                    DEFAULT_MAX_SECTIONALIZER_OPENS
                ),
                "panel_manifest_sha256": (
                    panel_info["manifest_sha256"] if panel_info else None
                ),
                "labels_sha256": labels_sha256,
                "split_manifest_sha256": (
                    panel_info["split_manifest_sha256"] if panel_info else None
                ),
                "dependence_definition_sha256": (
                    panel_info["dependence_definition_sha256"]
                    if panel_info else labels_meta.get(
                        "dependence_definition_sha256"
                    )
                ),
                "runner_code_sha256": runner_sha256,
                "artifact_generation_id": generation_id,
                "n_base_cases": len(slate),
                "n_episodes": len(records),
                "model_runtime": llm_runtime_meta,
                "completed": False,
                "completion_state": "running",
            }
            _atomic_write_artifact(
                args.out, {"meta": partial_meta, "records": records}
            )

        for tier in tiers:
            for orc in slate:
                pair = (orc["case_id"], tier)
                if pair in completed_pairs:
                    continue
                gid = orc["grid_id"]
                episode_rollout_seed = _rollout_seed(
                    orc["case_id"], args.arm, rollout_seed, tier
                )
                episode_decoding_seed = stable_seed(
                    (
                        "decoding",
                        orc["case_id"],
                        tier,
                        args.ablation,
                        int(args.decoding_seed),
                    )
                )
                llm.set_decoding_seed(episode_decoding_seed)
                r = run_llm_episode(llm, _net(gid), _fault(orc), _tier_label(orc, tier),
                                    cust_by.get(gid, {}), tier, sw_meta=sw_meta_by.get(gid),
                                    weights=wts_by.get(gid),
                                    seed=episode_rollout_seed, **abl_kw)
                records.append({**_record_common(orc, tier), **r})
                completed_pairs.add(pair)
                new_since_checkpoint += 1
                if new_since_checkpoint >= args.checkpoint_every:
                    checkpoint()
                    new_since_checkpoint = 0
            print(f"tier {tier}: {sum(1 for x in records if x['tier'] == tier)} episodes done", flush=True)
            checkpoint()
            new_since_checkpoint = 0
        llm_runtime_meta["peak_cuda_memory_allocated_gib"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )
        llm_runtime_meta["cuda_device_name"] = torch.cuda.get_device_name(0)

    # ---- M1 guard: the oracle scores exactly 0 and nothing beats it ----
    oracle_max = max((abs(r["regret_omni"]) for r in records), default=0.0) if args.arm == "oracle" else None
    beaten = [r for r in records if r.get("regret_raw", 0.0) < -1e-9]
    beaten_proven = [r["case_id"] for r in beaten if bool(r.get("proven_optimal", True))]
    beaten_best_known = [r["case_id"] for r in beaten if not bool(r.get("proven_optimal", True))]
    if args.arm == "oracle" and oracle_max is not None and oracle_max > 1e-9:
        print(f"WARN ORACLE_NONZERO max|regret_oracle|={oracle_max:.3e}", flush=True)

    proof_counts = {
        "oracle_proven": sum(
            bool(row.get("proven_optimal")) for row in slate
        ),
        "oracle_total": len(slate),
        "ceiling_proven_by_tier": {
            tier: sum(
                bool(row["by_tier"][tier].get("obs_ceiling_proven_bound"))
                for row in slate
            )
            for tier in tiers
        },
    }
    meta = {
        "arm": args.arm, "model": args.model if args.arm == "llm" else None,
        "ablation": args.ablation if args.arm == "llm" else None,
        "tiers": tiers,
        "seed": rollout_seed,
        "rollout_seed": rollout_seed,
        "decoding_seed": args.decoding_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0.0,
        "max_new_tokens": args.max_new_tokens,
        "max_committed_ops": DEFAULT_MAX_COMMITTED_OPS,
        "max_sectionalizer_opens": DEFAULT_MAX_SECTIONALIZER_OPENS,
        "true_state": bool(args.true_state or args.ablation == "A_TRUE_STATE"),
        "briefing": (
            "minimal"
            if args.ablation == "AB7_minimal_briefing"
            else args.briefing
        ),
        "rank_ties": bool(
            args.rank_ties and args.ablation != "AB6_no_tie_ranking"
        ),
        "slate_n": args.slate_n if panel_info is None else None,
        "slate_role": (
            panel_info["manifest"]["role"]
            if panel_info else ("heldout" if rollout_seed == 1 else "tuned")
        ),
        "substrate_version": "v4" if panel_info else "v1",
        "oracle_version": labels_meta.get("substrate_oracle_version", "v2"),
        "config_hash": json.load(open(os.path.join(GRIDS, "grid_manifest.json")))["config_hash"],
        "labels_sha256": labels_sha256,
        "panel_manifest_path": panel_info["manifest_path"] if panel_info else None,
        "panel_manifest_sha256": panel_info["manifest_sha256"] if panel_info else None,
        "split_manifest_path": panel_info["split_manifest_path"] if panel_info else None,
        "split_manifest_sha256": panel_info["split_manifest_sha256"] if panel_info else None,
        "dependence_definition_sha256": (
            panel_info["dependence_definition_sha256"]
            if panel_info else labels_meta.get("dependence_definition_sha256")
        ),
        "runner_code_sha256": runner_sha256,
        "belief_planner_config_path": (
            os.path.abspath(args.belief_planner_config)
            if args.belief_planner_config else None
        ),
        "belief_planner_config_sha256": belief_planner_config_sha256,
        "artifact_generation_id": generation_id,
        "n_episodes": len(records), "n_base_cases": len(slate),
        "n_dependence_blocks": len(dependence_blocks),
        "slate_difficulty_counts": slate_counts, "pool_difficulty_counts": pool_counts,
        "proof_counts": proof_counts,
        "scorable_contract": (
            "controllable_customers > 0 and initial_pf_feasible == true"
        ),
        "oracle_regret_max_abs": oracle_max,
        "ORACLE_BEATEN": sorted({r["case_id"] for r in beaten}),
        "PROVEN_ORACLE_BEATEN": sorted(set(beaten_proven)),
        "BEST_KNOWN_BEATEN": sorted(set(beaten_best_known)),
        "provenance": "rollout", "completed": True,
        "completion_state": "completed",
    }
    if panel_info is None:
        meta["n_storm_clusters"] = len(dependence_blocks)
        meta["n_storm_blocks"] = len(dependence_blocks)
    if llm_runtime_meta is not None:
        meta["model_runtime"] = llm_runtime_meta
    out = {"meta": meta, "records": records}
    _atomic_write_artifact(args.out, out)
    print(f"OK {args.arm} n_records={len(records)} n_base={len(slate)} "
          f"n_dependence_blocks={len(dependence_blocks)} "
          f"strata={slate_counts} beaten_proven={len(set(beaten_proven))} "
          f"beaten_best_known={len(set(beaten_best_known))} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
