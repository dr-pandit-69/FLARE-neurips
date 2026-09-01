from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .common import CONFIG, ROOT, atomic_json, atomic_text, load_config, markdown_report, now_utc


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def metric(row: dict, name: str) -> float:
    if name == "restored_fraction":
        denominator = float(row.get("restorable_customers", 0) or 0)
        return float(row.get("restored_customers", 0) or 0) / denominator if denominator else 0.0
    if name == "observation_gap":
        if row.get("regret_obs_gap") is not None:
            return float(row["regret_obs_gap"])
        return float(row.get("regret_omni", 0.0)) - float(row.get("regret_reasoning_gap", 0.0))
    if name == "reasoning_gap":
        if row.get("regret_reasoning_gap") is not None:
            return float(row["regret_reasoning_gap"])
        return float(row.get("regret_omni", 0.0)) - metric(row, "observation_gap")
    return float(row.get(name, 0.0) or 0.0)


def summarize(rows: list[dict], threshold: float) -> dict:
    regrets = [metric(row, "regret_omni") for row in rows]
    p95 = quantile(regrets, 0.95)
    tail_n = max(1, int(math.ceil(len(regrets) * 0.05))) if regrets else 0
    tail95 = sorted(regrets, reverse=True)[:tail_n]
    ceiling_proven = [row for row in rows if bool(row.get("obs_ceiling_proven_bound"))]
    worst_n = max(1, int(math.ceil(len(regrets) * 0.01))) if regrets else 0
    return {
        "n": len(rows),
        "mean_regret": mean(regrets),
        "median_regret": quantile(regrets, 0.5),
        "p90_regret": quantile(regrets, 0.9),
        "p95_regret": p95,
        "cvar95_regret": mean(tail95),
        "worst_1pct_regret": mean(sorted(regrets, reverse=True)[:worst_n]) if worst_n else None,
        "prob_regret_gt_threshold": mean([float(value > threshold) for value in regrets]),
        "mean_restored_fraction": mean([metric(row, "restored_fraction") for row in rows]),
        "complete_restore_rate": mean([
            float(int(row.get("restored_customers", 0)) >= int(row.get("restorable_customers", 0)))
            for row in rows
        ]),
        "hardfail_rate": mean([metric(row, "flisr_hardfail") for row in rows]),
        "mean_blocked_attempts": mean([metric(row, "blocked_total") for row in rows]),
        "mean_shadow_unsafe_attempts": mean([metric(row, "shadow_unsafe_attempts") for row in rows]),
        "malformed_reply_rate": mean([float(metric(row, "malformed_reply") > 0) for row in rows]),
        "observation_ceiling_proven_fraction": (
            len(ceiling_proven) / len(rows) if rows else None
        ),
        "n_observation_ceiling_proven": len(ceiling_proven),
        "mean_observation_gap_bound_all": mean([metric(row, "observation_gap") for row in rows]),
        "mean_reasoning_gap_bound_all": mean([metric(row, "reasoning_gap") for row in rows]),
        "mean_observation_gap_proven": mean([
            metric(row, "observation_gap") for row in ceiling_proven
        ]),
        "mean_reasoning_gap_proven": mean([
            metric(row, "reasoning_gap") for row in ceiling_proven
        ]),
    }


def cluster_interval(rows: list[dict], field: str, reps: int, seed: int) -> dict:
    by_block: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        block = str(row.get("dependence_block_id") or row.get("case_id"))
        by_block[block].append(metric(row, field))
    blocks = sorted(by_block)
    if not blocks:
        return {"mean": None, "low": None, "high": None, "n_blocks": 0}
    observed_values = [value for block in blocks for value in by_block[block]]
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sampled = [rng.choice(blocks) for _ in blocks]
        values = [value for block in sampled for value in by_block[block]]
        draws.append(statistics.fmean(values))
    return {
        "mean": statistics.fmean(observed_values),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_blocks": len(blocks),
        "method": "dependence-block cluster bootstrap",
    }


def _cluster_bootstrap_draws(
    values_by_block: dict[str, list[float]], reps: int, seed: int
) -> list[float]:
    blocks = sorted(values_by_block)
    if not blocks:
        return []
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sampled = [rng.choice(blocks) for _ in blocks]
        values = [value for block in sampled for value in values_by_block[block]]
        draws.append(statistics.fmean(values))
    return draws


def paired_cluster_contrast(
    reference: list[dict], treatment: list[dict], field: str, reps: int, seed: int
) -> dict:
    """Treatment-minus-reference contrast matched on physical case and tier."""
    reference_by_key = {
        (str(row["case_id"]), str(row["tier"])): row for row in reference
    }
    values_by_block: dict[str, list[float]] = defaultdict(list)
    for row in treatment:
        key = (str(row["case_id"]), str(row["tier"]))
        base = reference_by_key.get(key)
        if base is None:
            continue
        block = str(row.get("dependence_block_id") or base.get("dependence_block_id") or key[0])
        values_by_block[block].append(metric(row, field) - metric(base, field))
    observed = [value for values in values_by_block.values() for value in values]
    draws = _cluster_bootstrap_draws(values_by_block, reps, seed)
    estimate = mean(observed)
    if not draws:
        return {
            "estimate": estimate,
            "low": None,
            "high": None,
            "n_pairs": len(observed),
            "n_blocks": len(values_by_block),
            "randomization_p": None,
        }

    # Cluster sign-flip randomization keeps all tier replays from one dependence block together.
    rng = random.Random(seed + 1)
    block_values = [values_by_block[block] for block in sorted(values_by_block)]
    null_draws = []
    for _ in range(reps):
        signed = [
            sign * value
            for values in block_values
            for sign in [rng.choice((-1.0, 1.0))]
            for value in values
        ]
        null_draws.append(statistics.fmean(signed))
    p_value = (
        1 + sum(abs(value) >= abs(float(estimate)) for value in null_draws)
    ) / (reps + 1)
    return {
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_pairs": len(observed),
        "n_blocks": len(values_by_block),
        "randomization_p": float(p_value),
        "method": "paired dependence-block bootstrap and cluster sign-flip test",
    }


def independent_cluster_contrast(
    reference: list[dict], treatment: list[dict], field: str, reps: int, seed: int
) -> dict:
    """Treatment-minus-reference shift for panels with different physical cases."""
    by_ref: dict[str, list[float]] = defaultdict(list)
    by_treatment: dict[str, list[float]] = defaultdict(list)
    for row in reference:
        by_ref[str(row.get("dependence_block_id") or row.get("case_id"))].append(metric(row, field))
    for row in treatment:
        by_treatment[str(row.get("dependence_block_id") or row.get("case_id"))].append(metric(row, field))
    ref_values = [value for values in by_ref.values() for value in values]
    treatment_values = [value for values in by_treatment.values() for value in values]
    ref_draws = _cluster_bootstrap_draws(by_ref, reps, seed)
    treatment_draws = _cluster_bootstrap_draws(by_treatment, reps, seed + 1)
    draws = [right - left for left, right in zip(ref_draws, treatment_draws)]
    estimate = (
        statistics.fmean(treatment_values) - statistics.fmean(ref_values)
        if ref_values and treatment_values
        else None
    )
    return {
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)) if draws else None,
        "high": float(np.quantile(draws, 0.975)) if draws else None,
        "n_reference": len(ref_values),
        "n_treatment": len(treatment_values),
        "n_reference_blocks": len(by_ref),
        "n_treatment_blocks": len(by_treatment),
        "method": "independent dependence-block bootstrap",
    }


def _bh_q_values(named_p_values: dict[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, float(value)) for name, value in named_p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    result: dict[str, float | None] = {name: None for name in named_p_values}
    running = 1.0
    total = len(available)
    for rank_from_end in range(total - 1, -1, -1):
        name, p_value = available[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, p_value * total / rank)
        result[name] = float(min(1.0, running))
    return result


def reliability_summary(cells: dict[str, dict]) -> dict:
    seeds = sorted(name for name in cells if name.startswith("reliability_seed"))
    if not seeds:
        return {"n_seeds": 0, "complete": False, "design": "single_fixed_seed"}
    per_seed = {
        name: {
            (str(row["case_id"]), str(row["tier"])): metric(row, "regret_omni")
            for row in cells[name]["records"]
        }
        for name in seeds
    }
    common = sorted(set.intersection(*(set(values) for values in per_seed.values())))
    seed_means = {
        name: mean([per_seed[name][key] for key in common]) for name in seeds
    }
    if len(seeds) == 1:
        return {
            "n_seeds": 1,
            "complete": True,
            "design": "single_fixed_seed",
            "n_matched_case_tiers": len(common),
            "seed_mean_regret": seed_means,
            "between_seed_variance_estimated": False,
        }
    case_sds = [
        statistics.stdev([per_seed[name][key] for name in seeds]) for key in common
    ]
    correlations = []
    for left, right in combinations(seeds, 2):
        x = np.asarray([per_seed[left][key] for key in common], dtype=float)
        y = np.asarray([per_seed[right][key] for key in common], dtype=float)
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "n_seeds": len(seeds),
        "complete": True,
        "design": "supplementary_multi_seed",
        "n_matched_case_tiers": len(common),
        "mean_within_case_sd": mean(case_sds),
        "p95_within_case_sd": quantile(case_sds, 0.95),
        "mean_pairwise_pearson": mean(correlations),
        "min_pairwise_pearson": min(correlations) if correlations else None,
        "seed_mean_regret": seed_means,
        "range_seed_mean_regret": (
            max(seed_means.values()) - min(seed_means.values()) if seed_means else None
        ),
    }


def safety_taxonomy(cells: dict[str, dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    unsafe = 0
    for artifact in cells.values():
        for event in artifact.get("safety_events", []):
            total += 1
            unsafe += int(bool(event.get("unsafe")))
            for name in event.get("classes", []):
                counts[str(name)] += 1
    return {
        "n_guard_rejected_actions_audited": total,
        "n_physically_unsafe_counterfactuals": unsafe,
        "unsafe_fraction": unsafe / total if total else 0.0,
        "class_counts": dict(sorted(counts.items())),
    }


def load_artifacts(root: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for model_dir in sorted((root / "models").glob("*")):
        if not model_dir.is_dir():
            continue
        cells = {}
        for path in sorted(model_dir.glob("*.json")):
            if path.name in {"runtime.json", "worker_complete.json"}:
                continue
            if path.stem.startswith("reliability_seed") and path.stem != "reliability_seed0":
                continue
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("meta", {}).get("completed"):
                cells[path.stem] = artifact
        if cells:
            result[model_dir.name] = cells
    return result


def select_models(
    artifacts: dict[str, dict[str, dict]], model_arg: str
) -> dict[str, dict[str, dict]]:
    """Return an explicit complete-paper cohort without leaking partial models."""
    if model_arg == "all":
        return artifacts
    requested = [value.strip() for value in model_arg.split(",") if value.strip()]
    if not requested:
        raise ValueError("--models must name at least one model")
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate models requested: {duplicates}")
    missing = sorted(set(requested) - set(artifacts))
    if missing:
        raise ValueError(f"models have no completed artifacts: {missing}")
    return {name: artifacts[name] for name in requested}


def write_figure(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted(summary["models"])
    tiers = ["clean", "sparse", "noisy", "stale", "conflicting"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for model in models:
        cell = summary["models"][model].get("primary_A0", {})
        values = [cell.get(tier, {}).get("mean_regret") for tier in tiers]
        if all(value is not None for value in values):
            axes[0].plot(tiers, values, marker="o", label=model)
    axes[0].set_ylabel("Mean controllable-ENS regret")
    axes[0].set_xlabel("Observation tier")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    observation = []
    reasoning = []
    labels = []
    for model in models:
        rows = summary["models"][model].get("primary_A0", {}).get("all", {})
        if rows.get("mean_observation_gap_proven") is not None:
            labels.append(model)
            observation.append(rows["mean_observation_gap_proven"])
            reasoning.append(rows["mean_reasoning_gap_proven"])
    x = np.arange(len(labels))
    axes[1].bar(x, observation, label="observation gap")
    axes[1].bar(x, reasoning, bottom=observation, label="reasoning/planning gap")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("Mean regret decomposition")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/full")
    parser.add_argument("--models", default="all")
    parser.add_argument("--reps", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    reps = int(args.reps or config["analysis"]["bootstrap_replicates"])
    threshold = float(config["analysis"]["tail_threshold"])
    artifacts = select_models(load_artifacts(args.root), args.models)
    if not artifacts:
        raise SystemExit(f"no completed model artifacts under {args.root}")
    summary = {
        "schema_version": "neurips-grid-analysis-v1",
        "created_utc": now_utc(),
        "bootstrap_replicates": reps,
        "tail_threshold": threshold,
        "model_scope": list(artifacts),
        "models": {},
        "intervals": {},
        "paired_ablation_contrasts": {},
        "distribution_shift_contrasts": {},
        "reliability": {},
        "safety_taxonomy": {},
    }
    for model, cells in artifacts.items():
        summary["models"][model] = {}
        summary["intervals"][model] = {}
        for cell, artifact in cells.items():
            rows = artifact["records"]
            tiers = sorted({str(row["tier"]) for row in rows})
            summary["models"][model][cell] = {
                tier: summarize([row for row in rows if row["tier"] == tier], threshold)
                for tier in tiers
            }
            summary["models"][model][cell]["all"] = summarize(rows, threshold)
            summary["intervals"][model][cell] = {
                field: cluster_interval(rows, field, reps, int(stable_seed(model, cell, field)))
                for field in ("regret_omni", "restored_fraction", "shadow_unsafe_attempts")
            }
        reference = cells.get("primary_A0", {}).get("records", [])
        ablations = (
            "true_state",
            "no_tie_ranking",
            "no_simulator",
            "no_memory",
            "schema_only",
        )
        contrasts = {}
        for cell in ablations:
            treatment = cells.get(cell, {}).get("records", [])
            if not reference or not treatment:
                continue
            contrasts[cell] = paired_cluster_contrast(
                reference,
                treatment,
                "regret_omni",
                reps,
                stable_seed(model, cell, "paired_regret"),
            )
        q_values = _bh_q_values(
            {name: row.get("randomization_p") for name, row in contrasts.items()}
        )
        for name, q_value in q_values.items():
            contrasts[name]["bh_q_within_model_ablation_family"] = q_value
        summary["paired_ablation_contrasts"][model] = contrasts
        shift_contrasts = {}
        for cell in (
            "robustness_A0",
            "topology_ood_A0",
            "generation_ood_A0",
            "load_ood_A0",
            "morphology_rural_A0",
            "morphology_semiurban_A0",
            "morphology_urban_A0",
            "morphology_commercial_A0",
        ):
            treatment = cells.get(cell, {}).get("records", [])
            if not reference or not treatment:
                continue
            shift_contrasts[cell] = independent_cluster_contrast(
                reference,
                treatment,
                "regret_omni",
                reps,
                stable_seed(model, cell, "independent_regret"),
            )
        summary["distribution_shift_contrasts"][model] = shift_contrasts
        summary["reliability"][model] = reliability_summary(cells)
        summary["safety_taxonomy"][model] = safety_taxonomy(cells)
    analysis_dir = ROOT / "artifacts/analysis"
    atomic_json(analysis_dir / "summary.json", summary)
    failure_cases = {
        model: {
            cell: [
                {
                    key: row.get(key)
                    for key in (
                        "case_id",
                        "tier",
                        "dependence_block_id",
                        "regret_omni",
                        "regret_obs_gap",
                        "regret_reasoning_gap",
                        "restored_customers",
                        "restorable_customers",
                        "flisr_hardfail",
                        "blocked_total",
                        "shadow_unsafe_attempts",
                        "malformed_reply",
                        "last_action_result",
                    )
                }
                for row in sorted(
                    artifact["records"],
                    key=lambda value: (
                        -metric(value, "regret_omni"),
                        str(value.get("case_id")),
                        str(value.get("tier")),
                    ),
                )[:10]
            ]
            for cell, artifact in cells.items()
        }
        for model, cells in artifacts.items()
    }
    atomic_json(analysis_dir / "failure_cases.json", failure_cases)
    write_figure(summary, analysis_dir / "figures/results_overview.pdf")
    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $n$ & Regret [95\% CI] & P95 & CVaR95 & Restored & Unsafe attempts \\",
        r"\midrule",
    ]
    for model in sorted(summary["models"]):
        row = summary["models"][model].get("primary_A0", {}).get("all")
        if not row:
            continue
        escaped = model.replace("_", r"\_")
        interval = summary["intervals"][model]["primary_A0"]["regret_omni"]
        regret_with_ci = (
            f"{row['mean_regret']:.3f} [{interval['low']:.3f},{interval['high']:.3f}]"
        )
        table.append(
            f"{escaped} & {row['n']} & {regret_with_ci} & {row['p95_regret']:.3f} & "
            f"{row['cvar95_regret']:.3f} & {row['mean_restored_fraction']:.3f} & "
            f"{row['mean_shadow_unsafe_attempts']:.2f} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}", ""])
    atomic_text(analysis_dir / "tables/primary_summary.tex", "\n".join(table))
    report_lines = [
        "# FLARE analysis",
        "",
        f"Dependence-block bootstrap replicates: {reps}.",
        "",
        "| Model | Cell | n | Mean regret | P95 | CVaR95 | Restored fraction | Hard-fail | Shadow unsafe/episode | Obs-bound proof |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in sorted(summary["models"]):
        for cell in sorted(summary["models"][model]):
            row = summary["models"][model][cell]["all"]
            report_lines.append(
                f"| {model} | {cell} | {row['n']} | {row['mean_regret']:.4f} | "
                f"{row['p95_regret']:.4f} | {row['cvar95_regret']:.4f} | "
                f"{row['mean_restored_fraction']:.4f} | {row['hardfail_rate']:.4f} | "
                f"{row['mean_shadow_unsafe_attempts']:.4f} | "
                f"{row['observation_ceiling_proven_fraction']:.4f} |"
            )
    report = markdown_report("analysis", f"models{len(artifacts)}", report_lines)
    print(json.dumps({"summary": str(analysis_dir / 'summary.json'), "report": str(report)}, indent=2))
    return 0


def stable_seed(*parts: str) -> int:
    return int(__import__("hashlib").sha256("|".join(parts).encode()).hexdigest()[:8], 16)


if __name__ == "__main__":
    raise SystemExit(main())
