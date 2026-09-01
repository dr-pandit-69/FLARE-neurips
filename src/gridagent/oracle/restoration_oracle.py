"""GridAgent-Bench v2 — restoration oracle labels.  [GOAL_v2 M5 / M6 / M8]

Produces the offline COUNTERFACTUAL ground truth attached to every episode:

  * **Connectivity-optimal, PF-validated multi-step oracle** (M5) — ``reconfig_search`` enumerates
    tie closures AND sectionalizing opens, AC-PF validates every candidate (thermal + EN 50160),
    and returns the min weighted-controllable-ENS plan together with its switching ORDER. The
    single-tie ``greedy`` plan is scored under the SAME cost model so ``greedy_suboptimality`` is a
    measured quantity, not an assertion. The claim is "connectivity-optimal, PF-validated";
    ``proven_optimal`` is true only where the DFS ran to exhaustion, and the independent
    brute-force cross-check agreement is REPORTED, never assumed.
  * **Observation-limited belief ceiling** (M6) — ``obs_ceiling`` computes, per observation tier,
    the expected cost of the best policy in a declared, exhaustively-enumerated policy class. The
    v1 hand-set 0.15/0.08 fudge is deleted.
  * **Measured difficulty features + the D0-D4 ladder** (M8) — oracle switch-op count, PF-validated
    greedy suboptimality, posterior belief entropy, concurrent-fault count, back-feed conflict.
    The tier is DERIVED from those measurements.

Operating point. Every feeder is evaluated at a calibrated pre-fault operating point: the load
scale at which the most-loaded line sits at ``TARGET_PREFAULT_LOADING_PCT`` (85 %) with the whole
network AC-PF feasible. That leaves a documented ~15 % N-1 transfer headroom, which is what makes
a back-feed capacity-constrained rather than free. The per-episode level is that base point times a
load-trajectory factor taken from the REAL FeederBW diurnal profile at the event's hour.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from gridagent.agent_interface.cost_model import DEFAULT_HORIZON_MIN, regret as _regret
from gridagent.events.dependence import validate_annotated_events
from gridagent.observation.degradation_registry import TIERS
from gridagent.oracle.obs_ceiling import obs_limited_ceiling
from gridagent.oracle.reconfig_search import (
    DEFAULT_MAX_NODES, brute_force_optimum, isolate_fault, solve_reconfiguration,
)
from gridagent.simulation.pf_feasibility import pf_feasible

TARGET_PREFAULT_LOADING_PCT = 85.0     # calibrated normal operating point (N-1 transfer headroom)
STORM_GAP_STEPS = 96                   # 24 h at 15-min resolution: the weather-window merge gap
LOAD_BINS = 3                          # off-peak / shoulder / peak, from the real diurnal profile
FINAL_MAX_OPENS = 3                    # full declared action space under max_depth=4
RETRY_MAX_NODES = 20_000               # adaptive retry for searches that hit the initial cap
MAX_MULTIFAULT_PER_GRID = 50            # keeps the shipped D3/D4 pools above curation depth
# Critical-load priority weights (D4 axis). A DECLARED MODELLING ASSUMPTION, not a calibration:
# restoration practice prioritises industrial/commercial supply over residential. Reported as such.
CLASS_WEIGHTS = {"industrial": 3.0, "commercial": 2.0, "mixed": 1.5, "residential": 1.0}
CHECKPOINT_SCHEMA = "gridagent-oracle-checkpoint-v1"


# --------------------------------------------------------------------------- #
# operating point + load trajectory
# --------------------------------------------------------------------------- #
def calibrate_operating_point(net, *, target_pct: float = TARGET_PREFAULT_LOADING_PCT,
                              lo: float = 0.02, hi: float = 3.0, iters: int = 20) -> float:
    """Largest load scale at which the pre-fault net is AC-PF feasible and its most-loaded line is
    at ``target_pct``. Bisection; deterministic."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        ok, m = pf_feasible(isolate_fault(net, [], load_scale=mid))
        if ok and (m["max_line_loading_pct"] or 0.0) <= target_pct:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


def hourly_load_profile(series_dir: str) -> list[float]:
    """24 diurnal load factors from the REAL FeederBW measurements, normalised so the peak hour
    is 1.0 (the calibrated operating point) — the load trajectory the episodes ride on."""
    df = pd.read_parquet(os.path.join(series_dir, "real_feeders.parquet"),
                         columns=["hour_of_day", "active_power_kW"])
    prof = df.groupby("hour_of_day")["active_power_kW"].mean()
    prof = prof.reindex(range(24)).interpolate().bfill().ffill()
    arr = prof.to_numpy(dtype=float)
    peak = float(np.nanmax(arr)) or 1.0
    return [round(float(x) / peak, 4) for x in arr]


def load_bin_factors(profile: list[float], n_bins: int = LOAD_BINS) -> tuple[list[float], list[float]]:
    """(per-hour bin index, per-bin representative factor). Binning keeps the oracle cache small
    while preserving the real off-peak / shoulder / peak structure."""
    arr = np.asarray(profile, dtype=float)
    qs = np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    idx = [int(np.searchsorted(qs, v, side="right")) for v in arr]
    reps = []
    for b in range(n_bins):
        vals = arr[[i == b for i in idx]]
        reps.append(round(float(vals.max()) if len(vals) else 1.0, 4))
    return idx, reps


# --------------------------------------------------------------------------- #
# real storm clusters (M8) — replaces the calendar-day index
# --------------------------------------------------------------------------- #
def assign_storm_clusters(events: list[dict], gap_steps: int = STORM_GAP_STEPS) -> dict:
    """Group events into contiguous weather-window clusters and return {event_key: cluster_id}.

    v1 used ``sw_{step // 96}`` — a CALENDAR-DAY index, so a three-day storm was three
    "independent" bootstrap blocks and the clustered CI was near-i.i.d. Here a cluster is a maximal
    run of events separated by no more than ``gap_steps`` (24 h) on the shared regional timeline:
    a storm burst that spans several days is ONE block, which is what the two-level bootstrap
    needs. Fair-weather events far from any other event become singleton clusters, as they should.
    """
    order = sorted(range(len(events)), key=lambda i: int(events[i]["step"]))
    out, cid, prev = {}, -1, None
    peak_gust = {}
    for i in order:
        e = events[i]
        s = int(e["step"])
        if prev is None or s - prev > gap_steps:
            cid += 1
            peak_gust[cid] = 0.0
        peak_gust[cid] = max(peak_gust[cid], float(e.get("gust", 0.0) or 0.0))
        out[_event_key(e)] = f"sc_{cid:04d}"
        prev = s
    return out


def _event_key(e: dict) -> str:
    return f"{e['grid_id']}_{int(e['section_bus'])}_{int(e['step'])}"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"
    )
    os.replace(temporary, destination)


def _atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, destination)


def _key_payload(key: tuple) -> list:
    grid_id, buses, load_scale = key
    return [str(grid_id), [int(bus) for bus in buses], float(load_scale)]


def _payload_key(payload: list) -> tuple:
    return (
        str(payload[0]),
        tuple(int(bus) for bus in payload[1]),
        float(payload[2]),
    )


def _key_token(key: tuple) -> str:
    encoded = json.dumps(
        _key_payload(key), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(root: str | Path, stage: str, key: tuple) -> Path:
    return Path(root) / stage / f"{_key_token(key)}.json"


def _write_key_checkpoint(
    root: str | Path,
    stage: str,
    key: tuple,
    payload: dict,
) -> None:
    _atomic_json(
        _checkpoint_path(root, stage, key),
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "stage": stage,
            "key": _key_payload(key),
            **payload,
        },
    )


def _read_key_checkpoint(
    root: str | Path,
    stage: str,
    key: tuple,
) -> dict | None:
    path = _checkpoint_path(root, stage, key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint schema mismatch: {path}")
    if payload.get("stage") != stage:
        raise ValueError(f"checkpoint stage mismatch: {path}")
    if _payload_key(payload.get("key", [])) != key:
        raise ValueError(f"checkpoint key mismatch: {path}")
    return payload


def _prepare_checkpoint_root(root: str | Path, contract: dict) -> None:
    checkpoint_root = Path(root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_root / "manifest.json"
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "contract": contract,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text())
        if actual != expected:
            raise ValueError(
                "oracle checkpoint contract mismatch; use a new checkpoint "
                f"directory instead of mixing generations: {checkpoint_root}"
            )
        return
    _atomic_json(manifest_path, expected)


def multifault_bus_sets(buses) -> list[tuple[int, ...]]:
    """Deterministic within-storm variants; block independence remains at the storm level."""
    first = tuple(sorted({int(bus) for bus in buses})[:3])
    if len(first) < 2:
        return []
    candidates = [first, *combinations(first, 2)]
    return list(dict.fromkeys(tuple(sorted(item)) for item in candidates))


def build_multifault_episode_specs(
    events: list[dict], *, max_multifault_per_grid: int
) -> list[dict]:
    """Construct variants only inside one declared weather or Hawkes block."""
    allowed = {"weather_episode", "hawkes_family"}
    by_block: dict[tuple[str, str], list[dict]] = {}
    for event in events:
        if event.get("dependence_block_type") not in allowed:
            continue
        key = (str(event["grid_id"]), str(event["dependence_block_id"]))
        by_block.setdefault(key, []).append(event)

    multi_episodes: list[dict] = []
    per_grid: dict[str, int] = {}
    for (grid_id, block_id), block_events in sorted(by_block.items()):
        if len(block_events) < 2 or per_grid.get(grid_id, 0) >= max_multifault_per_grid:
            continue
        ordered = sorted(
            block_events, key=lambda row: (int(row["step"]), str(row["event_id"]))
        )
        bus_sets = multifault_bus_sets(int(row["section_bus"]) for row in ordered)
        for variant, bus_set in enumerate(bus_sets):
            if per_grid.get(grid_id, 0) >= max_multifault_per_grid:
                break
            contributors = [
                row for row in ordered if int(row["section_bus"]) in bus_set
            ]
            if not contributors:
                continue
            head = dict(contributors[0])
            head["_multi_buses"] = list(bus_set)
            head["_multi_ids"] = [str(row["event_id"]) for row in contributors]
            head["_multi_variant"] = variant
            head["_multi_dependence_block_id"] = block_id
            multi_episodes.append(head)
            per_grid[grid_id] = per_grid.get(grid_id, 0) + 1
    return multi_episodes


# --------------------------------------------------------------------------- #
# difficulty ladder (M8) — MEASURED, never asserted
# --------------------------------------------------------------------------- #
D2_MIN_OBS_GAP = 0.02      # partial observability must cost >= 2 % of the noop-oracle span
D1_MIN_GREEDY_GAP = 1e-9   # single-tie greedy must be strictly worse than the multi-step optimum


def assign_difficulty(*, n_concurrent_faults: int, greedy_suboptimality: float,
                      backfeed_conflict: bool, belief_entropy: float,
                      priority_divergence: bool, regret_obs_gap: float = 0.0,
                      critical_load_present: bool = False) -> str:
    """D0..D4 from MEASURED physics + MEASURED belief, per DESIGN_v2 §4.

    Two operationalisation choices, both declared and both reported alongside the counts:

    * **Ambiguity is measured by its COST, not by raw entropy.** At every degraded tier the
      posterior has non-zero entropy on essentially every episode, so ``belief_entropy > 0`` does
      not discriminate — it would put the whole slate in D2 and leave D1 empty. The rung therefore
      keys on ``regret_obs_gap``: partial observability must cost at least ``D2_MIN_OBS_GAP`` of
      the no-op-to-oracle span. That is exactly the quantity the M11 gate tests on D2+.
    * **D4's "critical-load priority" is measured as critical load being PRESENT in the restorable
      set** (weighted controllable > unweighted controllable), not as the rarer event that the
      weighted and unweighted optima happen to disagree. ``priority_divergence`` is still recorded
      per episode as its own feature.
    """
    multi = n_concurrent_faults >= 2
    ambiguous = float(regret_obs_gap) > D2_MIN_OBS_GAP
    constrained = bool(backfeed_conflict) or float(greedy_suboptimality) > D1_MIN_GREEDY_GAP
    if multi and ambiguous and (critical_load_present or priority_divergence):
        return "D4"
    if multi:
        return "D3"
    if ambiguous:
        return "D2"
    if constrained:
        return "D1"
    return "D0"


# --------------------------------------------------------------------------- #
# per-key solve (module level so it can be sent to a worker process)
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(grids_dir: str, cust_by: dict, weights_by: dict, sw_by: dict,
                 horizon_min: float, tiers: list, seed: int,
                 max_nodes: int, max_opens: int) -> None:
    _WORKER.update(grids_dir=grids_dir, cust_by=cust_by, weights_by=weights_by, sw_by=sw_by,
                   horizon_min=horizon_min, tiers=tiers, seed=seed, max_nodes=max_nodes,
                   max_opens=max_opens, nets={}, ceiling_plan_cache={})


def _net(gid: str):
    nets = _WORKER["nets"]
    if gid not in nets:
        nets[gid] = pp.from_json(os.path.join(_WORKER["grids_dir"], f"{gid}.json"))
    return nets[gid]


def solve_key(key: tuple) -> tuple:
    """key = (grid_id, (faulted buses...), load_scale) -> the full structural + per-tier label."""
    gid, buses, load_scale = key
    cust = _WORKER["cust_by"].get(gid, {})
    weights = _WORKER["weights_by"].get(gid, {})
    sw_meta = _WORKER["sw_by"].get(gid, {})
    hz = _WORKER["horizon_min"]
    net = _net(gid)
    started = time.monotonic()
    r = solve_reconfiguration(net, list(buses), cust, sw_meta, weights=weights,
                              load_scale=load_scale, horizon_min=hz,
                              max_nodes=_WORKER["max_nodes"], max_opens=_WORKER["max_opens"])
    runtime_s = time.monotonic() - started
    record = _record_from_result(key, r)
    record["search_runtime_s"] = float(runtime_s)
    return key, record


def _record_from_result(key: tuple, r) -> dict:
    """Build the complete structural and observation-tier label for one solve."""
    gid, buses, load_scale = key
    cust = _WORKER["cust_by"].get(gid, {})
    weights = _WORKER["weights_by"].get(gid, {})
    sw_meta = _WORKER["sw_by"].get(gid, {})
    hz = _WORKER["horizon_min"]
    net = _net(gid)
    rec = {
        "oracle_switch_sequence": r.switch_sequence,
        "oracle_cost_kwh": round(r.cost_kwh, 6),
        "cost_oracle_kwh": round(r.cost_kwh, 6),
        "cost_noop_kwh": round(r.cost_noop_kwh, 6),
        "controllable_customers": r.controllable_customers,
        "controllable_weighted": round(r.controllable_weighted, 4),
        "oracle_restored_customers": r.restored_customers,
        "oracle_n_switch_ops": r.n_switch_ops,
        "oracle_n_tie_closures": r.n_tie_closures,
        "oracle_n_sectionalizing_opens": r.n_sectionalizing_opens,
        "down_customers": r.down_customers,
        "greedy_switch_sequence": r.greedy_switch_sequence,
        "greedy_cost_kwh": round(r.greedy_cost_kwh, 6),
        "greedy_restored_customers": r.greedy_restored_customers,
        "greedy_suboptimality": round(r.greedy_suboptimality, 6),
        "backfeed_conflict": bool(r.backfeed_conflict),
        "priority_divergence": bool(r.priority_divergence),
        "initial_pf_feasible": bool(r.initial_pf_feasible),
        "initial_pf_metrics": r.initial_pf_metrics,
        "proven_optimal": bool(r.proven_optimal),
        "oracle_method": r.method,
        "n_leaves": r.n_leaves,
        "n_pf_calls": r.n_pf_calls,
        "search_notes": r.notes,
        "search_max_nodes": _WORKER["max_nodes"],
        "search_max_opens": _WORKER["max_opens"],
        "load_scale": round(float(load_scale), 5),
        "n_concurrent_faults": len(buses),
    }
    by_tier = {}
    # Single-fault hypothesis plans repeat across many event keys. Reuse them
    # within a worker for the same feeder and operating point. Multifault rows
    # retain a private cache because their declared single-fault belief model
    # uses the multifault oracle only for its physical lower bound.
    if len(buses) == 1:
        plans_cache = _WORKER["ceiling_plan_cache"].setdefault(gid, {})
    else:
        plans_cache = {}
    for tier in _WORKER["tiers"]:
        c = obs_limited_ceiling(net, list(buses), cust, sw_meta, tier, weights=weights,
                                load_scale=load_scale, horizon_min=hz, seed=_WORKER["seed"],
                                oracle_result=r, plans_cache=plans_cache,
                                max_opens=_WORKER["max_opens"])
        gap = _regret(c["cost_obs_ceiling_kwh"], r.cost_kwh, r.cost_noop_kwh)
        c["regret_obs_gap"] = round(gap, 6)
        c["difficulty_tier"] = assign_difficulty(
            n_concurrent_faults=len(buses), greedy_suboptimality=r.greedy_suboptimality,
            backfeed_conflict=r.backfeed_conflict, belief_entropy=c["belief_entropy"],
            priority_divergence=r.priority_divergence, regret_obs_gap=gap,
            critical_load_present=r.controllable_weighted > r.controllable_customers + 1e-9)
        by_tier[tier] = c
    rec["by_tier"] = by_tier
    return rec


def _search_trace_entry(r, runtime_s: float) -> dict:
    return {
        "max_nodes": int(_WORKER["max_nodes"]),
        "proven_optimal": bool(r.proven_optimal),
        "cost_kwh": round(float(r.cost_kwh), 6),
        "controllable_customers": int(r.controllable_customers),
        "controllable_weighted": round(float(r.controllable_weighted), 4),
        "method": str(r.method),
        "n_leaves": int(r.n_leaves),
        "n_pf_calls": int(r.n_pf_calls),
        "runtime_s": float(runtime_s),
        "notes": list(r.notes),
    }


def solve_retry_key(task: tuple) -> tuple:
    """Retry one unresolved key and defer ceiling work until its final stage."""
    key, emit_unproven_full = task
    gid, buses, load_scale = key
    cust = _WORKER["cust_by"].get(gid, {})
    weights = _WORKER["weights_by"].get(gid, {})
    sw_meta = _WORKER["sw_by"].get(gid, {})
    started = time.monotonic()
    r = solve_reconfiguration(
        _net(gid),
        list(buses),
        cust,
        sw_meta,
        weights=weights,
        load_scale=load_scale,
        horizon_min=_WORKER["horizon_min"],
        max_nodes=_WORKER["max_nodes"],
        max_opens=_WORKER["max_opens"],
    )
    runtime_s = time.monotonic() - started
    record = (
        _record_from_result(key, r)
        if r.proven_optimal or bool(emit_unproven_full)
        else None
    )
    if record is not None:
        record["search_runtime_s"] = float(runtime_s)
    return key, _search_trace_entry(r, runtime_s), record


def crosscheck_key(key: tuple) -> tuple:
    """Independent brute-force optimum for the CIGRE-scale cross-check (M5)."""
    gid, buses, load_scale = key
    cust = _WORKER["cust_by"].get(gid, {})
    weights = _WORKER["weights_by"].get(gid, {})
    sw_meta = _WORKER["sw_by"].get(gid, {})
    net = _net(gid)
    bf = brute_force_optimum(net, list(buses), cust, sw_meta, weights=weights,
                             load_scale=load_scale, horizon_min=_WORKER["horizon_min"])
    if "skipped" in bf:
        return key, bf
    r = solve_reconfiguration(net, list(buses), cust, sw_meta, weights=weights,
                              load_scale=load_scale, horizon_min=_WORKER["horizon_min"],
                              max_nodes=_WORKER["max_nodes"], max_opens=_WORKER["max_opens"])
    bf["dfs_cost_kwh"] = round(r.cost_kwh, 6)
    bf["agree"] = bool(abs(bf["cost_kwh"] - r.cost_kwh) <= 1e-6 * max(1.0, abs(r.cost_kwh)))
    return key, bf


def _run_search_stage(
    *,
    keys: list[tuple],
    checkpoint_root: str,
    stage: str,
    grids_dir: str,
    cust_by: dict,
    weights_by: dict,
    sw_by: dict,
    horizon_min: float,
    tiers: list,
    seed: int,
    max_nodes: int,
    max_opens: int,
    n_jobs: int,
    emit_unproven_full: bool,
) -> dict[tuple, dict]:
    results: dict[tuple, dict] = {}
    pending = []
    for key in keys:
        cached = _read_key_checkpoint(checkpoint_root, stage, key)
        if cached is None:
            pending.append(key)
        else:
            results[key] = {
                "trace": cached["trace"],
                "record": cached.get("record"),
            }
    print(
        f"[oracle] stage={stage} cached={len(results)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return results

    with ProcessPoolExecutor(
        max_workers=min(max(1, int(n_jobs)), len(pending)),
        initializer=_init_worker,
        initargs=(
            grids_dir,
            cust_by,
            weights_by,
            sw_by,
            horizon_min,
            tiers,
            seed,
            max_nodes,
            max_opens,
        ),
    ) as executor:
        future_to_key = {
            executor.submit(
                solve_retry_key, (key, bool(emit_unproven_full))
            ): key
            for key in pending
        }
        errors = []
        for completed, future in enumerate(as_completed(future_to_key), 1):
            expected_key = future_to_key[future]
            try:
                key, trace, record = future.result()
            except Exception as error:
                errors.append((expected_key, repr(error)))
                print(
                    f"[oracle] stage={stage} failed key={expected_key!r}: "
                    f"{error!r}",
                    flush=True,
                )
                continue
            if key != expected_key:
                raise RuntimeError(
                    f"worker returned {key!r} for {expected_key!r}"
                )
            payload = {"trace": trace, "record": record}
            _write_key_checkpoint(
                checkpoint_root, stage, key, payload
            )
            results[key] = payload
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(
                    f"[oracle] stage={stage} completed={completed}/"
                    f"{len(pending)}",
                    flush=True,
                )
        if errors:
            raise RuntimeError(
                f"oracle stage {stage} failed for {len(errors)} keys; "
                f"first={errors[0]!r}"
            )
    return results


def _run_crosscheck_stage(
    *,
    keys: list[tuple],
    checkpoint_root: str,
    grids_dir: str,
    cust_by: dict,
    weights_by: dict,
    sw_by: dict,
    horizon_min: float,
    tiers: list,
    seed: int,
    max_nodes: int,
    max_opens: int,
    n_jobs: int,
) -> dict[tuple, dict]:
    stage = f"crosscheck_{int(max_nodes)}"
    results: dict[tuple, dict] = {}
    pending = []
    for key in keys:
        cached = _read_key_checkpoint(checkpoint_root, stage, key)
        if cached is None:
            pending.append(key)
        else:
            results[key] = cached["crosscheck"]
    print(
        f"[oracle] stage={stage} cached={len(results)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return results

    with ProcessPoolExecutor(
        max_workers=min(max(1, int(n_jobs)), len(pending)),
        initializer=_init_worker,
        initargs=(
            grids_dir,
            cust_by,
            weights_by,
            sw_by,
            horizon_min,
            tiers,
            seed,
            max_nodes,
            max_opens,
        ),
    ) as executor:
        future_to_key = {
            executor.submit(crosscheck_key, key): key for key in pending
        }
        errors = []
        for completed, future in enumerate(as_completed(future_to_key), 1):
            expected_key = future_to_key[future]
            try:
                key, result = future.result()
            except Exception as error:
                errors.append((expected_key, repr(error)))
                print(
                    f"[oracle] stage={stage} failed key={expected_key!r}: "
                    f"{error!r}",
                    flush=True,
                )
                continue
            if key != expected_key:
                raise RuntimeError(
                    f"worker returned {key!r} for {expected_key!r}"
                )
            _write_key_checkpoint(
                checkpoint_root,
                stage,
                key,
                {"crosscheck": result},
            )
            results[key] = result
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(
                    f"[oracle] stage={stage} completed={completed}/"
                    f"{len(pending)}",
                    flush=True,
                )
        if errors:
            raise RuntimeError(
                f"oracle stage {stage} failed for {len(errors)} keys; "
                f"first={errors[0]!r}"
            )
    return results


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_oracle_labels(grids_dir: str, events_path: str, obs_model_path: str,
                        out_path: str, seed: int, *, tiers: list | None = None,
                        horizon_min: float = DEFAULT_HORIZON_MIN, n_jobs: int = 16,
                        max_multifault_per_grid: int = MAX_MULTIFAULT_PER_GRID,
                        initial_max_nodes: int = DEFAULT_MAX_NODES,
                        retry_max_nodes: int = RETRY_MAX_NODES,
                        convergence_max_nodes: list[int] | tuple[int, ...] = (),
                        max_opens: int = FINAL_MAX_OPENS,
                        series_dir: str | None = None,
                        max_events: int | None = None,
                        checkpoint_dir: str | None = None) -> dict:
    """Attach v2 oracle + belief-ceiling + difficulty labels to every interrupting event."""
    tiers = list(tiers or TIERS)
    if series_dir is None:
        series_dir = os.path.join(os.path.dirname(os.path.abspath(grids_dir)), "series")

    bus_alloc = pd.read_parquet(os.path.join(series_dir, "bus_allocation.parquet"))
    cust_by = {g: {int(b): int(c) for b, c in zip(v["bus"], v["customers"])}
               for g, v in bus_alloc.groupby("grid_id")}
    weights_by = {g: {int(b): float(CLASS_WEIGHTS.get(str(k), 1.0))
                      for b, k in zip(v["bus"], v["customer_class"])}
                  for g, v in bus_alloc.groupby("grid_id")}
    sm = pd.read_csv(os.path.join(grids_dir, "switch_manifest.csv"))
    sw_by = {g: {int(r.switch_id): {"remote": bool(r.remote),
                                    "travel_time_min": float(r.travel_time_min),
                                    "type": str(r.type)} for r in v.itertuples()}
             for g, v in sm.groupby("grid_id")}

    all_events = [json.loads(l) for l in open(events_path) if l.strip()]
    v4_dependence = bool(all_events) and all(
        e.get("event_id")
        and e.get("dependence_block_id")
        and e.get("dependence_block_type")
        and e.get("dependence_definition_sha256")
        for e in all_events
    )
    if v4_dependence:
        dependence_errors = validate_annotated_events(all_events)
        if dependence_errors:
            raise ValueError(
                "event dependence validation failed: "
                + "; ".join(dependence_errors[:10])
            )
    events = [e for e in all_events if e.get("interrupts")]
    events.sort(key=lambda event: (str(event["grid_id"]), str(event.get("event_id", ""))))
    if max_events is not None:
        if int(max_events) <= 0:
            raise ValueError("max_events must be positive")
        events = events[: int(max_events)]
    grid_ids = sorted({e["grid_id"] for e in events})

    # ---- operating point per grid + the real diurnal load trajectory ----
    op_path = os.path.join(os.path.dirname(out_path), "operating_point.json")
    if os.path.exists(op_path):
        op = json.load(open(op_path))
        base_scale, profile, bin_of_hour, bin_factor = (
            op["base_scale"], op["hourly_profile"], op["bin_of_hour"], op["bin_factor"])
    else:
        base_scale = {g: calibrate_operating_point(pp.from_json(os.path.join(grids_dir, f"{g}.json")))
                      for g in grid_ids}
        profile = hourly_load_profile(series_dir)
        bin_of_hour, bin_factor = load_bin_factors(profile)
        _atomic_json(
            op_path,
            {
                "base_scale": base_scale,
                "hourly_profile": profile,
                "bin_of_hour": bin_of_hour,
                "bin_factor": bin_factor,
                "target_prefault_loading_pct": TARGET_PREFAULT_LOADING_PCT,
            },
        )

    def _hour(e) -> int:
        try:
            return int(pd.Timestamp(e["timestamp"]).hour)
        except Exception:
            return int((int(e["step"]) // 4) % 24)

    def _scale(e) -> float:
        return round(base_scale[e["grid_id"]] * bin_factor[bin_of_hour[_hour(e)]], 5)

    # Legacy generation remains readable for v3 reproduction, but every v4
    # multifault variant is constrained to one declared weather/Hawkes block.
    if v4_dependence:
        clusters = None
        multi_episodes = build_multifault_episode_specs(
            events, max_multifault_per_grid=max_multifault_per_grid
        )
    else:
        clusters = assign_storm_clusters(events)
        by_cluster: dict = {}
        for e in events:
            by_cluster.setdefault((e["grid_id"], clusters[_event_key(e)]), []).append(e)
        multi_episodes = []
        per_grid: dict = {}
        for (gid, cid), evs in sorted(by_cluster.items()):
            if len(evs) < 2 or per_grid.get(gid, 0) >= max_multifault_per_grid:
                continue
            evs = sorted(evs, key=lambda x: int(x["step"]))
            bus_sets = multifault_bus_sets(int(x["section_bus"]) for x in evs)
            if not bus_sets:
                continue
            for variant, bus_set in enumerate(bus_sets):
                if per_grid.get(gid, 0) >= max_multifault_per_grid:
                    break
                head = dict(evs[0])
                head["_multi_buses"] = list(bus_set)
                head["_multi_ids"] = [
                    _event_key(x) for x in evs if int(x["section_bus"]) in bus_set
                ]
                head["_multi_variant"] = variant
                multi_episodes.append(head)
                per_grid[gid] = per_grid.get(gid, 0) + 1

    # ---- unique solve keys ----
    keys = set()
    for e in events:
        keys.add((e["grid_id"], (int(e["section_bus"]),), _scale(e)))
    for e in multi_episodes:
        keys.add((e["grid_id"], tuple(e["_multi_buses"]), _scale(e)))
    keys = sorted(keys)

    budgets = sorted({
        int(value)
        for value in convergence_max_nodes
        if int(value) > max(int(retry_max_nodes), int(initial_max_nodes))
    })
    stage_budgets = [int(initial_max_nodes)]
    if int(retry_max_nodes) > int(initial_max_nodes):
        stage_budgets.append(int(retry_max_nodes))
    stage_budgets.extend(
        budget for budget in budgets if budget > stage_budgets[-1]
    )
    checkpoint_root = str(
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else Path(f"{out_path}.checkpoints")
    )
    grid_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(Path(grids_dir).glob("*.json"))
    }
    source_paths = {
        "restoration_oracle.py": Path(__file__),
        "reconfig_search.py": Path(
            solve_reconfiguration.__code__.co_filename
        ),
        "obs_ceiling.py": Path(obs_limited_ceiling.__code__.co_filename),
        "pf_feasibility.py": Path(pf_feasible.__code__.co_filename),
        "cost_model.py": Path(_regret.__code__.co_filename),
    }
    checkpoint_contract = {
        "events_sha256": _sha256_file(events_path),
        "obs_model_sha256": _sha256_file(obs_model_path),
        "switch_manifest_sha256": _sha256_file(
            Path(grids_dir) / "switch_manifest.csv"
        ),
        "bus_allocation_sha256": _sha256_file(
            Path(series_dir) / "bus_allocation.parquet"
        ),
        "real_feeders_sha256": _sha256_file(
            Path(series_dir) / "real_feeders.parquet"
        ),
        "operating_point_sha256": _sha256_file(op_path),
        "grid_json_sha256": grid_hashes,
        "code_sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
        "keys_sha256": hashlib.sha256(
            json.dumps(
                [_key_payload(key) for key in keys],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "n_keys": len(keys),
        "seed": int(seed),
        "tiers": tiers,
        "horizon_min": float(horizon_min),
        "max_multifault_per_grid": int(max_multifault_per_grid),
        "max_events": (
            None if max_events is None else int(max_events)
        ),
        "max_opens": int(max_opens),
        "stage_budgets": stage_budgets,
    }
    _prepare_checkpoint_root(checkpoint_root, checkpoint_contract)

    solved: dict = {}
    search_traces = {key: [] for key in keys}
    stage_remaining = []
    retry_keys: list[tuple] = []
    for stage_index, max_nodes in enumerate(stage_budgets):
        if stage_index == 0:
            targets = keys
        else:
            targets = [
                key
                for key in keys
                if not search_traces[key][-1]["proven_optimal"]
            ]
        if not targets:
            break
        stage = f"search_{int(max_nodes)}"
        results = _run_search_stage(
            keys=targets,
            checkpoint_root=checkpoint_root,
            stage=stage,
            grids_dir=grids_dir,
            cust_by=cust_by,
            weights_by=weights_by,
            sw_by=sw_by,
            horizon_min=horizon_min,
            tiers=tiers,
            seed=seed,
            max_nodes=int(max_nodes),
            max_opens=int(max_opens),
            n_jobs=int(n_jobs),
            emit_unproven_full=stage_index == len(stage_budgets) - 1,
        )
        for key in targets:
            result = results[key]
            search_traces[key].append(result["trace"])
            if result.get("record") is not None:
                solved[key] = result["record"]
        unresolved_count = sum(
            not search_traces[key][-1]["proven_optimal"] for key in keys
        )
        if stage_index == 0:
            retry_keys = [
                key
                for key in keys
                if not search_traces[key][-1]["proven_optimal"]
            ]
        else:
            stage_remaining.append({
                "max_nodes": int(max_nodes),
                "n_unproven": unresolved_count,
            })

    missing_records = [key for key in keys if key not in solved]
    if missing_records:
        raise RuntimeError(
            "final oracle stage did not emit full records for "
            f"{len(missing_records)} keys"
        )

    for key in keys:
        solved[key]["oracle_search_trace"] = search_traces[key]
        solved[key]["oracle_proof_max_nodes"] = int(
            search_traces[key][-1]["max_nodes"]
        )

    # ---- independent brute-force cross-check on the CIGRE-scale subset ----
    cc_keys = [k for k in keys if k[0] == "cigre_mv"]
    crosscheck = {"n_attempted": len(cc_keys), "n_evaluated": 0, "n_agree": 0,
                  "agreement_fraction": None, "disagreements": []}
    if cc_keys:
        crosscheck_max_nodes = max(stage_budgets)
        crosscheck_results = _run_crosscheck_stage(
            keys=cc_keys,
            checkpoint_root=checkpoint_root,
            grids_dir=grids_dir,
            cust_by=cust_by,
            weights_by=weights_by,
            sw_by=sw_by,
            horizon_min=horizon_min,
            tiers=tiers,
            seed=seed,
            max_nodes=crosscheck_max_nodes,
            max_opens=max_opens,
            n_jobs=n_jobs,
        )
        for key in cc_keys:
            bf = crosscheck_results[key]
            if "skipped" in bf:
                continue
            crosscheck["n_evaluated"] += 1
            crosscheck["n_agree"] += int(bf["agree"])
            if not bf["agree"]:
                crosscheck["disagreements"].append(
                    {
                        "key": [key[0], list(key[1]), key[2]],
                        "dfs": bf["dfs_cost_kwh"],
                        "brute_force": bf["cost_kwh"],
                    }
                )
        if crosscheck["n_evaluated"]:
            crosscheck["agreement_fraction"] = round(
                crosscheck["n_agree"] / crosscheck["n_evaluated"], 6
            )

    # ---- emit ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written = 0
    tier_counts: dict = {}
    output_lines = []
    for e in events + multi_episodes:
        multi = "_multi_buses" in e
        buses = tuple(e["_multi_buses"]) if multi else (int(e["section_bus"]),)
        key = (e["grid_id"], buses, _scale(e))
        base = solved.get(key)
        if base is None:
            continue
        multi_suffix = ""
        if multi:
            variant = int(e.get("_multi_variant", 0))
            multi_suffix = "_multi" if variant == 0 else f"_multi_v{variant}"
        event_identity = str(e["event_id"]) if v4_dependence else _event_key(e)
        cid = f"{_event_key(e)}_{event_identity[-8:]}{multi_suffix}"
        rec = {
            "case_id": cid,
            "grid_id": e["grid_id"],
            "section_bus": int(e["section_bus"]),
            "faulted_buses": list(buses),
            "cause": str(e["cause"]),
            "weather_feeder": e.get("weather_feeder"),
            "timestamp": str(e["timestamp"]),
            "feeder_year": (
                f"{e['grid_id']}_{e['feeder']}_{pd.Timestamp(e['timestamp']).year}"
            ),
            "variant_group_id": f"{e['grid_id']}_{e['feeder']}_{e['cause']}",
            "step": int(e["step"]), "hour": _hour(e),
            "duration_min": float(e.get("duration_min", 0.0)),
            "earthing_regime": e.get("earthing_regime"),
            "is_multifault": bool(multi),
            "source_event_ids": (
                list(e.get("_multi_ids", [])) if multi else [event_identity]
            ),
            "tier": None,
            **base,
        }
        if v4_dependence:
            rec.update(
                {
                    "event_id": event_identity,
                    "parent_event_id": e.get("parent_event_id"),
                    "branch_root_id": e["branch_root_id"],
                    "event_burst_id": e["event_burst_id"],
                    "weather_episode_id": e.get("weather_episode_id"),
                    "dependence_block_id": e["dependence_block_id"],
                    "dependence_block_type": e["dependence_block_type"],
                    "dependence_definition_sha256": e[
                        "dependence_definition_sha256"
                    ],
                }
            )
            for field in (
                "weather_episode_id_gap6h",
                "weather_episode_id_gap12h",
                "weather_episode_id_gap24h",
                "weather_episode_id_gap48h",
            ):
                rec[field] = e.get(field)
        else:
            rec.update(
                {
                    "storm_cluster_id": clusters[_event_key(e)],
                    "storm_window": clusters[_event_key(e)],
                }
            )
        output_lines.append(json.dumps(rec) + "\n")
        n_written += 1
        for t in tiers:
            d = base["by_tier"][t]["difficulty_tier"]
            tier_counts.setdefault(t, {}).setdefault(d, 0)
            tier_counts[t][d] += 1
    _atomic_text(out_path, "".join(output_lines))

    scorable = [
        solved[k] for k in keys
        if solved[k]["controllable_customers"] > 0
        and solved[k]["initial_pf_feasible"]
    ]
    labels_sha256 = _sha256_file(out_path)
    meta = {
        "substrate_oracle_version": "v2",
        "n_labeled": n_written,
        "n_unique_keys": len(keys),
        "n_multifault_episodes": len(multi_episodes),
        "max_multifault_per_grid": max_multifault_per_grid,
        "max_events": max_events,
        "n_jobs": n_jobs,
        "slurm_cpus_per_task": (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK") else None
        ),
        "multifault_policy": (
            "distinct buses inside one declared weather/Hawkes dependence block"
            if v4_dependence else
            "per storm-window: first three distinct fault buses as one episode "
            "plus their distinct pairs, capped per grid"
        ),
        "n_scorable_keys": len(scorable),
        "n_initial_pf_infeasible_keys": sum(
            not solved[k]["initial_pf_feasible"] for k in keys
        ),
        "proven_optimal_fraction": round(
            float(np.mean([s["proven_optimal"] for s in solved.values()])), 6) if solved else 0.0,
        "median_oracle_switch_ops": float(np.median(
            [s["oracle_n_switch_ops"] for s in scorable])) if scorable else 0.0,
        "frac_multi_op": round(float(np.mean(
            [s["oracle_n_switch_ops"] >= 2 for s in scorable])), 6) if scorable else 0.0,
        "frac_greedy_suboptimal": round(float(np.mean(
            [s["greedy_suboptimality"] > 1e-9 for s in scorable])), 6) if scorable else 0.0,
        "median_greedy_suboptimality_when_positive": round(float(np.median(
            [s["greedy_suboptimality"] for s in scorable if s["greedy_suboptimality"] > 1e-9]
        )), 6) if any(s["greedy_suboptimality"] > 1e-9 for s in scorable) else 0.0,
        "difficulty_counts_by_obs_tier": tier_counts,
        "n_calendar_days_with_events": len({int(e["step"]) // 96 for e in events}),
        "brute_force_crosscheck": crosscheck,
        "target_prefault_loading_pct": TARGET_PREFAULT_LOADING_PCT,
        "class_weights_assumption": CLASS_WEIGHTS,
        "obs_model_hash": json.load(open(obs_model_path)).get("config_hash"),
        "labels_sha256": labels_sha256,
        "label_generation_id": labels_sha256[:16],
        "search_policy": {
            "max_depth": 4,
            "max_opens": max_opens,
            "initial_max_nodes": initial_max_nodes,
            "retry_max_nodes": retry_max_nodes,
            "convergence_max_nodes": budgets,
            "n_retried_keys": len(retry_keys),
            "n_unproven_after_retry": (
                stage_remaining[0]["n_unproven"]
                if stage_remaining
                else sum(
                    not search_traces[k][-1]["proven_optimal"]
                    for k in keys
                )
            ),
            "stage_remaining": stage_remaining,
            "n_unproven_final": sum(
                not search_traces[k][-1]["proven_optimal"]
                for k in keys
            ),
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_manifest_sha256": _sha256_file(
                Path(checkpoint_root) / "manifest.json"
            ),
        },
        "out_path": out_path, "seed": seed, "tiers": tiers,
    }
    if v4_dependence:
        definition_hashes = {
            str(event["dependence_definition_sha256"]) for event in events
        }
        if len(definition_hashes) != 1:
            raise ValueError("events use mixed dependence-definition hashes")
        meta.update(
            {
                "substrate_oracle_version": "v4",
                "dependence_definition_sha256": next(iter(definition_hashes)),
                "n_dependence_blocks": len(
                    {str(event["dependence_block_id"]) for event in events}
                ),
                "dependence_block_type_counts": dict(
                    Counter(
                        {
                            str(event["dependence_block_id"]): str(
                                event["dependence_block_type"]
                            )
                            for event in events
                        }.values()
                    )
                ),
                "event_counts_by_dependence_block_type": dict(
                    Counter(
                        str(event["dependence_block_type"]) for event in events
                    )
                ),
            }
        )
    else:
        meta["n_storm_clusters"] = len(set(clusters.values()))
    _atomic_json(
        os.path.join(os.path.dirname(out_path), "oracle_meta.json"),
        meta,
    )
    return meta
