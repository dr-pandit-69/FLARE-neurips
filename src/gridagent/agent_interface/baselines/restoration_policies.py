"""GridAgent-Bench v2 — FLISR incumbents + floors, scored by REAL rollout (GOAL_v2 M1/M2/M10).

Every arm is driven through the REAL ``RestorationEpisode`` (guarded switching + AC power flow)
and scored by the SINGLE shared ``cost_model.restoration_cost``. No arm's cost is a closed-form
function of the oracle anchors, and no arm carries its own switching-time model — that was v1
defect #1, the reason the LLM could "beat" the optimum.

Arms
----
  noop            close nothing -> the regret-1 anchor
  oracle          replay the label's PF-validated min-ENS plan -> regret exactly 0
  obs_ceiling     replay the label's observation-limited belief-optimal plan (M6)
  greedy          the single highest-gain feasible tie (myopic, full information)
  flisr_ideal     full-information rule-based FLISR: keep closing the best feasible tie
  flisr_operator  the FAIR incumbent (M10): same fixed rule, but driven by the SAME degraded
                  observations the agent sees, so it mis-localizes under conflicting evidence
  random          a random feasible subset of ties (triviality floor)
  belief_planner  adaptive observation-matched receding-horizon planner
  belief_planner_no_sim  identical planner with simulate_switch disabled
  unsafe_probe    TEST ONLY: commits a guard-bypassing close so the safety metric has a positive

Safety (M2). A guard rejection is NOT a hard-fail — it is the protection system doing its job, and
is reported as ``blocked_*``. ``flisr_hardfail`` is measured from the FINAL COMMITTED state by
``RestorationEpisode.final_state_safety`` (thermal, EN 50160 voltage, radiality, energized fault).
``regret_hard`` disqualifies a hard-failing arm at 1.0 and is reported ALONGSIDE the outcome-only
``regret_omni``, never blended into it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from gridagent.agent_interface.cost_model import (
    DEFAULT_HORIZON_MIN, KW_PER_CUSTOMER, REMOTE_MIN, raw_regret, regret as _regret,
    restoration_cost, switch_time_min,
)
from gridagent.agent_interface.restoration_env import (
    ControllerActionViolation, EnergizationViolation, Fault,
    LiveSectionalizerOpenViolation, MAX_RESTORATION_MINUTES,
    PowerFlowViolation, RestorationEpisode,
)
from gridagent.simulation.grid_bank import RadialityViolation, unsupplied

HORIZON = MAX_RESTORATION_MINUTES        # 240 min
ARM_REGISTRY = [
    "noop", "random", "greedy", "flisr_ideal", "flisr_operator",
    "belief_planner", "belief_planner_no_sim", "obs_ceiling", "oracle",
]
GUARD_ERRORS = (
    RadialityViolation,
    EnergizationViolation,
    PowerFlowViolation,
    LiveSectionalizerOpenViolation,
    ControllerActionViolation,
)


def _ens(customers: float, minutes: float) -> float:
    """Controllable ENS proxy (kWh) for `customers` out for `minutes` (legacy helper)."""
    return max(customers, 0.0) * minutes / 60.0 * KW_PER_CUSTOMER


def _customers_of(buses, cust: dict) -> int:
    return int(sum(int(cust.get(int(b), 0)) for b in buses))


def _weighted_of(buses, cust: dict, weights: dict | None) -> float:
    if not weights:
        return float(_customers_of(buses, cust))
    return float(sum(int(cust.get(int(b), 0)) * float(weights.get(int(b), 1.0)) for b in buses))


def _open_ties(net, tie_ids=None) -> list:
    opn = [int(s) for s in net.switch.index[~net.switch.closed]]
    if tie_ids:
        opn = [s for s in opn if s in tie_ids]
    return sorted(opn)


class _Result(dict):
    pass


# --------------------------------------------------------------------------- #
# THE shared scorer — used by the baselines AND by the LLM agent
# --------------------------------------------------------------------------- #
def score_episode(ep: RestorationEpisode, label: dict, cust: dict, *,
                  weights: dict | None = None, horizon_min: float = DEFAULT_HORIZON_MIN) -> dict:
    """Physics-derived cost / regret of whatever the episode actually ended up in.

    The controllable set and the anchors come from the oracle LABEL (built by the same
    ``reconfig_search`` the env starts from), so the oracle arm replaying its own plan reproduces
    ``cost_oracle_kwh`` exactly and scores regret 0.
    """
    final = ep.energized_buses()
    initial_dead = set(ep.out0) - set(ep.fault.faulted_buses)
    initial_live = set(ep.energized0)

    last_gain, last_loss = {}, {}
    for buses, idx in ep.restore_events:
        for b in buses:
            last_gain[int(b)] = idx
    for buses, idx in ep.shed_events:
        for b in buses:
            last_loss[int(b)] = idx

    restored_by_op: dict = defaultdict(float)
    restored_buses = []
    for b in sorted(initial_dead & final):
        restored_by_op[last_gain.get(b, len(ep.committed_ops) - 1)] += _weighted_of([b], cust, weights)
        restored_buses.append(int(b))
    shed_by_op: dict = defaultdict(float)
    shed_buses = []
    for b in sorted(initial_live - final):
        shed_by_op[last_loss.get(b, 0)] += _weighted_of([b], cust, weights)
        shed_buses.append(int(b))

    restored_w = float(sum(restored_by_op.values()))
    controllable_w = float(label.get("controllable_weighted",
                                     label.get("controllable_customers", restored_w)))
    cost = restoration_cost(
        [(n, k) for k, n in sorted(restored_by_op.items())],
        max(controllable_w - restored_w, 0.0),
        ep.committed_ops, ep.sw_meta, horizon_min,
        shed_events=[(n, k) for k, n in sorted(shed_by_op.items())],
        extra_delay_min=ep.sensing_at_op or ep.sensing_minutes,
    )
    cost_noop = float(label.get("cost_noop_kwh",
                                restoration_cost(0, controllable_w, [], ep.sw_meta, horizon_min)))
    cost_oracle = float(label.get("cost_oracle_kwh", 0.0))

    safety = ep.final_state_safety()
    r_omni = _regret(cost, cost_oracle, cost_noop)
    rec = {
        "provenance": "rollout",
        "ens_ctrl_kwh": float(cost),
        "cost_noop_ctrl": float(cost_noop),
        "cost_oracle_ctrl": float(cost_oracle),
        "regret_omni": r_omni,
        "regret_raw": float(raw_regret(cost, cost_oracle, cost_noop)),
        "regret_hard": 1.0 if safety["flisr_hardfail"] else r_omni,
        "committed_actions": int(len(ep.committed_ops)),
        "n_tie_closures": int(sum(1 for o in ep.committed_ops if o["closed"])),
        "n_sectionalizing_opens": int(sum(1 for o in ep.committed_ops if not o["closed"])),
        "restored_customers": int(_customers_of(restored_buses, cust)),
        "restored_weighted": restored_w,
        "restored_buses": restored_buses,
        "shed_customers": int(_customers_of(shed_buses, cust)),
        "restorable_customers": int(label.get("controllable_customers", 0)),
        "restore_clock_min": float(round(ep.op_times()[-1] if ep.op_times() else 0.0, 2)),
        "sensing_minutes": float(round(ep.sensing_minutes, 2)),
        "n_test_switch": int(ep.n_test_switch),
        "max_committed_ops": int(ep.max_committed_ops),
        "switch_ops_remaining": int(ep.switch_ops_remaining),
        "max_sectionalizer_opens": int(ep.max_sectionalizer_opens),
        "sectionalizer_opens_remaining": int(
            ep.sectionalizer_opens_remaining
        ),
    }
    rec.update(safety)
    # M6 decomposition anchors carried through from the label so every record is self-describing
    for k in ("cost_obs_ceiling_kwh", "regret_obs_gap", "obs_ceiling_proven_bound",
              "difficulty_tier", "belief_entropy", "greedy_suboptimality", "oracle_n_switch_ops",
              "n_concurrent_faults", "backfeed_conflict", "storm_cluster_id",
              "initial_pf_feasible", "proven_optimal"):
        if k in label:
            rec[k] = label[k]
    if "regret_obs_gap" in label and label["regret_obs_gap"] is not None:
        rec["regret_reasoning_gap"] = float(r_omni) - float(label["regret_obs_gap"])
    return rec


# --------------------------------------------------------------------------- #
# arm policies
# --------------------------------------------------------------------------- #
def _try(ep: RestorationEpisode, sid: int, closed: bool = True) -> bool:
    try:
        result = ep.operate(int(sid), closed)
        return bool(result.get("applied", False))
    except GUARD_ERRORS:
        return False


def _replay(ep: RestorationEpisode, plan) -> None:
    for op in plan or []:
        if isinstance(op, dict):
            _try(ep, int(op["switch_id"]), bool(op.get("closed", True)))
        else:
            _try(ep, int(op), True)


def _best_feasible_tie(ep: RestorationEpisode, ties, cust: dict, weights=None):
    """The tie whose close yields the largest immediate weighted gain per minute spent.
    Full-information: it reads the true energized set (an incumbent CEILING, not the fair one)."""
    best, best_score, best_gain = None, 0.0, 0
    before = ep.energized_buses()
    for sid in ties:
        if not ep.net.switch.at[sid, "closed"]:
            probe = _probe_close(ep, sid)
            if probe is None:
                continue
            gained = probe - before
            w = _weighted_of(gained, cust, weights)
            if w <= 0:
                continue
            score = w / max(switch_time_min(sid, ep.sw_meta), 1e-9)
            if score > best_score:
                best, best_score, best_gain = int(sid), score, _customers_of(gained, cust)
    return best, best_gain


def _probe_close(ep: RestorationEpisode, sid: int):
    """Guard-check a close WITHOUT committing; returns the resulting energized set or None."""
    import copy as _copy
    from gridagent.simulation.grid_bank import energized_buses, n_independent_loops
    from gridagent.simulation.pf_feasibility import pf_feasible
    trial = _copy.deepcopy(ep.net)
    trial.switch.at[int(sid), "closed"] = True
    if n_independent_loops(trial) != 0:
        return None
    en = energized_buses(trial)
    if set(ep.fault.faulted_buses) & en:
        return None
    if ep.enforce_pf and not pf_feasible(trial)[0]:
        return None
    return en


def rollout_arm(arm: str, net, fault: Fault, label: dict, cust: dict,
                rng: np.random.Generator, *, sw_meta: dict | None = None,
                weights: dict | None = None, tier: str | None = None, seed: int = 0,
                horizon_min: float = DEFAULT_HORIZON_MIN,
                belief_planner_config: dict | None = None) -> _Result:
    """Run ONE arm through the real RestorationEpisode and score it with the shared cost model."""
    sw_meta = sw_meta or {}
    tie_ids = {s for s, m in sw_meta.items() if m.get("type") == "tie"}
    unsafe = (arm == "unsafe_probe")
    ep = RestorationEpisode(net, fault, sw_meta=sw_meta, enforce_pf=not unsafe,
                            enforce_radiality=not unsafe, enforce_energization=not unsafe,
                            horizon_min=horizon_min)
    open_ties = _open_ties(ep.net, tie_ids)

    if arm == "noop":
        pass
    elif arm == "oracle":
        _replay(ep, label.get("oracle_switch_sequence", []))
    elif arm == "obs_ceiling":
        _replay(ep, label.get("obs_ceiling_switch_sequence",
                              label.get("oracle_switch_sequence", [])))
        ep.charge_sensing_time(float(label.get("obs_ceiling_sensing_min", 0.0)))
    elif arm == "greedy":
        best, _g = _best_feasible_tie(ep, open_ties, cust, weights)
        if best is not None:
            _try(ep, best, True)
    elif arm == "flisr_ideal":
        # full-information rule-based FLISR: keep closing the best feasible tie while it helps
        for _ in range(4):
            best, gain = _best_feasible_tie(ep, open_ties, cust, weights)
            if best is None or gain <= 0:
                break
            if not _try(ep, best, True):
                break
    elif arm == "flisr_operator":
        from gridagent.agent_interface.baselines.operator_policy import run_flisr_operator
        run_flisr_operator(ep, open_ties, cust, tier or "clean", seed=seed, weights=weights)
    elif arm in ("belief_planner", "belief_planner_no_sim"):
        from gridagent.agent_interface.baselines.belief_planner import (
            config_from_dict,
            run_belief_planner,
        )
        planner_trace = run_belief_planner(
            ep,
            open_ties,
            cust,
            tier or "clean",
            seed=seed,
            use_simulator=arm == "belief_planner",
            config=config_from_dict(belief_planner_config),
        )
    elif arm == "random":
        order = list(open_ties)
        rng.shuffle(order)
        for sid in order[: max(1, len(order) // 2)]:
            _try(ep, int(sid), True)
    elif arm == "unsafe_probe":
        # deliberately commit the first connectivity-valid close with every guard disabled, so the
        # safety metric has a positive control (M2 acceptance test).
        for sid in open_ties:
            if _try(ep, int(sid), True):
                break
    else:
        raise ValueError(f"unknown arm {arm}")

    rec = score_episode(ep, label, cust, weights=weights, horizon_min=horizon_min)
    if arm in ("belief_planner", "belief_planner_no_sim"):
        rec.update(planner_trace)
    rec["arm"] = arm
    rec["manual_override"] = False
    return _Result(rec)


# ---- backward-compatible shim: the old evaluate_arm signature, now a real rollout ---- #
def evaluate_arm(arm: str, episode_info: dict, rng: np.random.Generator) -> _Result:
    net = episode_info["net"]
    fault = episode_info.get("fault") or Fault(
        faulted_lines=list(episode_info.get("faulted_lines", [])),
        faulted_buses=list(episode_info.get("faulted_buses", []))
        or ([int(episode_info["faulted_bus"])] if episode_info.get("faulted_bus") is not None else []),
    )
    label = episode_info.get("label") or {
        "controllable_customers": episode_info.get("restorable_customers", 0),
        "oracle_switch_sequence": episode_info.get("switch_seq", []),
    }
    return rollout_arm(arm, net, fault, label, episode_info.get("cust", {}), rng,
                       sw_meta=episode_info.get("sw_meta"), tier=episode_info.get("tier"))
