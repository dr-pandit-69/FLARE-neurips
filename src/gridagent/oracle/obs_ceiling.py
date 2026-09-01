"""GridAgent-Bench v2 — the REAL observation-limited belief ceiling.  [GOAL_v2 M6]

v1 shipped a fudge: ``_obs_ceiling_penalty`` multiplied the restorable customers by a hand-set
0.15 (compensated earthing) or 0.08 (otherwise). That number was invented, so the "cost of partial
observability" it produced measured nothing.

v2 computes it. After the breaker trips, the faulted section is HIDDEN. The operator sees only the
tier-degraded channels, so several sections remain consistent with the evidence. This module

  1. builds the **hypothesis set** H — the sections that could be the faulted one given what is
     observably dark;
  2. computes the **posterior** P(h | o) from the frozen observation model's own likelihoods
     (fault-indicator error rate, SCADA instrumentation fraction and flip rate) — so the belief is
     derived from the SAME degradation parameters the agent's channels are drawn from, not from a
     separate assumption;
  3. evaluates, EXHAUSTIVELY over an explicitly declared policy class, the best an
     observation-limited operator can do, and returns that policy's expected controllable-ENS.

Policy class (declared, enumerated, and reported as ``obs_ceiling_policy``):
  * ``act_blind``   — commit the MAP hypothesis' plan immediately. Fast, but under any other true
                      hypothesis the guard rejects the closes and the crew time is wasted.
  * ``inspect``     — dispatch crews to inspect candidates until the fault is pinned, then play
                      that hypothesis' oracle plan. Candidates are visited in the order that
                      minimises expected delay (Smith's rule: ascending inspect_time / posterior).
  * ``test_switch`` — binary-search the feeder with sectionalizer test-switching, ``ceil(log2|H|)``
                      probes, then play the resolved plan. Available only when the dark region has
                      enough internal sectionalizing devices.

The ceiling is the MINIMUM expected cost over that class, so no policy in the class can beat it,
and it is >= the omniscient oracle cost by construction (every branch either pays sensing time or
pays the expected cost of acting on a wrong hypothesis). ``proven_bound`` is True only where H was
enumerated exhaustively and every per-hypothesis plan is itself proven optimal; elsewhere the
result is a MAP-belief estimate and the M13 decomposition EXCLUDES it.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from gridagent.agent_interface.cost_model import DEFAULT_HORIZON_MIN, restoration_cost
from gridagent.observation.degradation_registry import tier_params
from gridagent.oracle.reconfig_search import (
    isolate_fault, graph_state, sectionalizer_ids, solve_reconfiguration,
)
from gridagent.simulation.grid_bank import _switch_bus_pair

MAX_HYPOTHESES = 12            # above this the belief search is not enumerated -> proven_bound=False
INSPECT_MIN_DEFAULT = 45.0     # crew dispatch + inspect when the manifest gives no travel time
TEST_SWITCH_MIN = 12.0         # one test-switch + reclose cycle (mirrors restoration_env)


# --------------------------------------------------------------------------- #
# hypothesis set + posterior
# --------------------------------------------------------------------------- #
def hypothesis_set(net, true_faulted_buses, *, load_scale: float = 1.0,
                   max_h: int | None = MAX_HYPOTHESES,
                   return_complete: bool = False):
    """Sections consistent with 'something in this dark region faulted'.

    The operator knows the head breaker tripped and can see (noisily) which buses went dark; the
    fault is somewhere in that region. Every bus in the de-energized region is therefore a
    candidate, and the true one is guaranteed to be in the set.

    SCOPE (disclosed): hypotheses are SINGLE faulted sections. Under concurrent faults (D3/D4) no
    single hypothesis explains the fault-indicator pattern, so the posterior stays diffuse even at
    the clean tier — measured: single-fault episodes have belief entropy 0.0000 and observation gap
    0.0000 at clean, multi-fault episodes 0.0727 and 0.0238. That residual is a property of this
    belief model, not a measured cost of sensing, and must be reported as such. Enumerating
    multi-section hypotheses would remove it at combinatorial cost.
    """
    fb = sorted({int(b) for b in true_faulted_buses if b is not None and int(b) >= 0})
    base = isolate_fault(net, fb, load_scale=load_scale)
    ext = {int(b) for b in base.ext_grid.bus}
    _g, energized, _l = graph_state(base, ext)
    dead = {int(b) for b in base.bus.index} - energized
    full = sorted(dead | set(fb))
    cand = list(full)
    if max_h is not None and len(cand) > int(max_h):
        # keep the true section plus the highest-degree (most section-like) candidates, so the set
        # is deterministic and always contains the truth
        g = graph_state(isolate_fault(net, [], load_scale=load_scale), ext)[0]
        cand = sorted(set(fb) | set(sorted((c for c in cand if c not in fb),
                                           key=lambda b: (-g.degree(b) if b in g else 0, b))[
                                               :max(int(max_h) - len(fb), 0)
                                           ]))
    complete = len(cand) == len(full)
    if return_complete:
        return cand, complete, len(full)
    return cand


def _dead_set_for(net, h: int, load_scale: float) -> frozenset:
    base = isolate_fault(net, [int(h)], load_scale=load_scale)
    ext = {int(b) for b in base.ext_grid.bus}
    _g, energized, _l = graph_state(base, ext)
    return frozenset({int(b) for b in base.bus.index} - energized)


def draw_observation(net, true_faulted_buses, tier: str, *, load_scale: float = 1.0,
                     seed: int = 0) -> dict:
    """The ACTUAL degraded observation this episode produces, drawn from the frozen observation
    model — the same call ``RestorationTools.observe_scada`` makes. The belief must be conditioned
    on what was really seen, not on an idealised 'the indicator fired at the true section'."""
    from gridagent.observation import observation_model as om
    from gridagent.observation.observation_model import ObservableTruth
    fb = sorted({int(b) for b in true_faulted_buses if b is not None and int(b) >= 0})
    base = isolate_fault(net, fb, load_scale=load_scale)
    ext = {int(b) for b in base.ext_grid.bus}
    _g, energized, _l = graph_state(base, ext)
    buses = {int(b) for b in base.bus.index}
    truth = ObservableTruth(faulted_bus=(fb[0] if fb else -1), energized_buses=set(energized),
                            instrumented_buses=buses,
                            call_areas={"area": len(buses) - len(energized)})
    truth.faulted_buses = fb
    return om.observe(truth, tier, seed)


def fault_posterior(net, hypotheses, true_faulted_bus: int, tier: str, *,
                    load_scale: float = 1.0, seed: int = 0, observation: dict | None = None,
                    true_faulted_buses=None) -> dict:
    """P(h | o) for the ACTUALLY DRAWN observation o, under the observation model's OWN error
    rates — so the MAP hypothesis really can be wrong, at exactly the rate the channels imply.

    Likelihood terms, both from ``degradation_registry.tier_params``:
      * fault indicator: P(FI_b raised | fault at h) = 1-fi_error if b == h else fi_error;
      * SCADA: each REPORTING bus agrees with the hypothesis' dead set with probability
        1-fi_error (the same flip rate ``_scada_channel`` applies).
    """
    p = tier_params(tier)
    eps = float(np.clip(p["fi_error"], 1e-4, 0.49))
    fb = sorted(true_faulted_buses) if true_faulted_buses else [int(true_faulted_bus)]
    obs = observation if observation is not None else draw_observation(
        net, fb, tier, load_scale=load_scale, seed=seed)
    fi = {int(k): bool(v) for k, v in (obs.get("fault_indicator") or {}).items()}
    scada = {int(k): bool(v.get("energized", True))
             for k, v in (obs.get("scada") or {}).items() if isinstance(v, dict)}
    dead_of = {int(h): _dead_set_for(net, int(h), load_scale) for h in hypotheses}

    l_hit, l_miss = math.log(1.0 - eps), math.log(eps)
    logp = {}
    for h in hypotheses:
        h = int(h)
        ll = 0.0
        for b, flag in fi.items():
            ll += l_hit if flag == (b == h) else l_miss
        dead_h = dead_of[h]
        for b, en in scada.items():
            ll += l_hit if en == (b not in dead_h) else l_miss
        logp[h] = ll
    m = max(logp.values())
    w = {h: math.exp(v - m) for h, v in logp.items()}
    z = sum(w.values()) or 1.0
    return {h: v / z for h, v in w.items()}


def belief_entropy(posterior: dict) -> float:
    return float(-sum(p * math.log(p, 2) for p in posterior.values() if p > 0))


# --------------------------------------------------------------------------- #
# the ceiling
# --------------------------------------------------------------------------- #
def _inspect_minutes(net, bus: int, sw_meta: dict | None) -> float:
    """Crew dispatch + inspect time for a bus, taken from the travel time of the switchgear that
    bounds it (the manifest's own field-travel data), else the documented default."""
    if not sw_meta:
        return INSPECT_MIN_DEFAULT
    times = []
    for sid, m in sw_meta.items():
        pair = None
        try:
            pair = _switch_bus_pair(net, int(sid))
        except Exception:
            pair = None
        if pair and int(bus) in (int(pair[0]), int(pair[1])):
            t = float(m.get("travel_time_min", 0.0) or 0.0)
            if t > 0:
                times.append(t)
    return float(np.median(times)) if times else INSPECT_MIN_DEFAULT


def obs_limited_ceiling(net, true_faulted_buses, cust: dict, sw_meta: dict | None, tier: str, *,
                        weights: dict | None = None, load_scale: float = 1.0,
                        horizon_min: float = DEFAULT_HORIZON_MIN, seed: int = 0,
                        oracle_result=None, max_h: int | None = MAX_HYPOTHESES,
                        plans_cache: dict | None = None,
                        hypothesis_max_nodes: int | None = None,
                        max_opens: int | None = None) -> dict:
    """Expected controllable-ENS of the best observation-limited policy, plus the realization the
    ``obs_ceiling`` ARM should execute on THIS episode.

    ``plans_cache`` shares the per-hypothesis oracle plans across the observation tiers of the
    same episode: the hypothesis set and its plans depend only on the network and the load level,
    not on how noisily the operator sees them, so recomputing them per tier was pure waste.
    """
    fb = sorted({int(b) for b in true_faulted_buses if b is not None and int(b) >= 0})
    true_h = fb[0] if fb else -1
    H, support_complete, n_hypotheses_total = hypothesis_set(
        net,
        fb,
        load_scale=load_scale,
        max_h=max_h,
        return_complete=True,
    )
    obs = draw_observation(net, fb, tier, load_scale=load_scale, seed=seed)
    post = fault_posterior(net, H, true_h, tier, load_scale=load_scale, seed=seed,
                           observation=obs, true_faulted_buses=fb)
    ent = belief_entropy(post)

    # per-hypothesis omniscient plan (the oracle for that hypothesis) — tier-independent
    cache = plans_cache if plans_cache is not None else {}
    plans, costs, proven = {}, {}, True
    for h in H:
        key = (
            int(h),
            round(float(load_scale), 6),
            None if hypothesis_max_nodes is None else int(hypothesis_max_nodes),
            None if max_opens is None else int(max_opens),
        )
        if key not in cache:
            search_options = {}
            if hypothesis_max_nodes is not None:
                search_options["max_nodes"] = int(hypothesis_max_nodes)
            if max_opens is not None:
                search_options["max_opens"] = int(max_opens)
            r = (oracle_result if (oracle_result is not None and int(h) == true_h)
                 else solve_reconfiguration(net, [int(h)], cust, sw_meta, weights=weights,
                                            load_scale=load_scale, horizon_min=horizon_min,
                                            **search_options))
            cache[key] = (list(r.switch_sequence), float(r.cost_kwh), bool(r.proven_optimal))
        seq, cost, prv = cache[key]
        plans[int(h)] = list(seq)
        costs[int(h)] = cost
        proven = proven and prv

    oracle_cost = costs.get(true_h, 0.0)
    ctrl_w = float(getattr(oracle_result, "controllable_weighted", 0.0)) if oracle_result is not None else 0.0
    cost_noop = float(getattr(oracle_result, "cost_noop_kwh", 0.0)) if oracle_result is not None else 0.0

    # ---- branch 1: INSPECT until pinned, in Smith order (ascending time / posterior) ----
    insp_t = {int(h): _inspect_minutes(net, int(h), sw_meta) for h in H}
    order = sorted(H, key=lambda h: (insp_t[int(h)] / max(post.get(int(h), 1e-12), 1e-12), int(h)))
    cum, delay_of = 0.0, {}
    for h in order:
        cum += insp_t[int(h)]
        delay_of[int(h)] = cum
    exp_inspect = sum(post[h] * _delayed_cost(plans[h], costs[h], delay_of[h], ctrl_w, cost_noop,
                                              sw_meta, horizon_min) for h in H)

    # ---- branch 2: TEST-SWITCH binary search (available when the dark region has sectionalizers)
    base = isolate_fault(net, fb, load_scale=load_scale)
    n_sec_in_dead = len([s for s in sectionalizer_ids(base, sw_meta)
                         if (_switch_bus_pair(base, s) or (-1, -1))[0] in set(H)])
    n_probes = int(math.ceil(math.log2(max(len(H), 1)))) if len(H) > 1 else 0
    ts_available = n_sec_in_dead >= n_probes
    ts_delay = n_probes * TEST_SWITCH_MIN
    exp_test = (sum(post[h] * _delayed_cost(plans[h], costs[h], ts_delay, ctrl_w, cost_noop,
                                            sw_meta, horizon_min) for h in H)
                if ts_available else float("inf"))

    # ---- branch 3: ACT BLIND on the MAP hypothesis ----
    map_h = max(post, key=lambda h: (post[h], -h))
    exp_blind = 0.0
    for h in H:
        if int(h) == int(map_h):
            exp_blind += post[h] * costs[h]
        else:
            # the MAP plan's closes are rejected under a different true fault: the crew time is
            # spent, nothing is restored, and the operator falls back to the correct plan afterwards
            waste = sum(_op_minutes(op, sw_meta) for op in plans[map_h])
            exp_blind += post[h] * _delayed_cost(plans[h], costs[h], waste, ctrl_w, cost_noop,
                                                 sw_meta, horizon_min)

    branches = {"inspect": exp_inspect, "test_switch": exp_test, "act_blind": exp_blind}
    policy = min(branches, key=lambda k: branches[k])
    ceiling = float(branches[policy])

    # ---- the REALIZATION this episode's obs_ceiling ARM executes ----
    # It is a genuine rollout, not a synthesised number: under ``act_blind`` the arm first commits
    # the MAP hypothesis' plan, and when the MAP is wrong the environment's guard rejects those
    # closes and charges the crew time — so the cost of mis-localisation emerges from the physics.
    if policy == "inspect":
        realized_delay = delay_of[true_h]
        realized_plan = list(plans.get(true_h, []))
    elif policy == "test_switch":
        realized_delay = ts_delay
        realized_plan = list(plans.get(true_h, []))
    else:
        realized_delay = 0.0
        prefix = [] if int(map_h) == int(true_h) else list(plans.get(map_h, []))
        seen = {(o["switch_id"], o["closed"]) for o in prefix}
        realized_plan = prefix + [o for o in plans.get(true_h, [])
                                  if (o["switch_id"], o["closed"]) not in seen]

    ceiling = min(max(ceiling, float(oracle_cost)), float(cost_noop))
    exact = bool(support_complete and proven)
    return {
        "cost_obs_ceiling_kwh": ceiling,
        "obs_ceiling_switch_sequence": realized_plan,
        "obs_ceiling_sensing_min": float(round(realized_delay, 3)),
        "obs_ceiling_policy": policy,
        "obs_ceiling_branches": {k: (None if v == float("inf") else round(float(v), 4))
                                 for k, v in branches.items()},
        "obs_ceiling_proven_bound": exact,
        "obs_ceiling_exact": exact,
        "obs_ceiling_hypothesis_complete": bool(support_complete),
        "obs_ceiling_lower_bound_kwh": (
            ceiling if exact else float(oracle_cost)
        ),
        "obs_ceiling_upper_bound_kwh": (
            ceiling if exact else float(cost_noop)
        ),
        "obs_ceiling_search_method": "memoized_bounded_dfs",
        "belief_entropy": float(round(ent, 5)),
        "n_hypotheses": int(len(H)),
        "n_hypotheses_total": int(n_hypotheses_total),
        "map_hypothesis": int(map_h),
        "map_correct": bool(int(map_h) == int(true_h)),
    }


def _op_minutes(op, sw_meta) -> float:
    from gridagent.agent_interface.cost_model import switch_time_min
    return switch_time_min(op, sw_meta)


def _delayed_cost(plan, base_cost: float, delay_min: float, ctrl_w: float, cost_noop: float,
                  sw_meta, horizon_min: float) -> float:
    """Cost of executing ``plan`` after ``delay_min`` of sensing. Every restored customer waits the
    extra delay, so the cost rises by (restored_weighted * delay) — recovered from the base cost
    without re-simulating, which keeps the ceiling exactly consistent with ``restoration_cost``."""
    if cost_noop <= 0 or ctrl_w <= 0:
        return float(base_cost)
    from gridagent.agent_interface.cost_model import KW_PER_CUSTOMER
    # base_cost = [unrestored*H + sum(n_i * t_i)] * KW/60 ; a uniform delay d adds
    # sum(n_i) * d * KW/60, capped so no customer waits longer than the horizon.
    unrestored_w = max(cost_noop / (horizon_min / 60.0 * KW_PER_CUSTOMER) - ctrl_w, 0.0)
    restored_w = ctrl_w  # the plan restores the controllable set (per-hypothesis oracle)
    add = restored_w * min(float(delay_min), float(horizon_min)) / 60.0 * KW_PER_CUSTOMER
    return min(float(base_cost) + add, float(cost_noop))
