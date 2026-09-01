"""GridAgent-Bench v2 — the ONE switching-time / restoration-cost model.  [GOAL_v2 M1]

v1 defect #1: each arm carried its own notion of "how long a switch takes". The oracle was
credited an idealised instant restore (``cost_oracle = _ens(restorable, REMOTE_MIN)``) while the
LLM was credited the *env clock*, so an LLM that closed a remote tie on turn 1 could be scored
BELOW the oracle — 68/395 cases where an arm "beat" the optimum. That is a metric bug, not a
result.

v2 fixes it structurally: switching time is a property of the SWITCH, never of the arm, and every
arm — floors, incumbents, oracle, LLM — is scored by the single function ``restoration_cost``.

Timing model (shared, deterministic):
  * a REMOTE switch operates in ``REMOTE_MIN`` minutes (SCADA-commanded);
  * a MANUAL switch costs its ``travel_time_min`` from the switch manifest (crew drives out);
  * committed operations are executed as ONE serialized control-room queue, so the completion time
    of op *k* is the cumulative sum of the op times up to and including *k*.

Cost is CONTROLLABLE ENS (kWh): customers never restored inside the horizon are charged the full
horizon; customers restored after op *k* are charged the time until that op completed. The
un-restorable faulted-section repair energy is identical across policies and excluded (Scope §17.1).

Because the ORACLE ARM is scored by this same function on its OWN realized switching sequence, the
oracle's cost is a real (non-zero, non-idealised) cost and ``regret(oracle) == 0`` exactly, while no
arm can be faster than the oracle unless the oracle search itself missed a better configuration —
which is what the ``ORACLE_BEATEN`` flag exists to surface.
"""
from __future__ import annotations

KW_PER_CUSTOMER = 1.2          # nominal per-customer demand proxy (kW)
REMOTE_MIN = 1.0               # a SCADA-commanded switch operates in ~1 minute
DEFAULT_MANUAL_MIN = 45.0      # fallback crew travel time when the manifest has none
DEFAULT_HORIZON_MIN = 240.0    # restoration horizon (MAX_RESTORATION_MINUTES)


def op_switch_id(op) -> int:
    """A committed operation is either a bare switch id or {'switch_id': int, 'closed': bool}."""
    if isinstance(op, dict):
        return int(op["switch_id"])
    return int(op)


def op_is_close(op) -> bool:
    return bool(op["closed"]) if isinstance(op, dict) else True


def switch_time_min(switch_id, sw_meta: dict | None) -> float:
    """Minutes to operate this switch. A property of the SWITCH — identical for every arm."""
    m = (sw_meta or {}).get(op_switch_id(switch_id), {})
    if bool(m.get("remote", False)):
        return REMOTE_MIN
    travel = float(m.get("travel_time_min", DEFAULT_MANUAL_MIN) or 0.0)
    # a manual operation cannot be instantaneous: a crew must at minimum reach and throw the switch
    return max(travel, REMOTE_MIN)


def op_completion_times(switch_ops, sw_meta: dict | None) -> list[float]:
    """Cumulative completion time (min) of each committed op in a serialized action queue."""
    t, out = 0.0, []
    for sid in switch_ops:
        t += switch_time_min(sid, sw_meta)
        out.append(t)
    return out


def restoration_cost(restored_customers, unrestored_customers: float, switch_ops,
                     sw_meta: dict | None, horizon_min: float = DEFAULT_HORIZON_MIN,
                     shed_events=None, extra_delay_min: float = 0.0) -> float:
    """Controllable ENS (kWh) of one arm's realized restoration. THE shared scoring function.

    restored_customers : int  -> that many customers came back when the LAST op completed, or
                         list[(n_customers, op_index)] -> n came back when ``switch_ops[op_index]``
                         completed (0-based). Use the list form whenever an arm restores load in
                         more than one stage.
    unrestored_customers : controllable customers still out at the horizon.
    switch_ops : the ORDERED operations the arm actually committed (guard-rejected attempts are
                 not operations and cost nothing).
    shed_events : list[(n_customers, op_index)] — HEALTHY customers the arm itself de-energized by
                  over-isolating. Charged from that op until the horizon.
    extra_delay_min : sensing time (crew inspection, test-switching) that delays every operation.
    """
    raw = op_completion_times(switch_ops, sw_meta)
    if isinstance(extra_delay_min, (list, tuple)):
        pad = list(extra_delay_min) + [0.0] * max(len(raw) - len(extra_delay_min), 0)
        times = [t + max(float(pad[i]), 0.0) for i, t in enumerate(raw)]
        base_delay = max(float(pad[0]), 0.0) if pad else 0.0
    else:
        times = [t + max(float(extra_delay_min), 0.0) for t in raw]
        base_delay = max(float(extra_delay_min), 0.0)
    hz = float(horizon_min)

    def _at(k: int) -> float:
        if not times:
            return base_delay
        return times[min(max(int(k), 0), len(times) - 1)]

    if isinstance(restored_customers, (int, float)):
        n = float(restored_customers)
        events = [(n, times[-1] if times else 0.0)] if n > 0 else []
    else:
        events = [(float(n), _at(k)) for n, k in restored_customers if float(n) > 0]

    cost = max(float(unrestored_customers), 0.0) * hz
    for n, t in events:
        cost += n * min(max(t, 0.0), hz)
    for n, k in (shed_events or []):
        if float(n) > 0:
            cost += float(n) * max(hz - min(max(_at(k), 0.0), hz), 0.0)
    return cost / 60.0 * KW_PER_CUSTOMER


def regret(cost_arm: float, cost_oracle: float, cost_noop: float) -> float:
    """Normalized regret in [0,1]. 0 = the oracle's own realized cost, 1 = restore nothing."""
    denom = max(float(cost_noop) - float(cost_oracle), 1e-9)
    r = (float(cost_arm) - float(cost_oracle)) / denom
    return float(min(max(r, 0.0), 1.0))


def raw_regret(cost_arm: float, cost_oracle: float, cost_noop: float) -> float:
    """UNCLIPPED regret — negative iff the arm genuinely beat the oracle (the ORACLE_BEATEN probe)."""
    denom = max(float(cost_noop) - float(cost_oracle), 1e-9)
    return (float(cost_arm) - float(cost_oracle)) / denom
