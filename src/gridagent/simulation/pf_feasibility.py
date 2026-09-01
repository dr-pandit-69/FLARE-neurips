"""GridAgent-Bench v2 — AC power-flow feasibility (thermal + EN 50160 voltage).  [GOAL_v2 M4]

The v1 code called a reconfiguration "feasible" whenever ``pp.runpp`` converged. pandapower
converges happily at 250 % line loading, so a back-feed that thermally destroys the neighbouring
feeder was labelled feasible and the "capacity-constrained" difficulty tier could never exist.

``pf_feasible`` is the single feasibility predicate used by BOTH the oracle candidate search
(``restoration_oracle._restore_config``) and the environment guard
(``restoration_env.RestorationEpisode``), so the oracle can never propose something the env would
reject and vice versa.

Criteria (all must hold):
  * the AC power flow converged;
  * ``res_line.loading_percent.max() <= MAX_LOADING_PCT`` (thermal, 100 % + solver tolerance);
  * ``res_trafo.loading_percent.max() <= MAX_LOADING_PCT`` when transformers are present;
  * every energized bus voltage in ``[V_MIN_PU, V_MAX_PU]`` = [0.90, 1.10] pu (EN 50160 ±10 %).

De-energized buses are EXCLUDED from the voltage test: an out-of-service island solves to
``vm_pu = NaN`` (or 0), which is an outage, not a voltage violation.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp

MAX_LOADING_PCT = 100.5      # 100 % rating + 0.5 % solver tolerance
V_MIN_PU = 0.90              # EN 50160 lower limit
V_MAX_PU = 1.10              # EN 50160 upper limit
_V_DEENERGIZED = 0.10        # below this a bus is treated as de-energized, not under-voltage


def run_pf(net, *, quiet: bool = True) -> bool:
    """Solve the AC power flow in place. Returns True iff it converged."""
    try:
        pp.runpp(net, numba=True)
        return bool(net.converged)
    except Exception:
        try:
            pp.runpp(net, numba=False)
            return bool(net.converged)
        except Exception:
            return False


def pf_feasible(net, *, solve: bool = True) -> tuple[bool, dict]:
    """(feasible, metrics) under thermal + EN 50160 voltage limits.

    solve=False assumes ``net`` already carries a converged result table (avoids a redundant
    ``runpp`` when the caller has just solved).
    """
    converged = run_pf(net) if solve else bool(getattr(net, "converged", False))
    metrics = {
        "converged": bool(converged),
        "max_line_loading_pct": None,
        "max_trafo_loading_pct": None,
        "min_vm_pu": None,
        "max_vm_pu": None,
        "n_undervoltage": 0,
        "n_overvoltage": 0,
        "thermal_ok": bool(converged),
        "voltage_ok": bool(converged),
    }
    if not converged:
        metrics["thermal_ok"] = metrics["voltage_ok"] = False
        return False, metrics

    # ---- thermal ----
    max_line = 0.0
    if len(net.res_line):
        vals = net.res_line["loading_percent"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        max_line = float(vals.max()) if vals.size else 0.0
    max_trafo = 0.0
    if hasattr(net, "res_trafo") and len(net.res_trafo):
        vals = net.res_trafo["loading_percent"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        max_trafo = float(vals.max()) if vals.size else 0.0
    metrics["max_line_loading_pct"] = round(max_line, 3)
    metrics["max_trafo_loading_pct"] = round(max_trafo, 3)
    thermal_ok = (max_line <= MAX_LOADING_PCT) and (max_trafo <= MAX_LOADING_PCT)

    # ---- voltage on the ENERGIZED buses only ----
    vm = net.res_bus["vm_pu"].to_numpy(dtype=float)
    live = vm[np.isfinite(vm) & (vm > _V_DEENERGIZED)]
    if live.size:
        metrics["min_vm_pu"] = round(float(live.min()), 4)
        metrics["max_vm_pu"] = round(float(live.max()), 4)
        metrics["n_undervoltage"] = int((live < V_MIN_PU).sum())
        metrics["n_overvoltage"] = int((live > V_MAX_PU).sum())
    voltage_ok = (metrics["n_undervoltage"] == 0) and (metrics["n_overvoltage"] == 0)

    metrics["thermal_ok"] = bool(thermal_ok)
    metrics["voltage_ok"] = bool(voltage_ok)
    return bool(thermal_ok and voltage_ok), metrics


def violation_reason(metrics: dict) -> str:
    """Short human/label-friendly reason string for an infeasible result ('' when feasible)."""
    if not metrics.get("converged", False):
        return "pf_diverged"
    bad = []
    if not metrics.get("thermal_ok", True):
        bad.append(f"thermal({metrics.get('max_line_loading_pct')}%)")
    if not metrics.get("voltage_ok", True):
        bad.append(f"voltage([{metrics.get('min_vm_pu')},{metrics.get('max_vm_pu')}]pu)")
    return "+".join(bad)
