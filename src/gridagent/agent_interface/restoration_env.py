"""GridAgent-Bench v2 — stateful FLISR RestorationEpisode engine (Scope §16.1, GOAL_v2 M2/M4/M7).

A multi-step POMDP: the agent restores a half-blind distribution feeder after a fault, reading
DEGRADED observations and issuing sensing / switching actions under tool + time budgets.

What v2 changed
---------------
* **One clock, one cost model.** The episode no longer invents its own restoration time. The
  completion time of the *k*-th committed operation is
  ``cost_model.op_completion_times(committed_ops, sw_meta)[k]`` — the same function every arm is
  scored with (M1). Switching time is a property of the SWITCH (remote vs crew travel), never of
  the arm, so no arm can be faster than the oracle by accident.
* **One starting state.** The episode isolates the fault with ``reconfig_search.isolate_fault``,
  the exact routine the oracle search starts from, including the load-trajectory scale. The oracle
  therefore cannot propose a plan the env would not reproduce.
* **PF in the guard (M4).** A close that is thermally infeasible (>100.5 % loading) or violates
  EN 50160 (±10 % voltage) is REJECTED pre-commit, exactly as the oracle search rejects it. Guard
  rejections are *the safety system working* — they are counted as ``blocked_*`` attempts, NOT as
  hard-fails (M2). A hard-fail is an unblocked, committed unsafe FINAL state, which is what
  ``final_state_safety`` measures. ``enforce_pf=False`` exists so an intentionally-unsafe arm can
  be constructed for the safety test; no shipped arm uses it.
* **Explicit responsibility boundary.** Production episodes begin after protection-assisted fault
  isolation and therefore expose ``RESTORE`` as their initial operator stage. The intentionally
  unisolated guard-test path (``isolate_bus=False``) begins at ``LOCATE``. Sectionalizers may be
  OPENED only as true-dark cuts, so restoration can split an interrupted island without
  re-interrupting live customers. ``test_switch`` remains available to the generic environment as
  a charged momentary-interruption probe, but fault location is not part of the reported model arm.

Determinism (hard): fixed travel matrix, fixed tie-break ordering, NO RNG at step time — seeds
vary only LLM sampling and the observation noise draw.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import networkx as nx

from gridagent.agent_interface.cost_model import (
    DEFAULT_HORIZON_MIN, op_completion_times, switch_time_min,
)
from gridagent.oracle.reconfig_search import DEFAULT_MAX_DEPTH, isolate_fault
from gridagent.simulation.grid_bank import (
    RadialityViolation, collapsed_graph, energized_buses, n_independent_loops, source_buses,
    unsupplied, _switch_bus_pair,
)
from gridagent.simulation.pf_feasibility import pf_feasible


DEFAULT_MAX_COMMITTED_OPS = DEFAULT_MAX_DEPTH
DEFAULT_MAX_SECTIONALIZER_OPENS = 3

MAX_TOTAL_CALLS = 20
MAX_SIMULATE_SWITCH_CALLS = 8
MAX_STEPS = 12
MAX_RESTORATION_MINUTES = DEFAULT_HORIZON_MIN     # 240 min
TEST_SWITCH_MIN = 12.0                            # a test-switch + reclose cycle, minutes


class EnergizationViolation(Exception):
    """Raised when an action would energize a known-faulted element."""


class PowerFlowViolation(Exception):
    """Raised when a close would violate a thermal rating or the EN 50160 voltage band."""


class LiveSectionalizerOpenViolation(Exception):
    """Raised when an OPEN would interrupt energized load or is not a dark cut."""


class ControllerActionViolation(Exception):
    """Raised when a switch operation is outside the declared controller class."""


def energized_graph(net) -> nx.Graph:
    """Collapsed simple graph of the in-service / closed-switch network (parallel edges merged)."""
    return collapsed_graph(net)


@dataclass
class Fault:
    """A fault incident. ``faulted_buses`` is a SET so a storm can carry concurrent faults (M8)."""
    faulted_lines: list = field(default_factory=list)
    faulted_buses: list = field(default_factory=list)
    earthing_regime: str = "resonant/compensated"
    load_scale: float = 1.0

    def __init__(self, faulted_lines=None, faulted_bus=None, faulted_buses=None,
                 earthing_regime: str = "resonant/compensated", load_scale: float = 1.0):
        self.faulted_lines = list(faulted_lines or [])
        buses = list(faulted_buses) if faulted_buses is not None else []
        if faulted_bus is not None and int(faulted_bus) >= 0:
            buses.append(int(faulted_bus))
        self.faulted_buses = sorted({int(b) for b in buses if b is not None and int(b) >= 0})
        self.earthing_regime = earthing_regime
        self.load_scale = float(load_scale)

    @property
    def faulted_bus(self) -> int:
        """Back-compat scalar view: the first faulted bus (-1 when there is none)."""
        return self.faulted_buses[0] if self.faulted_buses else -1

    @property
    def n_concurrent_faults(self) -> int:
        return len(self.faulted_buses)


@dataclass
class Crew:
    crew_id: int
    available_at: float = 0.0    # clock minute the crew is free


class RestorationEpisode:
    def __init__(self, net, fault: Fault, *, sw_meta: dict | None = None, crews: int = 2,
                 config_hash: str = "v2", oracle_version: str = "v2",
                 enforce_pf: bool = True, enforce_radiality: bool = True,
                 enforce_energization: bool = True, isolate_bus: bool = True,
                 horizon_min: float = MAX_RESTORATION_MINUTES,
                 max_committed_ops: int = DEFAULT_MAX_COMMITTED_OPS,
                 max_sectionalizer_opens: int | None = None):
        self.sw_meta = sw_meta or {}
        self.fault = fault
        self.horizon_min = float(horizon_min)
        self.max_committed_ops = int(max_committed_ops)
        if self.max_committed_ops <= 0:
            raise ValueError("max_committed_ops must be positive")
        self.max_sectionalizer_opens = int(
            min(DEFAULT_MAX_SECTIONALIZER_OPENS, self.max_committed_ops)
            if max_sectionalizer_opens is None
            else max_sectionalizer_opens
        )
        if not 0 <= self.max_sectionalizer_opens <= self.max_committed_ops:
            raise ValueError(
                "max_sectionalizer_opens must be between zero and "
                "max_committed_ops"
            )
        self.enforce_pf = bool(enforce_pf)
        self.enforce_radiality = bool(enforce_radiality)
        self.enforce_energization = bool(enforce_energization)
        # IDENTICAL starting state to the oracle search: load trajectory + faulted-section
        # isolation. ``isolate_bus=False`` leaves the section only PARTIALLY isolated (the faulted
        # lines are out but the sectionalizers around the faulted bus are not) — the scenario the
        # energization guard exists for, used by the guard tests.
        self.net = isolate_fault(net, fault.faulted_buses if isolate_bus else [],
                                 load_scale=fault.load_scale)
        for li in fault.faulted_lines:
            if li in self.net.line.index:
                self.net.line.at[li, "in_service"] = False
        typed_ties = {
            int(switch_id) for switch_id, metadata in self.sw_meta.items()
            if metadata.get("type") == "tie"
        }
        typed_sectionals = {
            int(switch_id) for switch_id, metadata in self.sw_meta.items()
            if metadata.get("type") in ("sectionalizer", "RMU/LBS")
        }
        initially_open = {
            int(switch_id)
            for switch_id in self.net.switch.index[~self.net.switch.closed]
        }
        initially_closed = {
            int(switch_id)
            for switch_id in self.net.switch.index[self.net.switch.closed]
        }
        self.controller_tie_ids = (
            typed_ties & initially_open if typed_ties else initially_open
        )
        self.controller_sectionalizer_ids = (
            typed_sectionals & initially_closed
            if typed_sectionals else initially_closed
        )

        self.crews = [Crew(i) for i in range(crews)]
        self.config_hash = config_hash
        self.oracle_version = oracle_version
        self.n_calls = 0
        self.n_steps = 0
        self.done = False
        # The production controller takes over after ``isolate_fault`` above. Advertising LOCATE
        # here contradicted both the physical starting state and the frozen model briefing. Keep
        # LOCATE only for the explicitly unisolated positive-control path.
        self.stage = "RESTORE" if isolate_bus else "LOCATE"

        self.committed_ops: list[dict] = []          # ordered [{'switch_id', 'closed'}]
        self.sensing_at_op: list[float] = []         # sensing minutes accrued before each op
        self.restore_events: list[tuple] = []        # [(frozenset(newly energized buses), op_idx)]
        self.shed_events: list[tuple] = []           # [(frozenset(de-energized buses), op_idx)]
        self.blocked = {"radiality": 0, "energization": 0, "power_flow": 0}
        self.sensing_minutes = 0.0                   # crew_inspect / test_switch / wasted-trip time
        self.attempt_waste_min = 0.0                 # time burnt on guard-rejected attempts
        self.n_test_switch = 0
        self.n_momentary_interruptions = 0
        self.n_noop_operations = 0

        self.energized = energized_buses(self.net)
        self.energized0 = set(self.energized)
        self.out0 = unsupplied(self.net)

    # -------------------------------------------------------------------- #
    # clock — derived from the SHARED cost model, never invented here
    # -------------------------------------------------------------------- #
    @property
    def clock_min(self) -> float:
        t = op_completion_times(self.committed_ops, self.sw_meta)
        return min((t[-1] if t else 0.0) + self.sensing_minutes, self.horizon_min)

    def op_times(self) -> list[float]:
        """Completion minute of each committed op = cumulative switching time + the sensing time
        that had already been spent when that op was issued."""
        base = op_completion_times(self.committed_ops, self.sw_meta)
        pad = self.sensing_at_op + [self.sensing_minutes] * max(len(base) - len(self.sensing_at_op), 0)
        return [min(x + pad[i], self.horizon_min) for i, x in enumerate(base)]

    def energized_buses(self) -> set:
        return energized_buses(self.net)

    @property
    def switch_ops_remaining(self) -> int:
        return max(self.max_committed_ops - len(self.committed_ops), 0)

    @property
    def sectionalizer_opens(self) -> int:
        return sum(not bool(operation["closed"]) for operation in self.committed_ops)

    @property
    def sectionalizer_opens_remaining(self) -> int:
        return max(
            self.max_sectionalizer_opens - self.sectionalizer_opens,
            0,
        )

    def _faulted_buses(self) -> set:
        return set(self.fault.faulted_buses)

    # -------------------------------------------------------------------- #
    # guard
    # -------------------------------------------------------------------- #
    def _charge_failed_attempt(self, switch_id: int) -> None:
        """A rejected attempt is not free: the crew still drove to the manual switch, and the
        dispatcher still spent the SCADA cycle on a remote one. Charging it is what gives partial
        observability a real, physical price — a policy that mis-localizes wastes crew minutes
        while customers stay dark. The oracle plan never triggers a rejection, so this cannot
        make ``regret(oracle) != 0``."""
        self.attempt_waste_min += switch_time_min(int(switch_id), self.sw_meta)
        self.sensing_minutes += switch_time_min(int(switch_id), self.sw_meta)

    def _guard_close(self, switch_id: int) -> None:
        """Guard a switch CLOSE pre-commit: mesh, fault re-energization, AC-PF feasibility.
        A rejection leaves the network state UNCHANGED and is the safety system working (M2), not
        a hard-fail — but it does consume the attempt's switching time."""
        if int(switch_id) not in self.controller_tie_ids:
            self._charge_failed_attempt(switch_id)
            raise ControllerActionViolation(
                f"switch {switch_id} is not an initially-open controller tie"
            )
        trial = copy.deepcopy(self.net)
        trial.switch.at[switch_id, "closed"] = True
        if self.enforce_radiality and n_independent_loops(trial) != 0:
            self.blocked["radiality"] += 1
            self._charge_failed_attempt(switch_id)
            raise RadialityViolation(
                f"closing switch {switch_id} creates a mesh (radiality violation)")
        faulted = self._faulted_buses()
        new_energized = energized_buses(trial)
        if self.enforce_energization and (faulted & new_energized):
            self.blocked["energization"] += 1
            self._charge_failed_attempt(switch_id)
            raise EnergizationViolation(
                f"closing switch {switch_id} would energize known-faulted element "
                f"{sorted(faulted & new_energized)}")
        if self.enforce_pf:
            ok, metrics = pf_feasible(trial)
            if not ok:
                self.blocked["power_flow"] += 1
                self._charge_failed_attempt(switch_id)
                raise PowerFlowViolation(
                    f"closing switch {switch_id} is not AC-PF feasible "
                    f"(max_line_loading={metrics.get('max_line_loading_pct')}%, "
                    f"vm_pu=[{metrics.get('min_vm_pu')},{metrics.get('max_vm_pu')}])")

    def _guard_open(self, switch_id: int) -> None:
        """Permit only a sectionalizing cut whose endpoints are truly dark."""
        if int(switch_id) not in self.controller_sectionalizer_ids:
            self._charge_failed_attempt(switch_id)
            raise ControllerActionViolation(
                f"switch {switch_id} is not an initially-closed sectionalizer"
            )
        pair = _switch_bus_pair(self.net, int(switch_id))
        if (
            pair is None
            or int(pair[0]) in self.energized
            or int(pair[1]) in self.energized
        ):
            self._charge_failed_attempt(switch_id)
            raise LiveSectionalizerOpenViolation(
                f"opening switch {switch_id} would interrupt energized load"
            )
        trial = copy.deepcopy(self.net)
        trial.switch.at[switch_id, "closed"] = False
        graph = collapsed_graph(trial)
        if (
            int(pair[0]) in graph
            and int(pair[1]) in graph
            and nx.has_path(graph, int(pair[0]), int(pair[1]))
        ):
            self._charge_failed_attempt(switch_id)
            raise LiveSectionalizerOpenViolation(
                f"opening switch {switch_id} is not a dark sectionalizing cut"
            )

    # -------------------------------------------------------------------- #
    # actions
    # -------------------------------------------------------------------- #
    def operate(self, switch_id: int, closed: bool = True) -> dict:
        """Commit one switching operation. Raises on a guard rejection (state unchanged)."""
        sid = int(switch_id)
        if self.switch_ops_remaining <= 0:
            return {
                "applied": False,
                "operation_budget_exhausted": True,
                "n_gained": 0,
                "n_lost": 0,
                "reason": (
                    f"the {self.max_committed_ops}-operation switching budget "
                    "is exhausted"
                ),
                "clock_min": round(self.clock_min, 1),
            }
        # A switch already in the requested position is a NO-OP, not an operation. Recording it as
        # a committed op charged the crew's travel time for a trip nobody makes, and let an agent
        # burn its whole budget "opening" switches that were already open (observed in the 7B
        # transcripts). Report it and change nothing.
        if bool(self.net.switch.at[sid, "closed"]) == bool(closed):
            self.n_noop_operations += 1
            return {"applied": False, "noop": True, "n_gained": 0, "n_lost": 0,
                    "reason": f"switch {sid} is already {'closed' if closed else 'open'}",
                    "clock_min": round(self.clock_min, 1)}
        if not bool(closed) and self.sectionalizer_opens_remaining <= 0:
            return {
                "applied": False,
                "sectionalizer_open_budget_exhausted": True,
                "n_gained": 0,
                "n_lost": 0,
                "reason": (
                    f"the {self.max_sectionalizer_opens}-sectionalizer-open "
                    "budget is exhausted"
                ),
                "clock_min": round(self.clock_min, 1),
            }
        if bool(closed):
            self._guard_close(sid)
        else:
            self._guard_open(sid)
        before = set(self.energized)
        self.net.switch.at[sid, "closed"] = bool(closed)
        self.energized = energized_buses(self.net)
        idx = len(self.committed_ops)
        self.committed_ops.append({"switch_id": sid, "closed": bool(closed)})
        self.sensing_at_op.append(float(self.sensing_minutes))
        gained = self.energized - before
        lost = before - self.energized
        if gained:
            self.restore_events.append((frozenset(gained), idx))
        if lost:
            self.shed_events.append((frozenset(lost), idx))
        if bool(closed):
            self.stage = "RESTORE"
        elif self.stage == "LOCATE":
            self.stage = "ISOLATE"
        return {"applied": True, "n_gained": len(gained), "n_lost": len(lost),
                "clock_min": round(self.clock_min, 1)}

    def test_switch(self, switch_id: int) -> dict:
        """LOCATE probe (M7): open a sectionalizer and re-close the head breaker on the TRUE net.

        Returns ONLY the binary protection outcome — 'held' when the opened switch separates every
        faulted bus from every source (the fault is downstream of the opened point), 're_tripped'
        otherwise. It NEVER returns the fault edge. Costs a momentary re-interruption of the
        customers beyond the opened point plus ``TEST_SWITCH_MIN`` of clock, and the switch is
        restored to its prior state afterwards (the probe is not an isolation).
        """
        sid = int(switch_id)
        prior = bool(self.net.switch.at[sid, "closed"])
        if not prior:
            # Probing an already-open switch tells you nothing: the section beyond it is already
            # separated, so protection trivially "holds". Returning a confident 'held' here sent
            # the agent down a dead end, so the probe reports that it was uninformative instead.
            self.n_test_switch += 1
            return {"outcome": "uninformative"}
        trial = copy.deepcopy(self.net)
        trial.switch.at[sid, "closed"] = False
        # re-close the head: with the section beyond `sid` separated, would the fault still be fed?
        g = collapsed_graph(trial)
        srcs = {b for b in source_buses(trial) if b in g}
        reachable = set()
        for s in srcs:
            reachable |= nx.node_connected_component(g, s)
        held = not (self._faulted_buses() & reachable)
        self.n_test_switch += 1
        self.n_momentary_interruptions += 1
        self.sensing_minutes += TEST_SWITCH_MIN if not bool(
            self.sw_meta.get(sid, {}).get("remote", False)) else min(TEST_SWITCH_MIN, 3.0)
        self.net.switch.at[sid, "closed"] = prior
        return {"outcome": "held" if held else "re_tripped"}

    def charge_sensing_time(self, minutes: float) -> None:
        """A crew inspection / dispatch consumes clock without changing the network state."""
        self.sensing_minutes += max(float(minutes), 0.0)

    # -------------------------------------------------------------------- #
    # legacy step() facade (kept so existing callers/tests keep working)
    # -------------------------------------------------------------------- #
    def step(self, action: dict) -> dict:
        self.n_calls += 1
        self.n_steps += 1
        verb = action.get("verb")
        if verb == "operate_switch":
            self.operate(int(action["switch_id"]), bool(action.get("closed", True)))
        elif verb == "end_episode":
            self.done = True
        return self._observation()

    # -------------------------------------------------------------------- #
    # final-state safety (M2): a hard-fail is an UNBLOCKED, COMMITTED unsafe state
    # -------------------------------------------------------------------- #
    def final_state_safety(self) -> dict:
        ok, metrics = pf_feasible(copy.deepcopy(self.net))
        loops = n_independent_loops(self.net)
        energized = energized_buses(self.net)
        fault_live = bool(self._faulted_buses() & energized)
        hf_overload = int(metrics.get("converged", False) and not metrics.get("thermal_ok", True))
        hf_voltage = int(metrics.get("converged", False) and not metrics.get("voltage_ok", True))
        hf_diverged = int(not metrics.get("converged", False))
        hf_radiality = int(loops != 0)
        hf_energized_fault = int(fault_live)
        hardfail = int(bool(hf_overload or hf_voltage or hf_diverged or hf_radiality
                            or hf_energized_fault))
        return {
            "flisr_hardfail": hardfail,
            "hf_overload_backfeed": hf_overload,
            "hf_voltage_violation": hf_voltage,
            "hf_pf_diverged": hf_diverged,
            "hf_lost_radiality": hf_radiality,
            "hf_energized_fault": hf_energized_fault,
            "final_max_line_loading_pct": metrics.get("max_line_loading_pct"),
            "final_min_vm_pu": metrics.get("min_vm_pu"),
            "final_n_loops": int(loops),
            "blocked_radiality": int(self.blocked["radiality"]),
            "blocked_energization": int(self.blocked["energization"]),
            "blocked_power_flow": int(self.blocked["power_flow"]),
            "blocked_total": int(sum(self.blocked.values())),
            "attempt_waste_min": float(round(self.attempt_waste_min, 2)),
        }

    # -------------------------------------------------------------------- #
    def restored_bus_events(self) -> list[tuple]:
        """[(frozenset(buses), op_index)] — the buses each committed op brought back."""
        return list(self.restore_events)

    def _observation(self) -> dict:
        return {
            "clock_min": round(self.clock_min, 1),
            "n_energized": len(self.energized_buses()),
            "n_out": len(unsupplied(self.net)),
            "stage": self.stage,
            "budget_left": MAX_TOTAL_CALLS - self.n_calls,
            "switch_operations_remaining": self.switch_ops_remaining,
        }
