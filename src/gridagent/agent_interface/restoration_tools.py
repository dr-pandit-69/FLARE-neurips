"""GridAgent-Bench v2 — the FLISR tool interface.  [Scope §16.2-16.5; GOAL_v2 M3 / M7]

Intent-grouped tools over a ``RestorationEpisode``. Two anti-leak cores:

``simulate_switch`` reconstructs a BELIEF net from observations only and returns a coarse
feasibility class — never the true post-switch state.

``observed_customers_out`` (M3) replaces the v1 leak. v1 handed the agent
``len(unsupplied(true_net))`` verbatim every turn: an EXACT function of hidden state, which made
the partial-observability claim false. v2 derives the number the agent sees from the public
channels only — the customers on buses whose (degraded, sparsely instrumented) SCADA reports
de-energized, inflated by the instrumented fraction, blended with the noisy trouble-call count.
Two hidden truths a sensor cannot distinguish therefore produce the same estimate.

``test_switch`` (M7) is the LOCATE probe: open a sectionalizer, re-close the head breaker, and
learn only whether protection HELD or RE-TRIPPED. It costs a momentary re-interruption and clock
time, and its result is whitelisted to a single binary key so it can never surface the fault edge.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from gridagent.agent_interface.restoration_env import RestorationEpisode
from gridagent.observation import observation_model as om
from gridagent.observation.degradation_registry import tier_params
from gridagent.observation.observation_model import ObservableTruth
from gridagent.simulation.grid_bank import n_independent_loops, stable_seed, _switch_bus_pair

# the coarse feasibility classes simulate_switch may return (NO true energized-gain / violations)
FEASIBILITY = ["likely_ok", "uncertain", "likely_infeasible"]
# the ONLY keys simulate_switch's result may contain (leak-guard whitelist)
SIMULATE_RESULT_KEYS = {
    "feasibility_class", "radiality_legal", "uncertainty_band",
    "reliability", "confidence", "known_limitations", "condition_hint", "evidence_category",
}
# the ONLY keys test_switch's result may contain (leak-guard whitelist)
TEST_SWITCH_RESULT_KEYS = {"outcome"}
TEST_SWITCH_OUTCOMES = {"held", "re_tripped", "uninformative"}

CREW_INSPECT_MIN = 45.0          # crew dispatch + inspect, minutes (charged to the clock)


class RestorationTools:
    def __init__(self, episode: RestorationEpisode, tier: str, seed: int = 0,
                 instrumented_buses: Optional[set] = None):
        self.ep = episode
        self.tier = tier
        self.seed = seed
        self.instrumented = instrumented_buses or set(int(b) for b in episode.net.bus.index)

    # -- shared condition metadata block (mirrors tools.py:_condition_metadata shape) -- #
    def _meta(self) -> dict:
        return {"reliability": self.tier, "confidence": 0.8 if self.tier == "clean" else 0.5,
                "known_limitations": [] if self.tier == "clean" else ["degraded sensing"],
                "condition_hint": self.tier, "evidence_category": "sensing"}

    def _truth(self) -> ObservableTruth:
        """The OBSERVABLE truth the observation model may sense (noisily)."""
        energized = self.ep.energized_buses()
        t = ObservableTruth(
            faulted_bus=int(self.ep.fault.faulted_bus),
            energized_buses=energized,
            instrumented_buses=self.instrumented,
            call_areas={"area": len(self.ep.net.bus) - len(energized)},
        )
        t.faulted_buses = tuple(self.ep.fault.faulted_buses)
        return t

    # ----- OBSERVE (degraded, noisy functions of the OBSERVABLE state) ----- #
    def observe_scada(self) -> dict:
        obs = om.observe(self._truth(), self.tier, self.seed)
        return {"result": {"scada": obs["scada"], "network_model": obs["network_model"]}, **self._meta()}

    def observe_calls(self) -> dict:
        obs = om.observe(self._truth(), self.tier, self.seed)
        return {"result": {"calls": obs["calls"]}, **self._meta()}

    def observe_network_model(self) -> dict:
        return {"result": {"as_built": True, "live_state": None}, **self._meta()}

    # ----- the M3 anti-leak estimate the agent is allowed to see ----- #
    def observed_dark_buses(self) -> set:
        """Buses whose degraded SCADA reports de-energized. A PUBLIC, noisy, sparse channel."""
        try:
            scada = self.observe_scada()["result"].get("scada", {}) or {}
        except Exception:
            return set()
        return {int(b) for b, v in scada.items()
                if isinstance(v, dict) and not v.get("energized", True)}

    def observed_customers_out(self, cust: dict) -> int:
        """The agent-visible customers-out ESTIMATE. Never the hidden truth.

        Two independent public channels are fused the way a control room does it:
          * SCADA — customers on buses REPORTING de-energized, inflated by 1/instrumented_frac
            because only that fraction of buses reports at all;
          * trouble calls — the noisy area count, converted to customers by the same nominal
            per-bus occupancy.
        Both are functions of the observation only, so an agent cannot invert them to the true
        unsupplied set, and two indistinguishable hidden states give the same number.
        """
        p = tier_params(self.tier)
        frac = float(np.clip(p["scada_instrumented_frac"], 1e-3, 1.0))
        dark = self.observed_dark_buses()
        scada_est = sum(int(cust.get(int(b), 0)) for b in dark) / frac
        try:
            calls = self.observe_calls()["result"].get("calls", {}) or {}
            n_call_buses = float(sum(int(v) for v in calls.values()))
        except Exception:
            n_call_buses = 0.0
        per_bus = (float(sum(cust.values())) / max(len(cust), 1)) if cust else 0.0
        calls_est = n_call_buses * per_bus
        if self.tier == "clean":
            return int(round(scada_est))
        return int(round(0.5 * scada_est + 0.5 * calls_est))

    # ----- SENSE (legitimate LOCAL truth at a queried element, at a real time cost) ----- #
    def crew_inspect(self, elem_bus: int) -> dict:
        p = tier_params(self.tier)
        present = int(elem_bus) in set(self.ep.fault.faulted_buses)
        # a crew is accurate but not infallible at the degraded tiers (crew_fi_error)
        rng = np.random.default_rng(
            stable_seed(("crew_inspect", self.seed, self.tier, int(elem_bus)))
        )
        if rng.random() < float(p["crew_fi_error"]):
            present = not present
        self.ep.charge_sensing_time(CREW_INSPECT_MIN)
        return {"result": {"fault_present": bool(present), "element": int(elem_bus)}, **self._meta()}

    def read_fault_indicator(self, bus: int) -> dict:
        obs = om.observe(self._truth(), self.tier, self.seed)
        return {"result": {"fault_indicator": obs["fault_indicator"].get(str(int(bus)))}, **self._meta()}

    # ----- LOCATE probe (M7) ----- #
    def test_switch(self, switch_id: int) -> dict:
        """Open a sectionalizer, re-close the head breaker: 'held' (fault is downstream of the
        opened point) or 're_tripped' (upstream). LEAK-SAFE: the whitelist below is the entire
        payload — the fault edge, the true energized set and the customer counts never appear."""
        r = self.ep.test_switch(int(switch_id))
        out = {"outcome": str(r["outcome"])}
        assert set(out) <= TEST_SWITCH_RESULT_KEYS and out["outcome"] in TEST_SWITCH_OUTCOMES
        return {"result": out, **self._meta()}

    # ----- ACT (gated by the guard inside episode.operate) ----- #
    def operate_switch(self, switch_id: int, closed: bool = True, remote: bool = True,
                       travel_time_min: float = 45.0) -> dict:
        self.ep.n_calls += 1
        r = self.ep.operate(int(switch_id), bool(closed))
        out = {"applied": bool(r.get("applied", True)), "clock_min": r["clock_min"]}
        if r.get("noop"):
            out.update(noop=True, reason=r["reason"])
        if r.get("operation_budget_exhausted"):
            out.update(
                operation_budget_exhausted=True,
                reason=r["reason"],
            )
        if r.get("sectionalizer_open_budget_exhausted"):
            out.update(
                sectionalizer_open_budget_exhausted=True,
                reason=r["reason"],
            )
        return {"result": out, **self._meta()}

    # ----- DECIDE-SUPPORT (leak-safe belief-net what-if) ----- #
    def simulate_switch(self, switch_id: int, belief: Optional[dict] = None) -> dict:
        """BELIEF-net what-if (Scope §16.3). The OBSERVED de-energized buses (degraded SCADA — a
        public, noisy channel) define which sections the belief marks out of service. Closing a tie
        that reconnects an observed-out region WITHOUT forming a loop reads as radiality-legal; a
        tie between two observed-energized buses forms a loop -> likely_infeasible. NEVER reads
        ``self.ep.fault``, the true energized set, or true post-switch violations."""
        observed_out = self.observed_dark_buses()
        trial = copy.deepcopy(self.ep.net)
        trial.line["in_service"] = True  # as-built topology; unknown-state sections filled from obs
        for li in trial.line.index:
            if (int(trial.line.at[li, "from_bus"]) in observed_out
                    or int(trial.line.at[li, "to_bus"]) in observed_out):
                trial.line.at[li, "in_service"] = False
        reconnects = False
        try:
            pair = _switch_bus_pair(trial, int(switch_id))
            reconnects = pair is not None and (pair[0] in observed_out or pair[1] in observed_out)
            trial.switch.at[int(switch_id), "closed"] = True
            radiality_legal = (n_independent_loops(trial) == 0)
        except Exception:
            radiality_legal = False
        band = {"clean": 0.1, "sparse": 0.25, "noisy": 0.4, "stale": 0.4, "conflicting": 0.55}[self.tier]
        if not radiality_legal:
            fc = "likely_infeasible"
        elif reconnects:
            fc = "likely_ok" if self.tier in ("clean", "sparse") else "uncertain"
        else:
            fc = "uncertain"  # legal but no observed restoration benefit
        return {"result": {"feasibility_class": fc, "radiality_legal": bool(radiality_legal),
                           "uncertainty_band": band, **self._meta()}}

    def forecast_load(self, horizon_h: int = 6) -> dict:
        # a public forecast (never a function of the hidden fault/energization)
        return {"result": {"horizon_h": horizon_h, "forecast_kw": 100.0, "p90_kw": 130.0}, **self._meta()}

    # ----- CONTROL-FLOW ----- #
    def escalate_to_human(self) -> dict:
        return {"result": {"escalated": True}, **self._meta()}

    def end_episode(self) -> dict:
        self.ep.done = True
        return {"result": {"ended": True, "clock_min": round(self.ep.clock_min, 1)}, **self._meta()}


# --------------------------------------------------------------------------- #
# public-derived tie ranking, shared by the LLM agent and flisr_operator
# --------------------------------------------------------------------------- #
def public_tie_ranking(net, open_ties, observed_dark: set) -> dict:
    """Hop distance from each open tie's endpoints to the nearest OBSERVED-dark bus.

    Derived from two PUBLIC channels only: the as-built wiring (observe_network_model reports
    as_built=True — a real operator has the single-line diagram) and the agent's own degraded SCADA.
    No hidden state is touched, so this leaks nothing.

    It exists because the boolean "does this tie touch a reported-dark bus" is a false negative for
    EVERY tie whenever SCADA is sparse: on comm_0 at the noisy tier only 29 of 107 buses report, so
    the tie that actually reaches the dead region looked no different from the eight that do not. A
    graded distance degrades gracefully where the boolean collapses. Both the agent and
    flisr_operator consume this same function, so neither is handed a sharper map than the other.
    """
    import networkx as nx
    from gridagent.simulation.grid_bank import _switch_bus_pair
    # as-built graph: every line in service, every switch closed (the wiring, not the live state)
    g = nx.Graph()
    for li in net.line.index:
        g.add_edge(int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"]))
    for _i, tr in net.trafo.iterrows():
        g.add_edge(int(tr["hv_bus"]), int(tr["lv_bus"]))
    dark = {int(b) for b in observed_dark if int(b) in g}
    dist = ({} if not dark else
            nx.multi_source_dijkstra_path_length(g, dark, weight=lambda *_: 1))
    out = {}
    for s in open_ties:
        pair = _switch_bus_pair(net, int(s))
        if pair is None:
            continue
        a, b = int(pair[0]), int(pair[1])
        da, db = dist.get(a), dist.get(b)
        near = [d for d in (da, db) if d is not None]
        out[int(s)] = {"buses": [a, b],
                       "hops_to_nearest_reported_dark_bus": (min(near) if near else None),
                       "one_end_dark_other_live": bool((a in dark) != (b in dark))}
    return out


def public_radiality_legal_ties(net, open_ties) -> list[int]:
    """Return ties whose closure cannot create a loop in public topology.

    This controller-side filter uses only current switch states and the
    published as-built network.  Callers recompute it after every successful
    topology operation because an open can make a previously meshing tie legal.
    """
    import networkx as nx
    from gridagent.simulation.grid_bank import collapsed_graph

    graph = collapsed_graph(net)
    component_by_bus = {
        int(bus): component_id
        for component_id, component in enumerate(nx.connected_components(graph))
        for bus in component
    }
    legal = []
    for switch_id in open_ties:
        pair = _switch_bus_pair(net, int(switch_id))
        if pair is None:
            continue
        first = component_by_bus.get(int(pair[0]))
        second = component_by_bus.get(int(pair[1]))
        if first is None or second is None or first != second:
            legal.append(int(switch_id))
    return legal
