"""GridAgent-Bench v2 — the REAL LLM ReAct restoration loop.  [Scope §16, §18; GOAL_v2 M1/M3/M7]

The agent drives the REAL ``RestorationEpisode`` through the leak-safe tools and is scored by the
SAME ``score_episode`` every baseline uses, so its regret is comparable by construction and the
oracle cannot be beaten by a metric artefact.

What v2 changed
---------------
* **No partial-observability leak (M3).** v1 put ``len(unsupplied(true_net))`` in the prompt every
  turn and reported the TRUE delta after each action. The agent was therefore told the exact hidden
  state it was supposed to be inferring. v2 shows ``customers_out_estimate`` from
  ``RestorationTools.observed_customers_out`` — SCADA-reported dark buses inflated by the
  instrumented fraction, fused with the noisy call count — and every action result is phrased in
  those observed terms.
* **Post-isolation model scope.** The generic environment retains inspection and protection-probe
  tools, but the reported model arm begins after protection-assisted isolation and controls service
  restoration. Sectionalizers may be opened to shed a dark pocket a tie cannot carry; ties are
  closed to back-feed.
* **Attempts cost time.** A guard-rejected close is not free any more — the crew still drove out.
  The prompt says so, because it is now true.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from gridagent.agent_interface.baselines.restoration_policies import score_episode
from gridagent.agent_interface.cost_model import DEFAULT_HORIZON_MIN, switch_time_min
from gridagent.agent_interface.restoration_env import (
    ControllerActionViolation, EnergizationViolation, Fault,
    LiveSectionalizerOpenViolation, MAX_TOTAL_CALLS, PowerFlowViolation,
    RestorationEpisode,
)
from gridagent.agent_interface.restoration_tools import (
    RestorationTools, public_radiality_legal_ties, public_tie_ranking,
)
from gridagent.simulation.grid_bank import RadialityViolation

ALL_VERBS = {"observe_scada", "observe_calls", "crew_inspect", "read_fault_indicator",
             "test_switch", "simulate_switch", "operate_switch", "open_switch", "end_episode"}

SYS_PROMPT = (
    "You are a distribution-grid control-room operator running FLISR restoration after a fault has "
    "de-energized part of a medium-voltage feeder. GOAL: restore power to as many customers as "
    "possible, as FAST as possible — every minute a customer is out counts against you.\n"
    "\nWHAT YOU CANNOT SEE: the faulted section is hidden. Your instruments are degraded, so "
    "customers_out_estimate is an ESTIMATE, not the truth.\n"
    "\nTHE FAULT IS ALREADY ISOLATED. Protection tripped and the sectionalizers around the faulted "
    "section have already opened before you take over. You do NOT need to isolate anything. Your "
    "one job is to get the healthy but de-energized sections back on supply.\n"
    "\nHOW RESTORATION WORKS. The de-energized sections are electrically fine — they are dark only "
    "because their path to the source runs through the faulted section. A normally-open TIE switch "
    "connects them to a NEIGHBOURING feeder. CLOSING a tie back-feeds them and restores customers. "
    "The ties available to you are listed each turn as ties_you_can_CLOSE_to_restore; they are "
    "currently OPEN and already screened against loop creation using the public network model, "
    "so the action you want is operate_switch, which closes them. Closing a tie is "
    "the ONLY action that restores anybody.\n"
    "\nPICK THE TIE THAT REACHES THE DARK REGION. Each tie is listed with the two buses it "
    "connects. A tie only helps if ONE of its buses is in "
    "the de-energized region and the OTHER is still live — that is what back-feeding means. Closing "
    "a tie whose two ends are both already live just creates a loop and is rejected, wasting the "
    "crew trip. The tie list is SORTED for you by hops_to_nearest_reported_dark_bus — how far that "
    "tie sits, over the as-built wiring, from the nearest bus your SCADA reports dark. Low hops "
    "means the tie probably reaches the dead region, so work DOWN THE LIST FROM THE TOP. The "
    "one_end_dark_other_live flag is the ideal signal, but your SCADA is sparse and it is often "
    "false even for the right tie — trust the hop count when the flag is not set.\n"
    "\nWHEN A TIE IS REFUSED FOR OVERLOAD. The neighbouring feeder may not have the capacity to "
    "carry the whole dead section. Then, and only then, open one sectionalizer from "
    "sectionalizers_you_can_OPEN_to_shed_load to drop the far end, and retry the tie. Restoring "
    "most customers beats restoring none. Opening switches never restores anyone by itself.\n"
    "\nSENSING. `read_fault_indicator` is instant but can be wrong. `test_switch` on a CLOSED "
    "sectionalizer opens it and recloses the head breaker: 'held' means the fault is beyond that "
    "switch, 're_tripped' means it is between the source and that switch. `crew_inspect` checks one "
    "bus directly and is reliable. Sensing does not restore anyone.\n"
    "\nTHE CLOCK IS THE SCORE. You are scored on customer-minutes lost over a 240-minute horizon, "
    "so EVERY action you take delays restoration for everyone still out. Approximate costs:\n"
    "  read_fault_indicator / observe_*  ~0 min (free, but noisy)\n"
    "  simulate_switch                   ~0 min (free belief check)\n"
    "  test_switch                       ~12 min\n"
    "  crew_inspect                      ~45 min\n"
    "  operating a REMOTE switch         ~1 min;  a MANUAL one 30-90 min (crew drives out)\n"
    "A customer restored at minute 200 is worth almost nothing; restored at minute 20 is worth "
    "nearly everything. Sense only as much as you need, then ACT.\n"
    "\nATTEMPTS COST TIME EVEN WHEN REJECTED. A close is auto-rejected if it would energize the "
    "faulted section, create a loop, or OVERLOAD the neighbouring feeder. The network is unharmed, "
    "but the crew still spent the trip. Use the observations rather than brute-forcing every tie. "
    "Prefer a remote tie WHEN IT REACHES THE DEAD SECTION — speed never beats reaching the right "
    "place. A slow manual tie that restores 40 customers is worth far more than a fast remote one "
    "that restores nobody, so work through the whole tie list, not only the remote ones.\n"
    "\nDO NOT OPEN A SWITCH THAT IS STILL FEEDING CUSTOMERS. open_switch is for the DE-ENERGIZED "
    "region only (to bound the fault or shed a pocket a tie cannot carry). Opening a live switch "
    "blacks out healthy customers and makes your score WORSE than doing nothing.\n"
    "\nACTION IDS. Copy switch_id exactly from the candidate list for the action you choose. Values "
    "inside connects_buses are BUS numbers, not switch ids, and must never be placed in switch_id. "
    "Use operate_switch only with ties_you_can_CLOSE_to_restore and open_switch only with "
    "sectionalizers_you_can_OPEN_to_shed_load. A rejected switch is removed from its candidate "
    "list until another operation changes the topology; do not retry an id that is no longer "
    "listed.\n"
    "\nReply with EXACTLY ONE JSON object per turn and NOTHING else:\n"
    '  {"verb":"test_switch","switch_id":<id>}       probe: held / re_tripped\n'
    '  {"verb":"crew_inspect","bus":<id>}            check one bus (slow, reliable)\n'
    '  {"verb":"read_fault_indicator","bus":<id>}    instant, noisy\n'
    '  {"verb":"simulate_switch","switch_id":<id>}   belief-based feasibility check (free)\n'
    '  {"verb":"open_switch","switch_id":<id>}       OPEN a closed sectionalizer to shed load\n'
    '  {"verb":"operate_switch","switch_id":<id>}    CLOSE a tie to back-feed — THE ONLY ACTION\n'
    '                                                THAT RESTORES CUSTOMERS\n'
    '  {"verb":"end_episode"}                        stop when nothing more can be restored'
)

MINIMAL_BRIEFING = (
    "You are operating a partially observed medium-voltage restoration episode. "
    "The fault is already isolated. Restore healthy dark sections quickly by closing "
    "a listed normally-open tie. Every switching or sensing minute increases outage "
    "cost. Candidate switch IDs, their public as-built bus pairs, and degraded "
    "observations are supplied each turn. All operations pass through the same "
    "radiality, fault-energization, voltage, and thermal guard. You may gather public "
    "evidence, simulate a listed tie, close a listed tie with operate_switch, open a "
    "listed dark-region sectionalizer, or end the episode. Reply with exactly one JSON "
    "action object using the supplied switch or bus IDs and no additional prose."
)

SCHEMA_ONLY_BRIEFING = (
    "You control a post-isolation medium-voltage service-restoration episode. "
    "Use only information in the supplied observation. All switching actions pass through "
    "the same radiality, fault-energization, voltage, thermal, and power-flow guard. "
    "Every nonzero-duration action increases outage cost. Reply with exactly one JSON object "
    "and no additional prose. The allowed action schemas are: "
    '{"verb":"observe_scada"}; '
    '{"verb":"observe_calls"}; '
    '{"verb":"crew_inspect","bus":<id>}; '
    '{"verb":"read_fault_indicator","bus":<id>}; '
    '{"verb":"test_switch","switch_id":<id>}; '
    '{"verb":"simulate_switch","switch_id":<id>}; '
    '{"verb":"operate_switch","switch_id":<id>}; '
    '{"verb":"open_switch","switch_id":<id>}; '
    '{"verb":"end_episode"}. '
    "Use only bus and switch identifiers present in the observation."
)


def _pair(ep, sid) -> tuple:
    """The two buses a switch bridges — public as-built topology, never hidden state."""
    from gridagent.simulation.grid_bank import _switch_bus_pair
    p = _switch_bus_pair(ep.net, int(sid))
    return (int(p[0]), int(p[1])) if p else (-1, -1)


PARSER_STATUSES = (
    "valid_action",
    "malformed_json",
    "wrong_schema",
    "unknown_switch_id",
    "wrong_operation_class",
    "duplicate/repeated_action",
    "unavailable_verb",
)


def _parse_action_detailed(text: str) -> tuple[dict | None, str, str | None]:
    """Parse one model reply without repairing it or silently dropping duplicates.

    The public runner permits harmless surrounding prose/code fences for diagnostics, but it
    still requires exactly one flat JSON action object.  Every failure is returned as a named
    status so production metrics and transcript audits can account for it without exceptions.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "malformed_json", "empty_reply"
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    if len(matches) != 1:
        return None, "malformed_json", "expected_one_json_object"

    duplicate = False

    def pairs_hook(pairs):
        nonlocal duplicate
        keys = [key for key, _ in pairs]
        duplicate = len(keys) != len(set(keys))
        return dict(pairs)

    try:
        obj = json.loads(matches[0], object_pairs_hook=pairs_hook)
    except Exception as exc:
        return None, "malformed_json", type(exc).__name__
    if duplicate:
        return None, "malformed_json", "duplicate_json_key"
    if not isinstance(obj, dict) or "verb" not in obj:
        return None, "wrong_schema", "missing_verb_or_non_object"
    if not isinstance(obj.get("verb"), str):
        return None, "wrong_schema", "verb_not_string"
    verb = obj["verb"]
    if verb in {"operate_switch", "open_switch", "simulate_switch", "test_switch"}:
        if "switch_id" not in obj or isinstance(obj.get("switch_id"), bool):
            return None, "wrong_schema", "missing_or_boolean_switch_id"
    if verb in {"crew_inspect", "read_fault_indicator"}:
        if "bus" not in obj or isinstance(obj.get("bus"), bool):
            return None, "wrong_schema", "missing_or_boolean_bus"
    return obj, "valid_action", None


def _parse_action(text: str):
    """Compatibility wrapper used by older callers."""
    obj, _, _ = _parse_action_detailed(text)
    return obj


def _action_int(action: dict, key: str, default: int) -> int:
    """Read an integer action field without trusting model-emitted JSON types."""
    value = action.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def classify_switch_target(
    sid: int,
    want_closed: bool,
    valid_close_ids,
    valid_open_ids,
    known_switch_ids=None,
) -> str:
    """Classify an operation target without conflating an unknown id and a wrong operation."""
    valid = set(valid_close_ids) if want_closed else set(valid_open_ids)
    if sid in valid:
        return "valid"
    known = set(known_switch_ids or (set(valid_close_ids) | set(valid_open_ids)))
    if sid in known:
        return "wrong_candidate_operation"
    return "unknown_switch_id"


def action_repeat_key(verb: str, target, topology_revision: int):
    """Switch feasibility changes after a successful topology operation."""
    try:
        hash(target)
    except TypeError:
        target = json.dumps(target, sort_keys=True, default=str)
    if verb in {"simulate_switch", "operate_switch", "open_switch"}:
        return (topology_revision, verb, target)
    return (verb, target)


def run_llm_episode(llm, net, fault: Fault, label: dict, cust: dict, tier: str, *,
                    sw_meta: dict | None = None, max_turns: int = 14, seed: int = 0,
                    rank_ties: bool = True,
                    true_state: bool = False,
                    briefing: str = "full",
                    use_memory: bool = True, allowed_verbs: set | None = None,
                    stall_limit: int = 6, weights: dict | None = None,
                    horizon_min: float = DEFAULT_HORIZON_MIN,
                    capture_trace: list | None = None) -> dict:
    """Real ReAct restoration episode. ``llm`` must expose chat_messages(list)->str (or chat) and
    the attribute last_completion_tokens. Returns a physics-derived record (provenance='rollout')."""
    briefings = {
        "full": SYS_PROMPT,
        "minimal": MINIMAL_BRIEFING,
        "schema_only": SCHEMA_ONLY_BRIEFING,
    }
    if briefing not in briefings:
        raise ValueError("briefing must be 'full', 'minimal', or 'schema_only'")
    system_prompt = briefings[briefing]
    sw_meta = sw_meta or {}
    tie_ids = {s for s, m in sw_meta.items() if m.get("type") == "tie"}
    sec_ids = {s for s, m in sw_meta.items() if m.get("type") in ("sectionalizer", "RMU/LBS")}
    allowed = allowed_verbs or ALL_VERBS
    ep = RestorationEpisode(net, fault, sw_meta=sw_meta, horizon_min=horizon_min)
    tools = RestorationTools(ep, tier, seed=seed)

    open_ties = sorted(int(s) for s in ep.net.switch.index[~ep.net.switch.closed]
                       if (not tie_ids) or int(s) in tie_ids)
    closed_secs = sorted(int(s) for s in ep.net.switch.index[ep.net.switch.closed]
                         if int(s) in sec_ids)
    known_switch_ids = set(open_ties) | set(closed_secs)

    messages = [{"role": "system", "content": system_prompt}]
    tokens_completion = tool_calls = simulate_switch_calls = invalid_action = 0
    malformed_reply = unknown_switch_id = wrong_candidate_operation = 0
    parser_status_counts = {status: 0 for status in PARSER_STATUSES}
    sense_calls = 0
    invalid_target_events = []
    operated = set()
    prior_actions: dict = {}
    topology_revision = 0
    repeats = 0
    stalled = 0
    last_result = "none yet"
    overload_shedding_authorized = False
    t0 = time.time()

    def est_out() -> int:
        return tools.observed_customers_out(cust)

    for turn in range(max_turns):
        if (
            ep.done
            or ep.n_calls >= MAX_TOTAL_CALLS
            or ep.switch_ops_remaining <= 0
        ):
            break
        # ---- OBSERVATION: public, degraded channels ONLY (no hidden state, M3) ----
        try:
            scada = tools.observe_scada()["result"].get("scada", {}) or {}
        except Exception:
            scada = {}
        dark = sorted(int(b) for b, v in scada.items()
                      if isinstance(v, dict) and not v.get("energized", True))
        current_open_ties = [
            s for s in open_ties
            if not bool(ep.net.switch.at[s, "closed"])
            and action_repeat_key("operate_switch", s, topology_revision) not in prior_actions
        ]
        radiality_legal_ties = public_radiality_legal_ties(
            ep.net, current_open_ties
        )
        current_closed_secs = [
            s for s in closed_secs
            if bool(ep.net.switch.at[s, "closed"])
            and action_repeat_key("open_switch", s, topology_revision) not in prior_actions
        ]
        displayed_open_secs = (
            [
                s for s in current_closed_secs
                if all(int(b) in set(dark) for b in _pair(ep, s))
            ][:15]
            if (
                ep.sectionalizer_opens_remaining > 0
                and (radiality_legal_ties or overload_shedding_authorized)
            )
            else []
        )
        # Sensing cannot create a restoration action. Once the public as-built
        # radiality screen has no closeable tie, only a real prior overload can
        # authorize the load-shedding operation described in the briefing.
        if not radiality_legal_ties and not overload_shedding_authorized:
            break
        # AB6: withhold the public hop ranking to measure how much of the fair-loop improvement
        # came from that hint rather than from the eight bug fixes. The agent still gets the tie
        # bus pairs (as-built topology is public); only the derived ordering is removed.
        rank = public_tie_ranking(
            ep.net, radiality_legal_ties, set(dark)
        ) if rank_ties else {}
        obs = {
            "tier": tier,
            "stage": ep.stage,
            "clock_min": round(ep.clock_min, 1),
            "customers_out_estimate": est_out(),
            "minutes_left_in_horizon": round(max(horizon_min - ep.clock_min, 0.0), 1),
            "scada_reported_dark_buses": dark[:40],
            "n_scada_reporting_buses": len(scada),
            "n_scada_reported_dark": len(dark),
            # The AS-BUILT wiring is a PUBLIC channel (observe_network_model: as_built=True) — a real
            # operator has the single-line diagram on the wall, and flisr_operator already reads
            # these bus pairs to rank its candidates. Withholding them from the agent was not
            # partial observability, it was withholding the map, and it made the incumbent
            # comparison unfair: the agent was guessing which tie even touches the dark region.
            "ties_you_can_CLOSE_to_restore": [
                {"switch_id": s, "connects_buses": rank.get(s, {}).get("buses", list(_pair(ep, s))),
                 "minutes": round(switch_time_min(s, sw_meta), 1),
                 "hops_to_nearest_reported_dark_bus":
                     rank.get(s, {}).get("hops_to_nearest_reported_dark_bus"),
                 "one_end_dark_other_live": rank.get(s, {}).get("one_end_dark_other_live", False)}
                for s in (sorted(radiality_legal_ties, key=lambda s: (
                    rank.get(s, {}).get("hops_to_nearest_reported_dark_bus") if
                    rank.get(s, {}).get("hops_to_nearest_reported_dark_bus") is not None else 999,
                    switch_time_min(s, sw_meta))) if rank_ties else radiality_legal_ties)],
            "sectionalizers_you_can_OPEN_to_shed_load": [
                {"switch_id": s, "connects_buses": list(_pair(ep, s))}
                for s in displayed_open_secs],
            "switches_operated_so_far": sorted(operated),
            "switch_operations_remaining": ep.switch_ops_remaining,
            "sectionalizer_opens_remaining": (
                ep.sectionalizer_opens_remaining
            ),
            "last_action_result": last_result,
            "turns_left": max_turns - turn,
        }
        if true_state and turn == 0:
            obs["true_state_at_episode_start"] = {
                "faulted_buses": sorted(int(bus) for bus in fault.faulted_buses),
                "switch_states": {
                    str(int(switch_id)): (
                        "closed"
                        if bool(ep.net.switch.at[switch_id, "closed"])
                        else "open"
                    )
                    for switch_id in ep.net.switch.index
                },
            }
        user = "Observation:\n" + json.dumps(obs, default=str) + "\nRespond with ONE JSON action."
        try:
            if use_memory and hasattr(llm, "chat_messages"):
                messages.append({"role": "user", "content": user})
                text = llm.chat_messages(messages)
                messages.append({"role": "assistant", "content": text})
            else:
                text = llm.chat(system_prompt, user)
        except Exception:
            text = ""
        tokens_completion += int(getattr(llm, "last_completion_tokens", 0) or 0)

        trace_row = None
        if capture_trace is not None:
            trace_row = {"turn": turn,
                         "observation": json.loads(json.dumps(obs, default=str)),
                         "public_observation": json.loads(json.dumps(obs, default=str)),
                         "raw_reply": text, "result_of_previous": last_result}
            capture_trace.append(trace_row)

        def mark_parser(status: str, error: str | None = None, parsed=None) -> None:
            parser_status_counts[status] += 1
            if trace_row is not None:
                trace_row["parser_status"] = status
                trace_row["parser_error"] = error
                trace_row["parsed_action"] = parsed

        act, parse_status, parse_error = _parse_action_detailed(text)
        if act is None:
            mark_parser(parse_status, parse_error)
            invalid_action += 1
            malformed_reply += int(parse_status == "malformed_json")
            last_result = "your reply was not a single valid JSON action; reply with ONE JSON object"
            continue
        verb = act.get("verb")
        if not isinstance(verb, str):
            mark_parser("wrong_schema", "verb_not_string", act)
            invalid_action += 1
            last_result = "verb must be a string naming one available action"
            continue
        if verb == "end_episode":
            mark_parser("valid_action", None, act)
            break
        if verb not in allowed:
            mark_parser("unavailable_verb", None, act)
            invalid_action += 1
            last_result = f"verb '{verb}' is not available; use one of {sorted(allowed)}"
            continue
        tool_calls += 1
        before_est = est_out()
        target = act.get("switch_id", act.get("bus"))
        akey = action_repeat_key(verb, target, topology_revision)
        if akey in prior_actions:
            mark_parser("duplicate/repeated_action", None, act)
            repeats += 1
            last_result = (f"You already did {verb} on {target} without any intervening topology "
                           f"change and the result was: "
                           f"{prior_actions[akey]}. Repeating it changes nothing. Try a DIFFERENT "
                           f"switch, or CLOSE a tie from ties_you_can_CLOSE_to_restore.")
            if repeats >= 3:
                break
            continue

        if verb == "observe_scada":
            sense_calls += 1
            last_result = f"scada re-read: {len(tools.observed_dark_buses())} buses report dark"
        elif verb == "observe_calls":
            sense_calls += 1
            try:
                tools.observe_calls()
            except Exception:
                pass
            last_result = "customer calls re-read"
        elif verb == "crew_inspect":
            sense_calls += 1
            b = _action_int(act, "bus", -1)
            try:
                r = tools.crew_inspect(b)
                last_result = (f"crew_inspect bus {b}: fault_present="
                               f"{r['result']['fault_present']} (cost 45 min)")
            except Exception:
                last_result = f"crew_inspect bus {b} failed"
        elif verb == "read_fault_indicator":
            sense_calls += 1
            b = _action_int(act, "bus", -1)
            try:
                r = tools.read_fault_indicator(b)
                last_result = f"fault_indicator at bus {b}: {r['result']['fault_indicator']}"
            except Exception:
                last_result = "read_fault_indicator failed"
        elif verb == "test_switch":
            sense_calls += 1
            sid = _action_int(
                act, "switch_id", closed_secs[0] if closed_secs else 0
            )
            try:
                r = tools.test_switch(sid)
                last_result = (f"test_switch {sid}: {r['result']['outcome']} "
                               f"({'fault is downstream of it' if r['result']['outcome'] == 'held' else 'fault is upstream of it'})")
            except Exception:
                last_result = f"test_switch {sid} failed"
        elif verb == "simulate_switch":
            simulate_switch_calls += 1
            sid = _action_int(
                act,
                "switch_id",
                radiality_legal_ties[0] if radiality_legal_ties else 0,
            )
            try:
                r = tools.simulate_switch(sid)
                last_result = f"simulate {sid}: feasibility={r['result']['feasibility_class']}"
            except Exception:
                last_result = f"simulate {sid} failed"
        elif verb in ("operate_switch", "open_switch"):
            sid = _action_int(
                act,
                "switch_id",
                radiality_legal_ties[0] if radiality_legal_ties else 0,
            )
            want_closed = (verb == "operate_switch") and bool(act.get("closed", True))
            valid = (
                set(radiality_legal_ties)
                if want_closed
                else set(displayed_open_secs)
            )
            if sid not in valid:
                invalid_action += 1
                category = classify_switch_target(
                    sid, want_closed, radiality_legal_ties,
                    displayed_open_secs, known_switch_ids)
                unknown_switch_id += int(category == "unknown_switch_id")
                wrong_candidate_operation += int(category == "wrong_candidate_operation")
                mark_parser(
                    "unknown_switch_id"
                    if category == "unknown_switch_id"
                    else "wrong_operation_class",
                    category,
                    act,
                )
                invalid_target_events.append({
                    "turn": turn, "verb": verb, "switch_id": sid, "category": category,
                    "valid_close_ids": list(radiality_legal_ties),
                    "valid_open_ids": list(displayed_open_secs),
                })
                if category == "unknown_switch_id":
                    last_result = (
                        f"switch {sid} is not present in either candidate list. Copy a switch_id "
                        "exactly; do not use a bus number.")
                else:
                    last_result = (
                        f"switch {sid} is a known candidate but cannot be "
                        f"{'CLOSED' if want_closed else 'OPENED'} with this action. Choose an id "
                        f"from {'ties_you_can_CLOSE_to_restore' if want_closed else 'sectionalizers_you_can_OPEN_to_shed_load'}.")
                prior_actions[akey] = last_result[:90]
                continue
            if (sid, want_closed) in operated:
                mark_parser("duplicate/repeated_action", None, act)
                last_result = f"switch {sid} is already in that position; try another action"
                continue
            mark_parser("valid_action", None, act)
            try:
                res = tools.operate_switch(sid, closed=want_closed)["result"]
                if res.get("operation_budget_exhausted"):
                    last_result = res["reason"]
                    break
                if res.get("sectionalizer_open_budget_exhausted"):
                    last_result = res["reason"]
                    continue
                if res.get("noop"):
                    # Tell the truth. Reporting a refused no-op as a success (the v2 scaffold's
                    # first attempt) left the agent repeating it — three times in one transcript.
                    stalled += 1
                    last_result = (
                        f"NO CHANGE — {res['reason']}. Nothing happened and no time was spent. "
                        + ("The ties listed in ties_you_can_close are ALREADY OPEN; to back-feed "
                           "you must CLOSE one with operate_switch."
                           if not want_closed else
                           "Pick a switch that is currently in the other position."))
                    continue
                operated.add((sid, want_closed))
                topology_revision += 1
                overload_shedding_authorized = False
                after = est_out()
                gain = before_est - after
                last_result = (f"{'closed' if want_closed else 'opened'} switch {sid}: "
                               f"customers_out_estimate {before_est} -> {after}"
                               f" (clock {ep.clock_min:.0f} min)")
                stalled = 0 if gain > 0 else stalled + 1
            except RadialityViolation:
                stalled += 1
                last_result = (f"tie {sid} REJECTED (would create a loop); the trip still cost "
                               f"time, clock is now {ep.clock_min:.0f} min")
            except EnergizationViolation:
                stalled += 1
                last_result = (f"switch {sid} REJECTED (would energize the faulted section — the "
                               f"fault is on that side); clock is now {ep.clock_min:.0f} min")
            except PowerFlowViolation:
                stalled += 1
                overload_shedding_authorized = True
                last_result = (f"tie {sid} REJECTED (back-feed would OVERLOAD the neighbouring "
                               f"feeder); shed part of the dead section with open_switch, then "
                               f"retry; clock is now {ep.clock_min:.0f} min")
            except LiveSectionalizerOpenViolation:
                stalled += 1
                last_result = (
                    f"switch {sid} REJECTED (the no-reinterruption guard permits "
                    "only a true-dark sectionalizing cut); "
                    f"clock is now {ep.clock_min:.0f} min"
                )
            except ControllerActionViolation:
                stalled += 1
                last_result = (
                    f"switch {sid} REJECTED (outside the declared restoration "
                    f"controller action class); clock is now {ep.clock_min:.0f} min"
                )
            except Exception:
                last_result = f"switch {sid} could not be operated"

        if trace_row is not None and "parser_status" not in trace_row:
            mark_parser("valid_action", None, act)
        if trace_row is not None:
            trace_row["action_result"] = last_result
        prior_actions[akey] = last_result[:90]
        if stalled >= stall_limit:
            break

    rec = score_episode(ep, label, cust, weights=weights, horizon_min=horizon_min)
    rec.update({
        "arm": "llm",
        "invalid_action": int(invalid_action),
        "malformed_reply": int(malformed_reply),
        # ``bad_target`` is retained as the Phase-A field and now means exactly what its detector
        # says: an id absent from every candidate list. Wrong operation/list choices are separate.
        "bad_target": int(unknown_switch_id),
        "unknown_switch_id": int(unknown_switch_id),
        "wrong_candidate_operation": int(wrong_candidate_operation),
        "parser_status_counts": parser_status_counts,
        "invalid_target_events": invalid_target_events,
        "repeated_actions": int(repeats),
        "n_noop_operations": int(ep.n_noop_operations),
        "tool_calls": int(tool_calls),
        "sense_calls": int(sense_calls),
        "simulate_switch_calls": int(simulate_switch_calls),
        "tokens_prompt": 0,
        "tokens_completion": int(tokens_completion),
        "turns_used": int(turn + 1),
        "wall_clock_s": round(time.time() - t0, 3),
        "safety_violation": int(rec.get("flisr_hardfail", 0)),
        "regret_obs": rec.get("regret_omni"),
        "true_state_ablation": bool(true_state),
        "briefing": briefing,
        "rank_ties": bool(rank_ties),
        "last_action_result": last_result,
    })
    if capture_trace is not None:
        for i, row in enumerate(capture_trace):
            if "action_result" not in row:
                row["action_result"] = (
                    capture_trace[i + 1].get("result_of_previous", last_result)
                    if i + 1 < len(capture_trace) else last_result
                )
    return rec
