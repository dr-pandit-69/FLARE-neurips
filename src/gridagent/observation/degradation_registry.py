"""GridAgent-Bench — sensing-channel degradation registry (Scope §16.5, §15).

The NEW degradation registry for the FLISR sensing channels (SCADA / trouble-calls /
fault-indicators / crew), mapped onto the frozen five-tier observability ladder
    clean -> sparse -> noisy -> stale -> conflicting
(Scope §15). This is NOT the legacy ``enrichment/tool_conditions.CONDITION_REGISTRY``
(which degrades only the archived single-shot tools) — it degrades the new channels.

Each tier defines, per channel, the degradation parameters (instrumented fraction, false+/-
rates, latency/staleness, contradiction injection). Monotone: information decreases along
the ladder; ``conflicting`` additionally injects mutually-contradictory evidence.
"""
from __future__ import annotations

TIERS = ["clean", "sparse", "noisy", "stale", "conflicting"]

# per-tier, per-channel degradation parameters.
#   scada_instrumented_frac : fraction of buses with live SCADA (P/Q/V + breaker status)
#   call_false_pos / call_false_neg : trouble-call / AMI last-gasp error rates
#   fi_error : fault-indicator false/stale probability
#   crew_fi_error : crew ground-truth inspection error
#   latency_min : staleness of the channel (minutes behind truth)
#   contradiction_frac : fraction of sections given mutually-inconsistent evidence
DEGRADATION = {
    "clean":       {"scada_instrumented_frac": 1.00, "call_false_pos": 0.00, "call_false_neg": 0.00,
                    "fi_error": 0.00, "crew_fi_error": 0.00, "latency_min": 0.0,  "contradiction_frac": 0.00},
    "sparse":      {"scada_instrumented_frac": 0.35, "call_false_pos": 0.05, "call_false_neg": 0.10,
                    "fi_error": 0.05, "crew_fi_error": 0.02, "latency_min": 5.0,  "contradiction_frac": 0.00},
    "noisy":       {"scada_instrumented_frac": 0.35, "call_false_pos": 0.20, "call_false_neg": 0.25,
                    "fi_error": 0.20, "crew_fi_error": 0.05, "latency_min": 10.0, "contradiction_frac": 0.00},
    "stale":       {"scada_instrumented_frac": 0.35, "call_false_pos": 0.20, "call_false_neg": 0.25,
                    "fi_error": 0.20, "crew_fi_error": 0.05, "latency_min": 45.0, "contradiction_frac": 0.00},
    "conflicting": {"scada_instrumented_frac": 0.25, "call_false_pos": 0.30, "call_false_neg": 0.30,
                    "fi_error": 0.30, "crew_fi_error": 0.08, "latency_min": 30.0, "contradiction_frac": 0.35},
}

# which channels each tier degrades (for the family x tier constructibility check)
CHANNELS = ["scada", "calls", "fault_indicator", "crew_inspect", "network_model"]

# admissible tiers per event family (all families admit all tiers in the FLISR toolset; the
# legacy per-family CONDITION_SUITABILITY restriction does NOT apply to the new channels).
EVENT_FAMILIES = ["single_section", "multi_section", "storm_cluster", "der_islanding", "ambiguous"]
ADMISSIBLE_TIERS = {fam: list(TIERS) for fam in EVENT_FAMILIES}


def tier_params(tier: str) -> dict:
    if tier not in DEGRADATION:
        raise KeyError(f"unknown tier {tier!r}; valid: {TIERS}")
    return dict(DEGRADATION[tier])


def is_constructible(family: str, tier: str) -> bool:
    return tier in ADMISSIBLE_TIERS.get(family, [])


def information_rank(tier: str) -> int:
    """Monotone information index (clean=0 most-informative ... conflicting=4 least)."""
    return TIERS.index(tier)
