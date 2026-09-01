"""GridAgent-Bench — observation model (static likelihood layer, Scope §16.5, §8.3-§8.5).

Maps the HIDDEN electrical truth to DEGRADED, partial PUBLIC observations across the sensing
channels (sparse SCADA, noisy trouble-calls / AMI last-gasp, sparse/stale fault indicators,
clean as-built network model). The interactive ``RestorationEpisode`` that CONSUMES this is
built later (Section 3); this module is the frozen, hash-pinned likelihood layer the oracle
ceiling (S2.5) is defined against.

ANTI-LEAK CONTRACT (Scope §8.3/§8.5): every observation is a function ONLY of the *observable*
state (fault present at a queried element, energized set, call counts) plus seeded noise — it
NEVER reads a hidden LABEL (latent lambda, oracle sequence, clean weather, generator
coefficients). Therefore no public field is a deterministic function of a hidden label, and two
hidden truths that differ only in a hidden label produce identical public observations.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from gridagent.observation.degradation_registry import TIERS, tier_params
from gridagent.simulation.grid_bank import stable_seed

# fields that must NEVER appear in a public observation (extends the forbidden-key wall)
LATENT_FORBIDDEN = {
    "latent_lambda", "true_vegetation", "true_insulation_state", "true_soil_state",
    "clean_weather", "generator_coefficients", "oracle_switch_sequence",
    "fault_edge_true", "live_energized_buses",
}


@dataclass
class ObservableTruth:
    """The OBSERVABLE part of the hidden state the model is allowed to sense (noisily)."""
    faulted_bus: int
    energized_buses: set          # true energized set (sensed noisily, never returned verbatim)
    instrumented_buses: set       # buses with SCADA
    call_areas: dict              # area -> true #customers-out (sensed noisily)
    faulted_buses: tuple = ()     # concurrent-fault set (M8); falls back to faulted_bus when empty
    # hidden LABELS below are carried for the leak test but MUST NOT influence any observation:
    latent_lambda: float = 0.0
    oracle_switch_sequence: tuple = ()


def instrumented_subset(instrumented_buses, frac: float, tier: str, seed: int) -> list:
    """WHICH buses carry telemetry is a property of the FEEDER, not of the moment.

    This selection is seeded independently of the network state, so the same buses report on every
    turn of an episode. Drawing it from the same generator as the noise (v2 initially did) meant the
    reporting set was resampled whenever a switching action changed the energized set — the agent
    saw a different subset of the grid after every action and could not track its own progress. Only
    the VALUES are noisy; the instrumentation is fixed.
    """
    buses = sorted(int(b) for b in instrumented_buses)
    rng = np.random.default_rng(stable_seed(f"scada_sites_{tier}_{seed}_{len(buses)}_{buses[:1]}"))
    keep = [b for b in buses if rng.random() < float(frac)]
    return keep or buses[: max(1, int(round(float(frac) * len(buses))))]


def _scada_channel(truth: ObservableTruth, params: dict, rng: np.random.Generator,
                   tier: str = "clean", seed: int = 0) -> dict:
    """P/Q/V + breaker status for SCADA-instrumented buses only (controllable fraction)."""
    frac = params["scada_instrumented_frac"]
    keep = instrumented_subset(truth.instrumented_buses, frac, tier, seed)
    out = {}
    for b in keep:
        energized = b in truth.energized_buses
        # noise/staleness may flip the breaker reading
        if rng.random() < params["fi_error"]:
            energized = not energized
        out[str(b)] = {
            "energized": bool(energized),
            "v_pu": round(float(np.clip(rng.normal(1.0 if energized else 0.0, 0.02), 0.0, 1.1)), 3),
            "stale_min": params["latency_min"],
        }
    return out


def _calls_channel(truth: ObservableTruth, params: dict, rng: np.random.Generator) -> dict:
    """Noisy trouble-call / AMI last-gasp counts by area (false+/false-)."""
    out = {}
    for area, n_out in truth.call_areas.items():
        n = int(n_out)
        # false negatives suppress, false positives add
        observed = int(rng.binomial(n, 1 - params["call_false_neg"]))
        observed += int(rng.poisson(params["call_false_pos"] * max(n, 1)))
        out[str(area)] = int(observed)
    return out


def _fi_channel(truth: ObservableTruth, params: dict, rng: np.random.Generator) -> dict:
    """Fault indicators: instant but sparse; can be stale/false (fi_error)."""
    faulted = set(truth.faulted_buses) if truth.faulted_buses else (
        {truth.faulted_bus} if truth.faulted_bus is not None else set())
    out = {}
    for b in sorted(truth.instrumented_buses):
        true_flag = (b in faulted)
        flag = true_flag if rng.random() >= params["fi_error"] else (not true_flag)
        out[str(b)] = bool(flag)
    return out


def observe(truth: ObservableTruth, tier: str, seed: int) -> dict:
    """Produce a degraded PUBLIC observation. Depends ONLY on the observable state + seeded
    noise — never on a hidden label. Returns a dict guaranteed free of forbidden keys."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}")
    params = tier_params(tier)
    # ANTI-LEAK (GOAL_v2 M3): the noise draw is seeded on the OBSERVABLE state only — the energized
    # set and the instrumented set — never on the hidden fault edge. Two hidden truths that a
    # sensor cannot tell apart therefore draw the SAME noise and produce byte-identical channels;
    # seeding on ``faulted_bus`` (v1) made the noise itself a function of hidden state.
    _obs_key = (f"{sorted(int(b) for b in truth.energized_buses)}|"
                f"{sorted(int(b) for b in truth.instrumented_buses)}|"
                f"{sorted((str(k), int(v)) for k, v in truth.call_areas.items())}")
    rng = np.random.default_rng(stable_seed(f"obs_{tier}_{seed}_{_obs_key}"))
    obs = {
        "tier": tier,
        "scada": _scada_channel(truth, params, rng, tier=tier, seed=seed),
        "calls": _calls_channel(truth, params, rng),
        "fault_indicator": _fi_channel(truth, params, rng),
        "network_model": {"as_built": True, "live_state": None},  # clean topology, NO live state
        "reliability": tier,
        "known_limitations": ["sparse SCADA", "noisy calls"] if tier != "clean" else [],
    }
    if params["contradiction_frac"] > 0 and rng.random() < params["contradiction_frac"]:
        # conflicting tier: inject an FI/SCADA disagreement (mutually inconsistent evidence)
        obs["known_limitations"].append("contradictory_evidence")
    assert_no_forbidden(obs)
    return obs


def assert_no_forbidden(obs: dict) -> None:
    """Hard guard: no forbidden (hidden-label) key anywhere in the observation."""
    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in LATENT_FORBIDDEN:
                    raise AssertionError(f"LEAK: forbidden key {k!r} in public observation")
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)
    _walk(obs)


def build_obs_model(grids_dir: str, out_path: str, seed: int) -> dict:
    """Freeze the observation-model config (degradation params + channel defs) and hash-pin it.
    The S2.5 oracle ceiling is defined against this exact hash."""
    from gridagent.observation.degradation_registry import DEGRADATION, CHANNELS, ADMISSIBLE_TIERS
    config = {
        "version": "v1", "seed": seed,
        "tiers": TIERS, "channels": CHANNELS,
        "degradation": DEGRADATION, "admissible_tiers": ADMISSIBLE_TIERS,
        "forbidden_keys": sorted(LATENT_FORBIDDEN),
    }
    blob = json.dumps(config, sort_keys=True)
    config["config_hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(config, fh, indent=2)
    return {"obs_model_path": out_path, "config_hash": config["config_hash"],
            "n_tiers": len(TIERS), "n_channels": len(CHANNELS)}
