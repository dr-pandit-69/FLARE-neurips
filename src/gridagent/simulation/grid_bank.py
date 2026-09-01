"""GridAgent-Bench — switchable German-MV feeder bank (Scope §4, pipeline S1-S3).

Builds the 5-wire-topology / 13-instance CIGRE + SimBench-MV bank, normalizes each
instance to a RADIAL normal config (the ring operated open at one point), tags
switchgear (type / remote / travel_time_min / section endpoints / earthing regime),
and emits the fleet ``switch_manifest.csv`` + ``section_manifest.csv`` — the ~466
restorable sections that the oracle / ENS consume.

Radiality is always checked on a *collapsed simple* ``networkx.Graph`` (MultiGraph
edges merged) so that parallel transformers / parallel lines never spoof a cycle —
the verified P3 landmine (Scope §3 S3).

Reused unchanged (Scope §4.5): the pandapower / simbench engine,
``create_cigre_network_mv``, ``get_simbench_net``, and ``stable_seed`` (mirrored
below to avoid importing the heavy legacy outage module). DISCARDS
``phase2_topology.SIMBENCH_CODE_MAP`` (the collapse-to-6-LV-labels anti-pattern).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as ppn
import pandapower.topology as top
import simbench as sb


# --------------------------------------------------------------------------- #
# reuse: stable_seed (events/run_outages.py:214) — mirrored, identical semantics
# --------------------------------------------------------------------------- #
def stable_seed(scenario_id) -> int:
    digest = hashlib.blake2b(str(scenario_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2 ** 32)


# --------------------------------------------------------------------------- #
# bank definition (Scope §4.1): 5 wire-topologies -> 13 grid-instances
# --------------------------------------------------------------------------- #
SIMBENCH_ARCHES = {
    "rural": "1-MV-rural",
    "semiurb": "1-MV-semiurb",
    "urban": "1-MV-urban",
    "comm": "1-MV-comm",
}
DER_SCENARIOS = ["0", "1", "2"]  # 2016 / 2024 / 2034 DER penetration — SAME wires
FEEDER_RE = re.compile(r"_Feeder\d+$")
EARTHING_REGIMES = ["resonant/compensated", "short-time low-ohmic", "low-ohmic"]
# German-MV default earthing mix (Scope §6.2 / §12.2; documented default, refined in S2.3).
EARTHING_MIX = np.array([0.60, 0.25, 0.15])
SWITCH_TYPES = ["sectionalizer", "tie", "breaker", "RMU/LBS"]


class RadialityViolation(Exception):
    """Raised when a net's normal config is not radial, or a tie closure does not
    create exactly one loop (mirrors the Part-1 P3 / Part-2 M1 guard)."""


def bank_grid_ids() -> list[str]:
    ids = ["cigre_mv"]
    for arch in SIMBENCH_ARCHES:
        for s in DER_SCENARIOS:
            ids.append(f"{arch}_{s}")
    return ids  # 1 + 4*3 = 13


def archetype_of(grid_id: str) -> str:
    return "cigre" if grid_id == "cigre_mv" else grid_id.rsplit("_", 1)[0]


def representative_grid_id(archetype: str) -> str:
    """The --0-sw (or CIGRE) instance that anchors archetype-level sections/feeders."""
    return "cigre_mv" if archetype == "cigre" else f"{archetype}_0"


def instantiate_grid(grid_id: str):
    if grid_id == "cigre_mv":
        net = ppn.create_cigre_network_mv(with_der="all")
    else:
        arch, s = grid_id.rsplit("_", 1)
        net = sb.get_simbench_net(f"{SIMBENCH_ARCHES[arch]}--{s}-sw")
    return net


# --------------------------------------------------------------------------- #
# graph / connectivity helpers (collapsed simple graph => no parallel-edge spoof)
# --------------------------------------------------------------------------- #
def collapsed_graph(net) -> nx.Graph:
    """Simple Graph respecting switch states + in_service; parallel edges merged."""
    return nx.Graph(top.create_nxgraph(net, respect_switches=True, include_out_of_service=False))


def n_independent_loops(net) -> int:
    return len(nx.cycle_basis(collapsed_graph(net)))


def unsupplied(net) -> set[int]:
    return {int(b) for b in top.unsupplied_buses(net)}


def energized_buses(net) -> set[int]:
    """Energized == connected to an ext_grid over in-service / closed-switch elements.
    Defined EXACTLY by ``pandapower.topology.unsupplied_buses`` — never stipulated."""
    return {int(b) for b in net.bus.index} - unsupplied(net)


def _switch_bus_pair(net, si: int) -> Optional[tuple[int, int]]:
    """The two buses a switch bridges (for a line/trafo switch, the element's endpoints)."""
    sw = net.switch.loc[si]
    et, bus, el = sw["et"], int(sw["bus"]), int(sw["element"])
    if et == "b":
        return (bus, el)
    if et == "l":
        ln = net.line.loc[el]
        return (int(ln["from_bus"]), int(ln["to_bus"]))
    if et == "t":
        tr = net.trafo.loc[el]
        return (int(tr["hv_bus"]), int(tr["lv_bus"]))
    return None


def source_buses(net) -> set[int]:
    src = {int(b) for b in net.ext_grid.bus}
    for _, tr in net.trafo.iterrows():
        src.add(int(tr["hv_bus"]))
        src.add(int(tr["lv_bus"]))
    return src


# --------------------------------------------------------------------------- #
# normalize to a RADIAL normal config (open the ring at one point per loop)
# --------------------------------------------------------------------------- #
def normalize_radial(net, log: Optional[list] = None) -> list[int]:
    """Open one closed switch per residual independent loop such that the net becomes
    radial (0 loops) while every bus stays energized (unsupplied set unchanged / empty).
    Returns the ids of switches opened. Ring-operated-radial: the ring exists physically
    but is operated open at a single point (Scope §4.3, §6.4)."""
    opened: list[int] = []
    base_unsupplied = len(unsupplied(net))
    for _ in range(400):  # safety cap
        G = collapsed_graph(net)
        cyc = nx.cycle_basis(G)
        if not cyc:
            break
        n_before = len(cyc)
        cycle = cyc[0]
        edges = {frozenset((int(u), int(v))) for u, v in zip(cycle, cycle[1:] + cycle[:1])}
        chosen = None
        for si in list(net.switch.index[net.switch.closed]):
            pair = _switch_bus_pair(net, si)
            if pair is None or frozenset(pair) not in edges:
                continue
            net.switch.at[si, "closed"] = False  # trial open
            if len(unsupplied(net)) <= base_unsupplied and n_independent_loops(net) < n_before:
                chosen = si
                break
            net.switch.at[si, "closed"] = True  # revert
        if chosen is None:
            if log is not None:
                log.append(f"WARN unbreakable loop (no de-energization-safe switch) on cycle {cycle}")
            break
        opened.append(int(chosen))
    return opened


def assert_radiality(net, ties_only: bool = True) -> None:
    """P3 guard: (a) normal config has 0 independent loops; (b) each open ring point,
    closed ALONE, yields exactly 1 loop. Collapses MultiGraph->Graph first (landmine)."""
    loops = n_independent_loops(net)
    if loops != 0:
        raise RadialityViolation(f"normal config has {loops} independent loops (expected 0 radial)")
    open_sw = list(net.switch.index[~net.switch.closed])
    if ties_only and "type" in net.switch.columns:
        open_sw = [s for s in open_sw if net.switch.at[s, "type"] == "tie"]
    for si in open_sw:
        trial = copy.deepcopy(net)
        trial.switch.at[si, "closed"] = True
        got = n_independent_loops(trial)
        if got != 1:
            raise RadialityViolation(
                f"closing tie {si} alone yields {got} loops (expected exactly 1)"
            )


# --------------------------------------------------------------------------- #
# feeder assignment (Scope §4.1: SimBench subnet labels canonical; CIGRE by graph)
# --------------------------------------------------------------------------- #
def _line_feeder_label(net, li: int, archetype: str) -> Optional[str]:
    subnet = str(net.line.at[li, "subnet"])
    if FEEDER_RE.search(subnet):
        return f"{archetype}_{subnet.split('_')[-1]}"  # e.g. rural_Feeder3
    return None  # Loop_Line / BS_Line / RS_Line / parent label -> not a feeder interior line


def count_feeders(net, grid_id: str) -> int:
    """Canonical feeder count: SimBench distinct ``_Feeder\\d+`` subnet labels; CIGRE = 2
    (structural). rural 8 + semiurb 9 + urban 14 + comm 9 = 40 SimBench + 2 CIGRE = 42."""
    if grid_id == "cigre_mv":
        return len(_cigre_feeder_map(net))
    arch = archetype_of(grid_id)
    labels = {
        _line_feeder_label(net, li, arch)
        for li in net.line.index
        if _line_feeder_label(net, li, arch) is not None
    }
    return len(labels)


def _cigre_feeder_map(net) -> dict[int, str]:
    """Assign CIGRE MV load/interior buses to its 2 feeders by connected component off the
    MV busbar (open ties removed)."""
    src = source_buses(net)
    G = collapsed_graph(net)
    G2 = G.copy()
    G2.remove_nodes_from([b for b in src if b in G2])
    comps = [c for c in nx.connected_components(G2) if len(c) > 0]
    comps.sort(key=lambda c: (-len(c), min(c)))
    fmap: dict[int, str] = {}
    for i, comp in enumerate(comps, start=1):
        for b in comp:
            fmap[int(b)] = f"cigre_Feeder{i}"
    return fmap


def assign_bus_feeders(net, grid_id: str) -> dict[int, Optional[str]]:
    arch = archetype_of(grid_id)
    if grid_id == "cigre_mv":
        return _cigre_feeder_map(net)
    bus_feeder: dict[int, Optional[str]] = {}
    for li in net.line.index:
        lab = _line_feeder_label(net, li, arch)
        if lab is None:
            continue
        for b in (int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])):
            # first _Feeder line wins; source-side buses never carry a _Feeder line
            bus_feeder.setdefault(b, lab)
    return bus_feeder


def _switch_feeder(net, si: int, bus_feeder: dict[int, Optional[str]], grid_id: str) -> Optional[str]:
    sw = net.switch.loc[si]
    et, el = sw["et"], int(sw["element"])
    arch = archetype_of(grid_id)
    if et == "l" and grid_id != "cigre_mv":
        lab = _line_feeder_label(net, el, arch)
        if lab is not None:
            return lab
    pair = _switch_bus_pair(net, si)
    if pair is not None:
        for b in pair:
            if bus_feeder.get(b):
                return bus_feeder[b]
    return None


# --------------------------------------------------------------------------- #
# switchgear tagging (Scope §4.3)
# --------------------------------------------------------------------------- #
def tag_switches(net, grid_id: str, bus_feeder: dict[int, Optional[str]],
                 feeder_earthing: dict[str, str], seed: int) -> None:
    rng = np.random.default_rng(stable_seed(f"{grid_id}_switchtags_{seed}"))
    n = len(net.switch)
    types, remotes, travel, sec_from, sec_to, earth, feeders = [], [], [], [], [], [], []
    for si in net.switch.index:
        sw = net.switch.loc[si]
        et = sw["et"]
        closed = bool(sw["closed"])
        pair = _switch_bus_pair(net, si)
        fa = bus_feeder.get(pair[0]) if pair else None
        fb = bus_feeder.get(pair[1]) if pair else None
        feeder = _switch_feeder(net, si, bus_feeder, grid_id)
        # classify
        if et == "t":
            stype = "breaker"
        elif not closed and fa is not None and fb is not None and fa != fb:
            stype = "tie"  # open ring point between two feeders
        elif not closed:
            stype = "tie"  # any normally-open ring point
        else:
            stype = "RMU/LBS" if rng.random() < 0.35 else "sectionalizer"
        remote = bool(rng.random() < 0.35)  # German MV remote-automation share (documented knob)
        # manual travel time from a seeded lognormal in the 30-90 min band (Scope §6.6)
        tmin = 0.0 if remote else float(np.clip(rng.lognormal(mean=np.log(50), sigma=0.45), 10, 180))
        types.append(stype)
        remotes.append(remote)
        travel.append(round(tmin, 1))
        sec_from.append(pair[0] if pair else -1)
        sec_to.append(pair[1] if pair else -1)
        earth.append(feeder_earthing.get(feeder, EARTHING_REGIMES[0]) if feeder else EARTHING_REGIMES[0])
        feeders.append(feeder)
    net.switch["type"] = types
    net.switch["remote"] = remotes
    net.switch["travel_time_min"] = travel
    net.switch["section_from"] = sec_from
    net.switch["section_to"] = sec_to
    net.switch["earthing_regime"] = earth
    net.switch["feeder_id"] = feeders


def assign_feeder_earthing(feeder_ids: list[str], seed: int) -> dict[str, str]:
    """Per-feeder earthing regime, seeded, from the German-MV mix (Scope §6.2)."""
    rng = np.random.default_rng(stable_seed(f"earthing_{seed}"))
    out = {}
    for fid in sorted(set(feeder_ids)):
        out[fid] = EARTHING_REGIMES[int(rng.choice(len(EARTHING_REGIMES), p=EARTHING_MIX))]
    return out


# --------------------------------------------------------------------------- #
# top-level bank build + persistence
# --------------------------------------------------------------------------- #
def config_hash(seed: int) -> str:
    payload = json.dumps(
        {"seed": seed, "arches": SIMBENCH_ARCHES, "der": DER_SCENARIOS,
         "earthing_mix": EARTHING_MIX.tolist()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def instantiate_bank(seed: int, log: Optional[list] = None) -> dict:
    """Build + power-flow-verify + radially-normalize all 13 instances."""
    bank = {}
    for gid in bank_grid_ids():
        net = instantiate_grid(gid)
        pp.runpp(net)
        assert bool(net.converged), f"{gid} did not converge"
        opened = normalize_radial(net, log=log)
        pp.runpp(net)
        assert bool(net.converged), f"{gid} did not converge after normalization"
        if log is not None:
            log.append(f"{gid}: normalized (opened {len(opened)} switch(es)); "
                       f"loops={n_independent_loops(net)} unsupplied={len(unsupplied(net))}")
        bank[gid] = net
    return bank


def build_switch_manifest(bank: dict, seed: int) -> pd.DataFrame:
    # first pass: collect all feeder ids to assign earthing consistently per feeder
    all_feeders: list[str] = []
    bus_feeders: dict[str, dict] = {}
    for gid, net in bank.items():
        bf = assign_bus_feeders(net, gid)
        bus_feeders[gid] = bf
        all_feeders.extend([f for f in bf.values() if f])
    feeder_earthing = assign_feeder_earthing(all_feeders, seed)
    rows = []
    for gid, net in bank.items():
        tag_switches(net, gid, bus_feeders[gid], feeder_earthing, seed)
        for si in net.switch.index:
            sw = net.switch.loc[si]
            rows.append({
                "grid_id": gid,
                "archetype": archetype_of(gid),
                "switch_id": int(si),
                "type": sw["type"],
                "feeder_id": sw["feeder_id"],
                "remote": bool(sw["remote"]),
                "travel_time_min": float(sw["travel_time_min"]),
                "section_from": int(sw["section_from"]),
                "section_to": int(sw["section_to"]),
                "earthing_regime": sw["earthing_regime"],
                "closed_normal": bool(sw["closed"]),
            })
    return pd.DataFrame(rows), feeder_earthing


def build_section_manifest(bank: dict, feeder_earthing: dict[str, str]) -> pd.DataFrame:
    """One restorable section per ``net.load`` on the representative (--0 / CIGRE) instance
    of each archetype (DER scenarios share wires). 448 SimBench + 18 CIGRE = 466."""
    rows = []
    seen_arch = set()
    for gid, net in bank.items():
        arch = archetype_of(gid)
        if arch in seen_arch or gid != representative_grid_id(arch):
            continue
        seen_arch.add(arch)
        bf = assign_bus_feeders(net, gid)
        for lo in net.load.index:
            bus = int(net.load.at[lo, "bus"])
            fid = bf.get(bus)
            rows.append({
                "section_id": f"{arch}_sec_{int(lo)}",
                "archetype": arch,
                "grid_id": gid,
                "load_id": int(lo),
                "bus": bus,
                "feeder_id": fid,
                "earthing_regime": feeder_earthing.get(fid, EARTHING_REGIMES[0]) if fid else EARTHING_REGIMES[0],
                "p_mw_nominal": float(net.load.at[lo, "p_mw"]),
            })
    return pd.DataFrame(rows)


def build_grids(out_dir: str, seed: int) -> dict:
    """S2.1 orchestrator: build the bank, persist per-instance JSON + the two manifests."""
    os.makedirs(out_dir, exist_ok=True)
    log: list[str] = []
    bank = instantiate_bank(seed, log=log)
    chash = config_hash(seed)
    for gid, net in bank.items():
        # NOTE: do not stamp custom keys onto the pandapowerNet dict — it breaks
        # from_json round-trip. Provenance lives in grid_manifest.json below.
        pp.to_json(net, os.path.join(out_dir, f"{gid}.json"))
    switch_manifest, feeder_earthing = build_switch_manifest(bank, seed)
    section_manifest = build_section_manifest(bank, feeder_earthing)
    switch_manifest.to_csv(os.path.join(out_dir, "switch_manifest.csv"), index=False)
    section_manifest.to_csv(os.path.join(out_dir, "section_manifest.csv"), index=False)
    meta = {
        "substrate_version": "v1",
        "config_hash": chash,
        "seed": seed,
        "n_grid_instances": len(bank),
        "n_feeders_canonical": int(switch_manifest["feeder_id"].nunique()),
        "n_sections": int(len(section_manifest)),
        "grid_ids": list(bank.keys()),
        "log": log,
    }
    with open(os.path.join(out_dir, "grid_manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260727)
    a = ap.parse_args()
    m = build_grids(a.out, a.seed)
    print(json.dumps(m, indent=2))
