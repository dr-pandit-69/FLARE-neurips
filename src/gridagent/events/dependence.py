"""Cause-aware event dependence identifiers for the v4 experiment generation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "goal_v4" / "dependence.json"
)
EVENT_ID_FIELDS = (
    "grid_id",
    "feeder",
    "weather_feeder",
    "timestamp",
    "step",
    "section_bus",
    "cause",
    "fault_type",
    "component",
    "klass",
    "interrupts",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_dependence_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text())
    validate_dependence_config(config)
    return config


def validate_dependence_config(config: Mapping) -> None:
    weather = config.get("weather", {})
    gaps = [int(value) for value in weather.get("sensitivity_merge_gap_hours", [])]
    primary = int(weather.get("primary_merge_gap_hours", -1))
    if config.get("schema_version") != "gridagent-dependence-v1":
        raise ValueError("unsupported dependence schema")
    if not 0.0 <= float(weather.get("active_threshold", -1.0)) <= 1.0:
        raise ValueError("weather active threshold must lie in [0, 1]")
    if gaps != sorted(set(gaps)) or primary not in gaps:
        raise ValueError("weather sensitivity gaps must be sorted, unique, and include primary")
    if int(config.get("timestep_minutes", 0)) <= 0:
        raise ValueError("timestep_minutes must be positive")


def dependence_definition_sha256(config: Mapping) -> str:
    validate_dependence_config(config)
    return canonical_sha256(config)


def compute_storm_score(gust_m_s, precip_kg_m2, config: Mapping | None = None):
    """Return the frozen v4 meteorological activity score."""
    config = config or load_dependence_config()
    score = config["weather"]["storm_score"]
    gust = np.nan_to_num(np.asarray(gust_m_s, dtype=float), nan=0.0)
    precip = np.nan_to_num(np.asarray(precip_kg_m2, dtype=float), nan=0.0)
    gust_component = np.clip(
        (gust - float(score["gust_onset_m_s"])) / float(score["gust_span_m_s"]),
        0.0,
        1.0,
    )
    precip_component = np.clip(
        precip / float(score["precip_span_kg_m2"]), 0.0, 1.0
    )
    result = (
        float(score["gust_weight"]) * gust_component
        + float(score["precip_weight"]) * precip_component
    )
    return float(result) if result.ndim == 0 else result


def _weather_groups(weather) -> dict[int, pd.DataFrame]:
    if isinstance(weather, pd.DataFrame):
        if "feeder_id" not in weather:
            raise ValueError("weather dataframe requires feeder_id")
        source = {
            int(feeder_id): group.copy()
            for feeder_id, group in weather.groupby("feeder_id", sort=True)
        }
    elif isinstance(weather, Mapping):
        source = {int(key): value.copy() for key, value in weather.items()}
    else:
        raise TypeError("weather must be a dataframe or feeder-to-dataframe mapping")
    return {
        feeder_id: group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        for feeder_id, group in source.items()
    }


def build_weather_episode_index(
    weather,
    *,
    config: Mapping | None = None,
    merge_gap_hours: int | None = None,
) -> dict:
    """Build deterministic active-weather episodes and a feeder-step lookup."""
    config = dict(config or load_dependence_config())
    validate_dependence_config(config)
    weather_cfg = config["weather"]
    gap_hours = int(
        weather_cfg["primary_merge_gap_hours"]
        if merge_gap_hours is None
        else merge_gap_hours
    )
    allowed = {int(value) for value in weather_cfg["sensitivity_merge_gap_hours"]}
    if gap_hours not in allowed:
        raise ValueError(f"merge gap {gap_hours} is not a frozen sensitivity definition")

    threshold = float(weather_cfg["active_threshold"])
    cadence = pd.Timedelta(minutes=int(config["timestep_minutes"]))
    max_inactive_gap = pd.Timedelta(hours=gap_hours)
    episodes: list[dict] = []
    by_step: dict[tuple[int, int], str] = {}
    by_timestamp: dict[tuple[int, int], str] = {}

    for feeder_id, group in _weather_groups(weather).items():
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        scores = compute_storm_score(
            group["max_wind_gust_m/s"].to_numpy(dtype=float),
            group["precipitation_kg/m2"].to_numpy(dtype=float),
            config,
        )
        active_positions = np.flatnonzero(scores >= threshold).tolist()
        if not active_positions:
            continue

        runs: list[list[int]] = [[active_positions[0]]]
        for position in active_positions[1:]:
            previous = runs[-1][-1]
            inactive_gap = timestamps.iloc[position] - timestamps.iloc[previous] - cadence
            if inactive_gap <= max_inactive_gap:
                runs[-1].append(position)
            else:
                runs.append([position])

        for positions in runs:
            start = timestamps.iloc[positions[0]]
            end = timestamps.iloc[positions[-1]]
            identity = {
                "schema": config["schema_version"],
                "weather_feeder": feeder_id,
                "merge_gap_hours": gap_hours,
                "active_threshold": threshold,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
            episode_id = f"we{gap_hours:02d}_{canonical_sha256(identity)[:16]}"
            episodes.append(
                {
                    "weather_episode_id": episode_id,
                    "weather_feeder": feeder_id,
                    "merge_gap_hours": gap_hours,
                    "active_threshold": threshold,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "n_active_steps": len(positions),
                    "max_storm_score": float(np.max(scores[positions])),
                }
            )
            # The episode covers the complete interval between its first and
            # last active step, including the permitted inactive merge gaps.
            for position in range(positions[0], positions[-1] + 1):
                by_step[(feeder_id, int(position))] = episode_id
                by_timestamp[(feeder_id, int(timestamps.iloc[position].value))] = episode_id

    episodes.sort(key=lambda row: (row["weather_feeder"], row["start"]))
    return {
        "merge_gap_hours": gap_hours,
        "active_threshold": threshold,
        "episodes": episodes,
        "by_step": by_step,
        "by_timestamp": by_timestamp,
    }


def _event_content(event: Mapping) -> dict:
    return {field: event.get(field) for field in EVENT_ID_FIELDS}


def ensure_event_ids(events: Iterable[Mapping]) -> list[dict]:
    """Assign stable unique IDs to records that do not already carry one."""
    rows = [dict(event) for event in events]
    counts: Counter[str] = Counter()
    for event in sorted(
        (row for row in rows if not row.get("event_id")),
        key=lambda row: json.dumps(_event_content(row), sort_keys=True, default=str),
    ):
        content = _event_content(event)
        fingerprint = canonical_sha256(content)
        occurrence = counts[fingerprint]
        counts[fingerprint] += 1
        event["event_id"] = f"ev_{canonical_sha256({'event': content, 'occurrence': occurrence})[:20]}"

    ids = [str(row["event_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("event_id values are not unique")
    known = set(ids)
    for row in rows:
        parent = row.get("parent_event_id")
        if parent is not None and str(parent) not in known:
            raise ValueError(f"unknown parent_event_id {parent!r}")
        row["parent_event_id"] = None if parent is None else str(parent)
        row["branch_root_id"] = str(row.get("branch_root_id") or row["event_id"])
    return rows


def _assign_event_bursts(events: list[dict], gap_steps: int) -> dict[str, str]:
    by_series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        by_series[(str(event["grid_id"]), str(event["feeder"]))].append(event)

    result: dict[str, str] = {}
    for series, rows in sorted(by_series.items()):
        ordered = sorted(rows, key=lambda row: (int(row["step"]), str(row["event_id"])))
        groups: list[list[dict]] = []
        for event in ordered:
            if not groups or int(event["step"]) - int(groups[-1][-1]["step"]) > gap_steps:
                groups.append([event])
            else:
                groups[-1].append(event)
        for group in groups:
            identity = {
                "series": series,
                "gap_steps": gap_steps,
                "first": group[0]["event_id"],
                "last": group[-1]["event_id"],
            }
            burst_id = f"eb_{canonical_sha256(identity)[:16]}"
            for event in group:
                result[event["event_id"]] = burst_id
    return result


def annotate_event_dependence(
    events: Iterable[Mapping],
    weather,
    *,
    config: Mapping | None = None,
) -> tuple[list[dict], dict]:
    """Attach weather, Hawkes, generic-burst, and primary dependence identifiers."""
    config = dict(config or load_dependence_config())
    validate_dependence_config(config)
    rows = ensure_event_ids(events)
    definition_hash = dependence_definition_sha256(config)
    weather_cfg = config["weather"]
    gap_indexes = {
        int(gap): build_weather_episode_index(
            weather, config=config, merge_gap_hours=int(gap)
        )
        for gap in weather_cfg["sensitivity_merge_gap_hours"]
    }
    primary_gap = int(weather_cfg["primary_merge_gap_hours"])

    for event in rows:
        weather_feeder = event.get("weather_feeder")
        step = event.get("step")
        timestamp = pd.Timestamp(event["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        for gap, index in gap_indexes.items():
            episode_id = None
            if str(event.get("cause")) == "weather" and weather_feeder is not None:
                episode_id = index["by_step"].get((int(weather_feeder), int(step)))
                if episode_id is None:
                    episode_id = index["by_timestamp"].get(
                        (int(weather_feeder), int(timestamp.value))
                    )
            event[f"weather_episode_id_gap{gap}h"] = episode_id
        event["weather_episode_id"] = event[
            f"weather_episode_id_gap{primary_gap}h"
        ]

    family_sizes = Counter(str(row["branch_root_id"]) for row in rows)
    burst_gap_steps = int(
        round(
            float(config["generic_event_burst_gap_hours"])
            * 60.0
            / float(config["timestep_minutes"])
        )
    )
    burst_ids = _assign_event_bursts(rows, burst_gap_steps)

    for event in rows:
        event["event_burst_id"] = burst_ids[event["event_id"]]
        weather_episode_id = event["weather_episode_id"]
        is_hawkes_family = family_sizes[str(event["branch_root_id"])] > 1
        if weather_episode_id is not None:
            block_type = "weather_episode"
            block_id = str(weather_episode_id)
        elif str(event.get("cause")) != "weather" and is_hawkes_family:
            block_type = "hawkes_family"
            block_id = f"hf_{canonical_sha256(str(event['branch_root_id']))[:16]}"
        else:
            block_type = "background_singleton"
            block_id = f"bg_{canonical_sha256(str(event['event_id']))[:16]}"
        event["dependence_block_id"] = block_id
        event["dependence_block_type"] = block_type
        event["dependence_definition_sha256"] = definition_hash

    event_counts_by_block_type = Counter(
        row["dependence_block_type"] for row in rows
    )
    block_type_by_id = {
        str(row["dependence_block_id"]): str(row["dependence_block_type"])
        for row in rows
    }
    block_counts = Counter(block_type_by_id.values())
    weather_rows = [row for row in rows if row.get("cause") == "weather"]
    summary = {
        "schema_version": config["schema_version"],
        "dependence_definition_sha256": definition_hash,
        "n_events": len(rows),
        "n_dependence_blocks": len({row["dependence_block_id"] for row in rows}),
        "dependence_block_type_counts": dict(sorted(block_counts.items())),
        "event_counts_by_dependence_block_type": dict(
            sorted(event_counts_by_block_type.items())
        ),
        "weather_event_count": len(weather_rows),
        "weather_event_association_count": sum(
            row["weather_episode_id"] is not None for row in weather_rows
        ),
        "weather_event_outside_active_count": sum(
            row["weather_episode_id"] is None for row in weather_rows
        ),
        "weather_episode_counts_by_gap_hours": {
            str(gap): len(index["episodes"]) for gap, index in gap_indexes.items()
        },
    }
    return rows, summary


def validate_annotated_events(events: Iterable[Mapping], summary: Mapping | None = None) -> list[str]:
    rows = [dict(event) for event in events]
    errors: list[str] = []
    required = {
        "event_id",
        "parent_event_id",
        "branch_root_id",
        "event_burst_id",
        "weather_episode_id",
        "dependence_block_id",
        "dependence_block_type",
        "dependence_definition_sha256",
    }
    allowed_types = {"weather_episode", "hawkes_family", "background_singleton"}
    ids = {str(row.get("event_id")) for row in rows}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{row.get('event_id', '<missing>')}: missing {missing}")
            continue
        if row["dependence_block_type"] not in allowed_types:
            errors.append(f"{row['event_id']}: invalid dependence block type")
        if row.get("parent_event_id") is not None and str(row["parent_event_id"]) not in ids:
            errors.append(f"{row['event_id']}: unknown parent")
        if row.get("cause") == "weather" and row["weather_episode_id"] is None:
            if row["dependence_block_type"] != "background_singleton":
                errors.append(f"{row['event_id']}: inactive weather event is not singleton")
        if row.get("cause") != "weather" and row["dependence_block_type"] == "weather_episode":
            errors.append(f"{row['event_id']}: non-weather event joined weather block")
        if any(key in row for key in ("storm_cluster_id", "storm_window")):
            errors.append(f"{row['event_id']}: deprecated storm field present")

    by_root: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_root[str(row.get("branch_root_id"))].append(row)
    for root, family in by_root.items():
        nonweather = [row for row in family if row.get("cause") != "weather"]
        if len(nonweather) > 1 and len(
            {row.get("dependence_block_id") for row in nonweather}
        ) != 1:
            errors.append(f"{root}: non-weather Hawkes family split across blocks")

    hashes = {row.get("dependence_definition_sha256") for row in rows}
    if len(hashes) > 1:
        errors.append("mixed dependence-definition hashes")
    if summary and summary.get("dependence_definition_sha256") not in hashes:
        errors.append("summary dependence-definition hash mismatch")
    return errors
