#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "DT" / "outputs" / "ims_dt.db"
DEFAULT_QUESTIONS = PROJECT_ROOT / "agent" / "eval" / "agent_english_questions_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_v1.json"

CORE_FEATURES = [
    "health_index",
    "rms",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "high_band_energy_ratio",
    "spectral_entropy",
    "dominant_frequency",
]
DEGRADATION_FEATURES = ["rms", "kurtosis", "crest_factor", "high_band_energy_ratio", "spectral_entropy"]
TREND_WINDOW = 100
Z_THRESHOLD = 3.0
NUMERIC_TOLERANCE = 1e-3
TIME_POLICY = "latest_available_record"
TIME_POLICY_DESCRIPTION = (
    "For IMS DT benchmark questions, current/latest/now means the latest available "
    "timestamp in the offline IMS digital-twin database for the requested experiment "
    "and bearing, not the real-world current date."
)

QUESTION_SPECS: dict[str, dict[str, Any]] = {
    "dt_feature_001": {"action": "features", "experiment": "1st_test", "bearing_id": "bearing_3", "features": ["rms"]},
    "dt_feature_002": {"action": "compare_feature", "experiment": "1st_test", "bearing_id": "bearing_3", "feature": "kurtosis", "scope": "latest_bearings_in_experiment"},
    "dt_feature_003": {"action": "features", "experiment": "2nd_test", "bearing_id": "bearing_1", "features": ["crest_factor"]},
    "dt_feature_004": {"action": "features", "experiment": "3rd_test", "bearing_id": "bearing_3", "features": ["high_band_energy_ratio"]},
    "dt_feature_005": {"action": "rank_feature", "experiment": "1st_test", "feature": "rms", "scope": "latest_bearings_in_experiment", "order": "desc"},
    "dt_feature_006": {"action": "rank_feature", "experiment": "2nd_test", "feature": "kurtosis", "scope": "latest_bearings_in_experiment", "order": "desc"},
    "dt_feature_007": {"action": "feature_abnormality", "experiment": "1st_test", "bearing_id": "bearing_4", "feature": "spectral_entropy"},
    "dt_feature_008": {"action": "features_all_bearings", "experiment": "3rd_test", "features": ["dominant_frequency"]},
    "dt_feature_009": {"action": "features", "experiment": "1st_test", "bearing_id": "bearing_3", "features": ["impulse_factor", "clearance_factor"]},
    "dt_feature_010": {"action": "rank_feature", "experiment": None, "feature": "high_band_energy_ratio", "scope": "latest_bearings_all_experiments", "order": "desc"},
    "dt_state_001": {"action": "latest_state", "experiment": "1st_test", "bearing_id": "bearing_3"},
    "dt_state_002": {"action": "rank_health", "experiment": "1st_test", "scope": "latest_bearings_in_experiment", "order": "desc"},
    "dt_state_003": {"action": "state_check", "experiment": "2nd_test", "bearing_id": "bearing_1", "positive_states": ["early_degradation", "degradation", "failure_near"]},
    "dt_state_004": {"action": "state_check", "experiment": "3rd_test", "bearing_id": "bearing_3", "positive_states": ["degradation", "failure_near"]},
    "dt_state_005": {"action": "rank_health", "experiment": None, "scope": "latest_bearings_all_experiments", "order": "desc"},
    "dt_state_006": {"action": "features", "experiment": "1st_test", "bearing_id": "bearing_4", "features": ["health_index"]},
    "dt_state_007": {"action": "filter_state", "experiment": None, "scope": "latest_bearings_all_experiments", "state": "failure_near"},
    "dt_state_008": {"action": "filter_state", "experiment": None, "scope": "latest_bearings_all_experiments", "state": "early_degradation"},
    "dt_state_009": {"action": "states_all_bearings", "experiment": "2nd_test"},
    "dt_state_010": {"action": "rank_health", "experiment": None, "scope": "latest_bearings_all_experiments", "order": "asc"},
    "dt_trend_001": {"action": "trend", "experiment": "1st_test", "bearing_id": "bearing_3", "metric": "health_index", "window": TREND_WINDOW},
    "dt_trend_002": {"action": "trend", "experiment": "1st_test", "bearing_id": "bearing_4", "metric": "rms", "window": TREND_WINDOW},
    "dt_trend_003": {"action": "near_end_trend", "experiment": "2nd_test", "bearing_id": "bearing_1", "metric": "kurtosis", "window": TREND_WINDOW},
    "dt_trend_004": {"action": "trend", "experiment": "3rd_test", "bearing_id": "bearing_3", "metric": "high_band_energy_ratio", "window": TREND_WINDOW},
    "dt_trend_005": {"action": "rank_trend", "experiment": "1st_test", "metric": "health_index", "window": TREND_WINDOW, "order": "desc"},
    "dt_trend_006": {"action": "compare_trends", "experiment": "2nd_test", "metric": "health_index", "window": TREND_WINDOW},
    "dt_trend_007": {"action": "trend", "experiment": "1st_test", "bearing_id": "bearing_3", "metric": "crest_factor", "window": TREND_WINDOW},
    "dt_trend_008": {"action": "first_degradation", "experiment": "1st_test", "bearing_id": "bearing_3", "state_order": ["early_degradation", "degradation", "failure_near"]},
    "dt_trend_009": {"action": "earliest_feature_change", "experiment": "1st_test", "bearing_id": "bearing_3", "features": DEGRADATION_FEATURES, "baseline_window": TREND_WINDOW, "z_threshold": Z_THRESHOLD},
    "dt_trend_010": {"action": "compare_metric_trends", "experiment": "1st_test", "bearing_id": "bearing_3", "metrics": ["rms", "kurtosis"], "window": TREND_WINDOW},
}


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_float(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_float(v, digits) for v in value]
    return value


def row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def latest_snapshot(conn: sqlite3.Connection, experiment: str, bearing_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        select * from bearing_snapshots
        where experiment = ? and bearing_id = ?
        order by sequence_index desc
        limit 1
        """,
        (experiment, bearing_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"No latest snapshot for {experiment}/{bearing_id}")
    return row_dict(row)


def latest_snapshots(conn: sqlite3.Connection, experiment: Optional[str] = None) -> list[dict[str, Any]]:
    if experiment:
        rows = conn.execute(
            """
            with latest as (
                select bearing_id, max(sequence_index) as max_seq
                from bearing_snapshots
                where experiment = ?
                group by bearing_id
            )
            select s.*
            from bearing_snapshots s
            join latest l on s.bearing_id = l.bearing_id and s.sequence_index = l.max_seq
            where s.experiment = ?
            order by s.experiment, s.bearing_id
            """,
            (experiment, experiment),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            with latest as (
                select experiment, bearing_id, max(sequence_index) as max_seq
                from bearing_snapshots
                group by experiment, bearing_id
            )
            select s.*
            from bearing_snapshots s
            join latest l
              on s.experiment = l.experiment
             and s.bearing_id = l.bearing_id
             and s.sequence_index = l.max_seq
            order by s.experiment, s.bearing_id
            """
        ).fetchall()
    return [row_dict(row) for row in rows]


def sequence_rows(conn: sqlite3.Connection, experiment: str, bearing_id: str, columns: Iterable[str]) -> list[dict[str, Any]]:
    safe_columns = [c for c in columns if c.replace("_", "").isalnum()]
    select_cols = ", ".join(["sequence_index", "timestamp", "health_state"] + safe_columns)
    rows = conn.execute(
        f"""
        select {select_cols}
        from bearing_snapshots
        where experiment = ? and bearing_id = ?
        order by sequence_index
        """,
        (experiment, bearing_id),
    ).fetchall()
    return [row_dict(row) for row in rows]


def mean(rows: list[dict[str, Any]], metric: str) -> float:
    vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(vals) / len(vals)


def trend_label(delta: float, metric: str) -> str:
    if metric == "health_index":
        if delta >= 0.20:
            return "strongly_increasing_degradation"
        if delta >= 0.05:
            return "increasing_degradation"
        if delta <= -0.05:
            return "decreasing_or_recovering"
        return "mostly_stable"
    if delta > 0:
        return "increasing"
    if delta < 0:
        return "decreasing"
    return "mostly_stable"


def trend_summary(conn: sqlite3.Connection, experiment: str, bearing_id: str, metric: str, window: int) -> dict[str, Any]:
    rows = sequence_rows(conn, experiment, bearing_id, [metric, "health_index"])
    if not rows:
        raise ValueError(f"No sequence rows for {experiment}/{bearing_id}")
    w = max(1, min(window, len(rows)))
    early = rows[:w]
    late = rows[-w:]
    early_mean = mean(early, metric)
    late_mean = mean(late, metric)
    delta = late_mean - early_mean
    percent_change = None if early_mean == 0 else delta / abs(early_mean)
    return {
        "experiment": experiment,
        "bearing_id": bearing_id,
        "metric": metric,
        "samples": len(rows),
        "window": w,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "delta": delta,
        "percent_change": percent_change,
        "trend": trend_label(delta, metric),
        "latest_sequence_index": rows[-1]["sequence_index"],
        "latest_timestamp": rows[-1]["timestamp"],
        "latest_health_state": rows[-1]["health_state"],
    }


def near_end_trend(conn: sqlite3.Connection, experiment: str, bearing_id: str, metric: str, window: int) -> dict[str, Any]:
    rows = sequence_rows(conn, experiment, bearing_id, [metric, "health_index"])
    if len(rows) < 2:
        raise ValueError(f"Too few rows for near-end trend: {experiment}/{bearing_id}")
    w = max(1, min(window, len(rows) // 2))
    previous = rows[-2 * w : -w]
    late = rows[-w:]
    previous_mean = mean(previous, metric)
    late_mean = mean(late, metric)
    delta = late_mean - previous_mean
    return {
        "experiment": experiment,
        "bearing_id": bearing_id,
        "metric": metric,
        "window": w,
        "previous_window_mean": previous_mean,
        "late_window_mean": late_mean,
        "near_end_delta": delta,
        "near_end_trend": trend_label(delta, metric),
        "latest_sequence_index": rows[-1]["sequence_index"],
        "latest_timestamp": rows[-1]["timestamp"],
        "latest_health_state": rows[-1]["health_state"],
    }


def first_degradation(conn: sqlite3.Connection, experiment: str, bearing_id: str, states: list[str]) -> dict[str, Any]:
    placeholders = ",".join("?" for _ in states)
    row = conn.execute(
        f"""
        select sequence_index, timestamp, health_state, health_index, rms, kurtosis, crest_factor,
               high_band_energy_ratio, spectral_entropy, evidence
        from bearing_snapshots
        where experiment = ? and bearing_id = ? and health_state in ({placeholders})
        order by sequence_index
        limit 1
        """,
        (experiment, bearing_id, *states),
    ).fetchone()
    if row is None:
        return {"experiment": experiment, "bearing_id": bearing_id, "found": False, "states": states}
    return {"experiment": experiment, "bearing_id": bearing_id, "found": True, **row_dict(row)}


def earliest_feature_change(conn: sqlite3.Connection, experiment: str, bearing_id: str, features: list[str], baseline_window: int, z_threshold: float) -> dict[str, Any]:
    rows = sequence_rows(conn, experiment, bearing_id, features + ["health_index"])
    if not rows:
        raise ValueError(f"No rows for {experiment}/{bearing_id}")
    w = max(2, min(baseline_window, len(rows)))
    baseline = rows[:w]
    feature_results = []
    for feature in features:
        values = [float(row[feature]) for row in baseline]
        baseline_mean = sum(values) / len(values)
        variance = sum((v - baseline_mean) ** 2 for v in values) / len(values)
        baseline_std = math.sqrt(variance)
        first = None
        for row in rows[w:]:
            value = float(row[feature])
            z_score = None if baseline_std == 0 else (value - baseline_mean) / baseline_std
            if z_score is not None and z_score >= z_threshold:
                first = {
                    "feature": feature,
                    "sequence_index": row["sequence_index"],
                    "timestamp": row["timestamp"],
                    "value": value,
                    "z_score": z_score,
                    "health_state": row["health_state"],
                }
                break
        feature_results.append({
            "feature": feature,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "first_crossing": first,
        })
    crossings = [item["first_crossing"] for item in feature_results if item["first_crossing"] is not None]
    earliest = min(crossings, key=lambda item: item["sequence_index"]) if crossings else None
    return {
        "experiment": experiment,
        "bearing_id": bearing_id,
        "baseline_window": w,
        "z_threshold": z_threshold,
        "features": feature_results,
        "earliest_feature_change": earliest,
    }


def compact_snapshot(row: dict[str, Any], features: Optional[list[str]] = None) -> dict[str, Any]:
    features = features or CORE_FEATURES
    return {
        "experiment": row.get("experiment"),
        "bearing_id": row.get("bearing_id"),
        "timestamp": row.get("timestamp"),
        "sequence_index": row.get("sequence_index"),
        "health_state": row.get("health_state"),
        "health_index": row.get("health_index"),
        "features": {feature: row.get(feature) for feature in features if feature in row},
        "evidence": row.get("evidence") or "",
    }


def rank_records(records: list[dict[str, Any]], metric: str, order: str) -> list[dict[str, Any]]:
    reverse = order == "desc"
    ranked = sorted(records, key=lambda row: (row.get(metric) is None, float(row.get(metric) or 0.0)), reverse=reverse)
    return [
        {
            "rank": idx,
            "experiment": row.get("experiment"),
            "bearing_id": row.get("bearing_id"),
            "timestamp": row.get("timestamp"),
            "sequence_index": row.get("sequence_index"),
            "health_state": row.get("health_state"),
            metric: row.get(metric),
        }
        for idx, row in enumerate(ranked, start=1)
    ]


def build_expected(conn: sqlite3.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    action = spec["action"]
    if action in {"features", "latest_state"}:
        row = latest_snapshot(conn, spec["experiment"], spec["bearing_id"])
        features = spec.get("features") or CORE_FEATURES
        result = compact_snapshot(row, features)
        if action == "latest_state":
            result["features"] = {"health_index": row.get("health_index")}
        return result

    if action == "compare_feature":
        records = latest_snapshots(conn, spec["experiment"])
        rankings = rank_records(records, spec["feature"], "desc")
        target = next(row for row in rankings if row["bearing_id"] == spec["bearing_id"])
        return {
            "experiment": spec["experiment"],
            "feature": spec["feature"],
            "target": target,
            "is_highest": target["rank"] == 1,
            "rankings": rankings,
        }

    if action == "rank_feature":
        records = latest_snapshots(conn, spec.get("experiment"))
        rankings = rank_records(records, spec["feature"], spec["order"])
        return {
            "experiment": spec.get("experiment") or "all",
            "feature": spec["feature"],
            "scope": spec["scope"],
            "winner": rankings[0] if rankings else None,
            "rankings": rankings,
        }

    if action == "feature_abnormality":
        row = latest_snapshot(conn, spec["experiment"], spec["bearing_id"])
        evidence = row.get("evidence") or ""
        feature = spec["feature"]
        abnormal = row.get("health_state") != "normal" and feature.replace("_", " ") in evidence.lower()
        return {
            **compact_snapshot(row, [feature, "health_index"]),
            "feature": feature,
            "is_abnormal": abnormal,
            "abnormality_rule": "True when the latest health_state is non-normal and the feature is explicitly cited in the snapshot evidence string.",
        }

    if action == "features_all_bearings":
        records = latest_snapshots(conn, spec["experiment"])
        return {
            "experiment": spec["experiment"],
            "features": spec["features"],
            "bearings": [compact_snapshot(row, spec["features"] + ["health_index"]) for row in records],
        }

    if action == "rank_health":
        records = latest_snapshots(conn, spec.get("experiment"))
        rankings = rank_records(records, "health_index", spec["order"])
        return {
            "experiment": spec.get("experiment") or "all",
            "scope": spec["scope"],
            "order": spec["order"],
            "selected": rankings[0] if rankings else None,
            "rankings": rankings,
        }

    if action == "state_check":
        row = latest_snapshot(conn, spec["experiment"], spec["bearing_id"])
        positive = row.get("health_state") in set(spec["positive_states"])
        return {
            **compact_snapshot(row, ["health_index"]),
            "positive_states": spec["positive_states"],
            "answer_boolean": positive,
        }

    if action == "filter_state":
        records = latest_snapshots(conn, spec.get("experiment"))
        matches = [compact_snapshot(row, ["health_index"]) for row in records if row.get("health_state") == spec["state"]]
        return {
            "experiment": spec.get("experiment") or "all",
            "scope": spec["scope"],
            "state": spec["state"],
            "matches": matches,
            "count": len(matches),
        }

    if action == "states_all_bearings":
        records = latest_snapshots(conn, spec["experiment"])
        return {
            "experiment": spec["experiment"],
            "bearings": [compact_snapshot(row, ["health_index"]) for row in records],
        }

    if action == "trend":
        return trend_summary(conn, spec["experiment"], spec["bearing_id"], spec["metric"], spec["window"])

    if action == "near_end_trend":
        return near_end_trend(conn, spec["experiment"], spec["bearing_id"], spec["metric"], spec["window"])

    if action == "rank_trend":
        records = []
        for row in latest_snapshots(conn, spec["experiment"]):
            records.append(trend_summary(conn, spec["experiment"], row["bearing_id"], spec["metric"], spec["window"]))
        reverse = spec["order"] == "desc"
        ranked = sorted(records, key=lambda row: row["delta"], reverse=reverse)
        for idx, row in enumerate(ranked, start=1):
            row["rank"] = idx
        return {
            "experiment": spec["experiment"],
            "metric": spec["metric"],
            "window": spec["window"],
            "selected": ranked[0] if ranked else None,
            "rankings": ranked,
        }

    if action == "compare_trends":
        records = []
        for row in latest_snapshots(conn, spec["experiment"]):
            records.append(trend_summary(conn, spec["experiment"], row["bearing_id"], spec["metric"], spec["window"]))
        records.sort(key=lambda row: row["delta"], reverse=True)
        return {"experiment": spec["experiment"], "metric": spec["metric"], "window": spec["window"], "trend_comparison": records}

    if action == "first_degradation":
        return first_degradation(conn, spec["experiment"], spec["bearing_id"], spec["state_order"])

    if action == "earliest_feature_change":
        return earliest_feature_change(conn, spec["experiment"], spec["bearing_id"], spec["features"], spec["baseline_window"], spec["z_threshold"])

    if action == "compare_metric_trends":
        return {
            "experiment": spec["experiment"],
            "bearing_id": spec["bearing_id"],
            "window": spec["window"],
            "trends": [trend_summary(conn, spec["experiment"], spec["bearing_id"], metric, spec["window"]) for metric in spec["metrics"]],
        }

    raise ValueError(f"Unsupported benchmark action: {action}")


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def reference_answer(item_id: str, question: str, spec: dict[str, Any], result: dict[str, Any]) -> str:
    action = spec["action"]
    if action == "features":
        features = result["features"]
        values = ", ".join(f"{name}={fmt(value)}" for name, value in features.items())
        return f"Using the latest available IMS DT record, {result['experiment']}/{result['bearing_id']} has {values} at sequence_index {result['sequence_index']} ({result['timestamp']}) and is classified as {result['health_state']}."
    if action == "latest_state":
        return f"Using the latest available IMS DT record, {result['experiment']}/{result['bearing_id']} is classified as {result['health_state']} with health_index={fmt(result['health_index'])} at sequence_index {result['sequence_index']} ({result['timestamp']})."
    if action == "compare_feature":
        target = result["target"]
        leader = result["rankings"][0]
        return f"{yes_no(result['is_highest'])}. In the latest {result['experiment']} records, {target['bearing_id']} has {result['feature']}={fmt(target[result['feature']])} and ranks {target['rank']}; the highest is {leader['bearing_id']} with {result['feature']}={fmt(leader[result['feature']])}."
    if action == "rank_feature":
        winner = result["winner"]
        return f"The highest latest {result['feature']} in scope {result['scope']} is {winner['experiment']}/{winner['bearing_id']} with {result['feature']}={fmt(winner[result['feature']])} at sequence_index {winner['sequence_index']}."
    if action == "feature_abnormality":
        feature = result["feature"]
        return f"{yes_no(result['is_abnormal'])}. The latest {result['experiment']}/{result['bearing_id']} has {feature}={fmt(result['features'][feature])}, health_state={result['health_state']}, and its evidence cites spectral entropy abnormality when applicable."
    if action == "features_all_bearings":
        feature = spec["features"][0]
        parts = [f"{row['bearing_id']}={fmt(row['features'][feature])}" for row in result["bearings"]]
        return f"In the latest {result['experiment']} records, {feature} values are: " + "; ".join(parts) + "."
    if action == "rank_health":
        selected = result["selected"]
        if result["order"] == "desc":
            return f"The bearing closest to failure / highest latest health_index in scope {result['scope']} is {selected['experiment']}/{selected['bearing_id']} with health_index={fmt(selected['health_index'])} and state={selected['health_state']}."
        return f"The healthiest bearing in scope {result['scope']} is {selected['experiment']}/{selected['bearing_id']} with the lowest latest health_index={fmt(selected['health_index'])} and state={selected['health_state']}."
    if action == "state_check":
        return f"{yes_no(result['answer_boolean'])}. The latest {result['experiment']}/{result['bearing_id']} state is {result['health_state']} with health_index={fmt(result['health_index'])}."
    if action == "filter_state":
        if result["matches"]:
            names = ", ".join(f"{row['experiment']}/{row['bearing_id']}" for row in result["matches"])
        else:
            names = "none"
        return f"The latest bearings classified as {result['state']} are: {names}. Count={result['count']}."
    if action == "states_all_bearings":
        parts = [f"{row['bearing_id']}={row['health_state']} (health_index={fmt(row['health_index'])})" for row in result["bearings"]]
        return f"The latest health states in {result['experiment']} are: " + "; ".join(parts) + "."
    if action == "trend":
        if spec["metric"] == "health_index":
            return f"Yes. {result['experiment']}/{result['bearing_id']} shows {result['trend']}: mean {result['metric']} changes from {fmt(result['early_mean'])} to {fmt(result['late_mean'])}, delta={fmt(result['delta'])}."
        direction = "increases" if result["delta"] > 0 else "decreases" if result["delta"] < 0 else "is mostly stable"
        return f"The {result['metric']} trend for {result['experiment']}/{result['bearing_id']} {direction}: early mean={fmt(result['early_mean'])}, late mean={fmt(result['late_mean'])}, delta={fmt(result['delta'])}, latest_state={result['latest_health_state']}."
    if action == "near_end_trend":
        direction = "increases" if result["near_end_delta"] > 0 else "decreases" if result["near_end_delta"] < 0 else "is mostly stable"
        return f"Near the end, {result['metric']} for {result['experiment']}/{result['bearing_id']} {direction}: previous-window mean={fmt(result['previous_window_mean'])}, late-window mean={fmt(result['late_window_mean'])}, delta={fmt(result['near_end_delta'])}."
    if action == "rank_trend":
        selected = result["selected"]
        return f"The strongest {result['metric']} degradation trend in {result['experiment']} is {selected['bearing_id']}, with delta={fmt(selected['delta'])} from early mean {fmt(selected['early_mean'])} to late mean {fmt(selected['late_mean'])}."
    if action == "compare_trends":
        parts = [f"{row['bearing_id']}: {row['trend']} (delta={fmt(row['delta'])})" for row in result["trend_comparison"]]
        return f"The {result['metric']} trends in {result['experiment']} rank as: " + "; ".join(parts) + "."
    if action == "first_degradation":
        if not result["found"]:
            return f"No clear degradation state was found for {result['experiment']}/{result['bearing_id']} in the available sequence."
        return f"{result['experiment']}/{result['bearing_id']} first shows clear degradation at sequence_index {result['sequence_index']} ({result['timestamp']}), with state={result['health_state']} and health_index={fmt(result['health_index'])}."
    if action == "earliest_feature_change":
        earliest = result["earliest_feature_change"]
        if earliest is None:
            return f"No monitored degradation feature crosses the z-score threshold {result['z_threshold']} for {result['experiment']}/{result['bearing_id']}."
        return f"The earliest feature change for {result['experiment']}/{result['bearing_id']} is {earliest['feature']} at sequence_index {earliest['sequence_index']} ({earliest['timestamp']}), z_score={fmt(earliest['z_score'])}, value={fmt(earliest['value'])}."
    if action == "compare_metric_trends":
        parts = [f"{row['metric']}: early={fmt(row['early_mean'])}, late={fmt(row['late_mean'])}, delta={fmt(row['delta'])}" for row in result["trends"]]
        return f"For {result['experiment']}/{result['bearing_id']}, the trend comparison is: " + "; ".join(parts) + "."
    raise ValueError(f"No reference answer template for {action}")


def expected_query(spec: dict[str, Any]) -> dict[str, Any]:
    action = spec["action"]
    query = OrderedDict()
    query["action"] = action
    for key in ["experiment", "bearing_id", "features", "feature", "metric", "scope", "order", "window", "baseline_window", "z_threshold", "state", "positive_states", "state_order", "metrics"]:
        if key in spec:
            query[key] = spec[key]
    query["time_policy"] = "full_available_sequence" if "trend" in action or action in {"first_degradation", "earliest_feature_change", "compare_metric_trends"} else TIME_POLICY
    query["sequence_index"] = None
    return dict(query)


def grading_spec(spec: dict[str, Any]) -> dict[str, Any]:
    checks = ["route", "dt_action"]
    for key in ["experiment", "bearing_id", "features", "feature", "metric", "scope", "state"]:
        if key in spec and spec[key] is not None:
            checks.append(key)
    checks.append("answer_factuality")
    return {
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "required_checks": checks,
        "notes": "Numerical answers should match expected_result within tolerance after rounding; state/ranking answers should match exact labels and selected bearing IDs.",
    }


def load_dt_questions(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected agent question file to be a JSON list.")
    return [row for row in rows if row.get("expected_route") == "dt_only"]


def build_benchmark(db_path: Path, question_path: Path) -> dict[str, Any]:
    questions = load_dt_questions(question_path)
    items = []
    missing = [row["id"] for row in questions if row["id"] not in QUESTION_SPECS]
    if missing:
        raise KeyError(f"Missing QUESTION_SPECS entries: {missing}")
    with connect(db_path) as conn:
        for row in questions:
            spec = QUESTION_SPECS[row["id"]]
            result = build_expected(conn, spec)
            item = {
                "id": row["id"],
                "question": row["question"],
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "expected_route": "dt_only",
                "expected_tools": ["dt"],
                "expected_dt_query": expected_query(spec),
                "expected_result": round_float(result),
                "reference_answer": reference_answer(row["id"], row["question"], spec, result),
                "grading": grading_spec(spec),
            }
            items.append(item)
    return {
        "benchmark": {
            "name": "ims_dt_benchmark_v1",
            "version": "1.0",
            "description": "DT-only benchmark for IMS bearing digital-twin question answering.",
            "source_db": str(db_path),
            "question_source": str(question_path),
            "generation_method": "Independent SQLite queries over bearing_snapshots; does not call DT high-level query functions.",
            "time_policy": TIME_POLICY,
            "time_policy_description": TIME_POLICY_DESCRIPTION,
            "trend_policy": {
                "window": TREND_WINDOW,
                "health_index_delta_thresholds": {
                    "strongly_increasing_degradation": ">= 0.20",
                    "increasing_degradation": ">= 0.05 and < 0.20",
                    "mostly_stable": "between -0.05 and 0.05",
                    "decreasing_or_recovering": "<= -0.05",
                },
                "earliest_feature_change": f"First record after the baseline window where feature z-score >= {Z_THRESHOLD}, using the first {TREND_WINDOW} records as baseline.",
            },
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "item_count": len(items),
            "subcategory_counts": {
                "feature": sum(1 for item in items if item["subcategory"] == "feature"),
                "state": sum(1 for item in items if item["subcategory"] == "state"),
                "trend": sum(1 for item in items if item["subcategory"] == "trend"),
            },
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible IMS DT-only QA benchmark from SQLite facts.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmark = build_benchmark(args.db, args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {benchmark['benchmark']['item_count']} DT benchmark items -> {args.output}")


if __name__ == "__main__":
    main()
