#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB = Path(__file__).resolve().parent / "outputs" / "ims_dt.db"
DEFAULT_TIME_POLICY = "latest_available_record"
TIME_POLICY_DESCRIPTION = (
    "For IMS DT queries, current/latest/now means the latest available timestamp "
    "in the offline IMS digital-twin database for the requested experiment and bearing, "
    "not the real-world current date."
)

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


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def normalize_bearing_id(bearing_id: str) -> str:
    text = str(bearing_id).strip().lower().replace(" ", "_").replace("-", "_")
    if text.isdigit():
        return f"bearing_{text}"
    if text.startswith("bearing") and "_" not in text:
        suffix = text.replace("bearing", "")
        if suffix.isdigit():
            return f"bearing_{suffix}"
    return text


def get_experiments(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "select experiment, channels, failure_description from experiments order by experiment"
        ).fetchall()
    return [dict(row) for row in rows]


def add_time_policy(result: dict[str, Any], resolved_time_mode: str = DEFAULT_TIME_POLICY) -> dict[str, Any]:
    result["time_policy"] = DEFAULT_TIME_POLICY
    result["time_policy_description"] = TIME_POLICY_DESCRIPTION
    result["resolved_time_mode"] = resolved_time_mode
    return result


def get_latest_state(db_path: str | Path, experiment: str, bearing_id: str) -> dict[str, Any]:
    """Return the latest available IMS DT snapshot for one bearing.

    In this offline run-to-failure dataset, user terms such as current, latest,
    now, and at present resolve to the latest stored record for the requested
    experiment/bearing.
    """
    bearing_id = normalize_bearing_id(bearing_id)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            select * from bearing_snapshots
            where experiment = ? and bearing_id = ?
            order by sequence_index desc
            limit 1
            """,
            (experiment, bearing_id),
        ).fetchone()
    result = row_to_dict(row)
    if result is None:
        raise ValueError(f"No latest state found for {experiment}/{bearing_id}")
    add_time_policy(result)
    result["summary"] = (
        f"Interpreting current/latest as the latest available IMS DT record, "
        f"{experiment}/{bearing_id} is classified as {result['health_state']} "
        f"at timestamp={result.get('timestamp')} and sequence_index={result.get('sequence_index')} "
        f"with health_index={result['health_index']:.3f}."
    )
    return result


def get_bearing_features(
    db_path: str | Path,
    experiment: str,
    bearing_id: str,
    sequence_index: Optional[int] = None,
    feature_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    bearing_id = normalize_bearing_id(bearing_id)
    feature_names = feature_names or CORE_FEATURES
    with connect(db_path) as conn:
        if sequence_index is None:
            row = conn.execute(
                """
                select * from bearing_snapshots
                where experiment = ? and bearing_id = ?
                order by sequence_index desc
                limit 1
                """,
                (experiment, bearing_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                select * from bearing_snapshots
                where experiment = ? and bearing_id = ? and sequence_index = ?
                limit 1
                """,
                (experiment, bearing_id, sequence_index),
            ).fetchone()
    record = row_to_dict(row)
    if record is None:
        raise ValueError(f"No feature snapshot found for {experiment}/{bearing_id}")
    features = {name: record.get(name) for name in feature_names if name in record.keys()}
    resolved_time_mode = DEFAULT_TIME_POLICY if sequence_index is None else "specified_sequence_index"
    return add_time_policy({
        "experiment": experiment,
        "bearing_id": bearing_id,
        "timestamp": record.get("timestamp"),
        "sequence_index": record.get("sequence_index"),
        "health_state": record.get("health_state"),
        "features": features,
        "evidence": record.get("evidence", ""),
    }, resolved_time_mode=resolved_time_mode)


def compare_bearings(db_path: str | Path, experiment: str, sequence_index: Optional[int] = None) -> dict[str, Any]:
    with connect(db_path) as conn:
        if sequence_index is None:
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
                order by s.health_index desc
                """,
                (experiment, experiment),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select * from bearing_snapshots
                where experiment = ? and sequence_index = ?
                order by health_index desc
                """,
                (experiment, sequence_index),
            ).fetchall()
    records = [dict(row) for row in rows]
    most_anomalous = records[0] if records else None
    resolved_time_mode = DEFAULT_TIME_POLICY if sequence_index is None else "specified_sequence_index"
    return add_time_policy({
        "experiment": experiment,
        "sequence_index": sequence_index,
        "bearings": records,
        "most_anomalous": most_anomalous,
        "summary": (
            f"Interpreting current/latest as the latest available IMS DT record per bearing, "
            f"the most anomalous bearing in {experiment} is {most_anomalous['bearing_id']} "
            f"with health_index={most_anomalous['health_index']:.3f} and state={most_anomalous['health_state']}."
            if most_anomalous else f"No bearing snapshots found for {experiment}."
        ),
    }, resolved_time_mode=resolved_time_mode)


def get_health_trend(db_path: str | Path, experiment: str, bearing_id: str, window: int = 100) -> dict[str, Any]:
    bearing_id = normalize_bearing_id(bearing_id)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            select sequence_index, timestamp, health_index, health_state, rms, kurtosis,
                   crest_factor, high_band_energy_ratio, spectral_entropy, evidence
            from bearing_snapshots
            where experiment = ? and bearing_id = ?
            order by sequence_index
            """,
            (experiment, bearing_id),
        ).fetchall()
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"No trend records found for {experiment}/{bearing_id}")

    early = records[: max(1, min(window, len(records)))]
    late = records[-max(1, min(window, len(records))):]
    early_mean = sum(r["health_index"] for r in early) / len(early)
    late_mean = sum(r["health_index"] for r in late) / len(late)
    delta = late_mean - early_mean
    if delta >= 0.20:
        trend = "strongly_increasing_degradation"
    elif delta >= 0.05:
        trend = "increasing_degradation"
    elif delta <= -0.05:
        trend = "decreasing_or_recovering"
    else:
        trend = "mostly_stable"

    return add_time_policy({
        "experiment": experiment,
        "bearing_id": bearing_id,
        "samples": len(records),
        "early_health_index_mean": early_mean,
        "late_health_index_mean": late_mean,
        "health_index_delta": delta,
        "latest_state": records[-1]["health_state"],
        "trend": trend,
        "latest_evidence": records[-1].get("evidence", ""),
        "summary": (
            f"Using the full available IMS DT sequence, {experiment}/{bearing_id} shows {trend}; "
            f"the mean health_index changed from {early_mean:.3f} to {late_mean:.3f} over the run. "
            f"The latest state refers to the latest available record in the sequence."
        ),
    }, resolved_time_mode="full_available_sequence")


def get_top_anomalies(db_path: str | Path, experiment: Optional[str] = None, top_k: int = 5) -> dict[str, Any]:
    params: tuple[Any, ...]
    where = ""
    if experiment:
        where = "where experiment = ?"
        params = (experiment,)
    else:
        params = ()
    query = f"""
        select experiment, bearing_id, timestamp, sequence_index, health_index, health_state,
               rms, kurtosis, crest_factor, high_band_energy_ratio, spectral_entropy, evidence
        from bearing_snapshots
        {where}
        order by health_index desc
        limit ?
    """
    with connect(db_path) as conn:
        rows = conn.execute(query, params + (top_k,)).fetchall()
    records = [dict(row) for row in rows]
    return add_time_policy({
        "experiment": experiment or "all",
        "top_k": top_k,
        "anomalies": records,
        "summary": (
            f"Across the selected IMS DT records, the top anomaly is {records[0]['experiment']}/{records[0]['bearing_id']} "
            f"at sequence_index={records[0]['sequence_index']} with health_index={records[0]['health_index']:.3f}."
            if records else "No anomaly records found."
        ),
    }, resolved_time_mode="all_available_records_ranked_by_health_index")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the IMS lightweight DT SQLite database.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to ims_dt.db")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("experiments", help="List IMS experiments.")

    p_latest = sub.add_parser("latest-state", help="Get the latest state of one bearing.")
    p_latest.add_argument("--experiment", required=True)
    p_latest.add_argument("--bearing", required=True)

    p_features = sub.add_parser("features", help="Get bearing features at latest or selected sequence index.")
    p_features.add_argument("--experiment", required=True)
    p_features.add_argument("--bearing", required=True)
    p_features.add_argument("--sequence-index", type=int, default=None)

    p_compare = sub.add_parser("compare", help="Compare bearings in one experiment.")
    p_compare.add_argument("--experiment", required=True)
    p_compare.add_argument("--sequence-index", type=int, default=None)

    p_trend = sub.add_parser("trend", help="Summarize a bearing health trend.")
    p_trend.add_argument("--experiment", required=True)
    p_trend.add_argument("--bearing", required=True)
    p_trend.add_argument("--window", type=int, default=100)

    p_top = sub.add_parser("top-anomalies", help="Return top anomalous snapshots.")
    p_top.add_argument("--experiment", default=None)
    p_top.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.command == "experiments":
        print_json(get_experiments(db_path))
    elif args.command == "latest-state":
        print_json(get_latest_state(db_path, args.experiment, args.bearing))
    elif args.command == "features":
        print_json(get_bearing_features(db_path, args.experiment, args.bearing, args.sequence_index))
    elif args.command == "compare":
        print_json(compare_bearings(db_path, args.experiment, args.sequence_index))
    elif args.command == "trend":
        print_json(get_health_trend(db_path, args.experiment, args.bearing, args.window))
    elif args.command == "top-anomalies":
        print_json(get_top_anomalies(db_path, args.experiment, args.top_k))


if __name__ == "__main__":
    main()
