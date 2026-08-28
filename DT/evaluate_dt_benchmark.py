#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from bearing_agent import DEFAULT_DB, answer_question, resolve_ollama_openai_base  # noqa: E402

DEFAULT_BENCHMARK = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_v1.json"
DEFAULT_RESULTS = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_results.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_summary.csv"

NUMERIC_KEYS = {
    "health_index", "rms", "kurtosis", "crest_factor", "impulse_factor", "clearance_factor",
    "high_band_energy_ratio", "spectral_entropy", "dominant_frequency", "delta", "early_mean",
    "late_mean", "near_end_delta", "previous_window_mean", "late_window_mean", "value", "z_score",
}

ACTION_EQUIVALENCE = {
    "features": {"features"},
    "latest_state": {"latest_state"},
    "compare_feature": {"compare"},
    "rank_feature": {"compare", "top_anomalies"},
    "feature_abnormality": {"features"},
    "features_all_bearings": {"compare"},
    "rank_health": {"compare", "top_anomalies"},
    "state_check": {"latest_state", "features"},
    "filter_state": {"top_anomalies", "compare"},
    "states_all_bearings": {"compare"},
    "trend": {"trend"},
    "near_end_trend": {"trend"},
    "rank_trend": {"compare", "top_anomalies"},
    "compare_trends": {"compare", "top_anomalies"},
    "first_degradation": {"trend"},
    "earliest_feature_change": {"trend"},
    "compare_metric_trends": {"trend"},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data: Any, path: list[Any]) -> Any:
    cur = data
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return None
            cur = cur[part]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
    return cur


def normalize_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    av = normalize_float(actual)
    ev = normalize_float(expected)
    if av is None or ev is None:
        return False
    return abs(av - ev) <= tolerance


def text_contains_value(text: str, value: Any, tolerance: float) -> bool:
    expected = normalize_float(value)
    if expected is None:
        return False
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return any(abs(num - expected) <= tolerance or abs(num - round(expected, 3)) <= tolerance for num in numbers)


def flatten_expected_query(expected: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in expected.items() if v is not None and k not in {"time_policy", "sequence_index"}}


def actual_query_from_result(result: dict[str, Any]) -> dict[str, Any]:
    route = result.get("route") or {}
    entities = route.get("dt_entities") or {}
    dt_result = result.get("dt_result") or {}
    actual = {
        "route": route.get("route"),
        "action": dt_result.get("action"),
        "experiment": entities.get("experiment"),
        "bearing_id": entities.get("bearing_id"),
        "metric": entities.get("metric"),
    }
    payload = dt_result.get("result") if isinstance(dt_result, dict) else None
    if isinstance(payload, dict):
        actual["result_experiment"] = payload.get("experiment")
        actual["result_bearing_id"] = payload.get("bearing_id")
        actual["time_policy"] = payload.get("time_policy")
        if "features" in payload and isinstance(payload["features"], dict):
            actual["features"] = list(payload["features"].keys())
    return {k: v for k, v in actual.items() if v is not None}


def compare_query(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    errors = []
    expected_action = expected.get("action")
    actual_action = actual.get("action")
    if actual.get("route") != "dt_only":
        errors.append(f"route expected dt_only got {actual.get('route')}")
    if expected_action:
        allowed = ACTION_EQUIVALENCE.get(expected_action, {expected_action})
        if actual_action not in allowed:
            errors.append(f"action expected {expected_action} compatible {sorted(allowed)} got {actual_action}")
    for key in ["experiment", "bearing_id"]:
        ev = expected.get(key)
        if ev is not None:
            av = actual.get(key) or actual.get(f"result_{key}")
            if av != ev:
                errors.append(f"{key} expected {ev} got {av}")
    expected_features = expected.get("features")
    if expected_features and actual.get("features"):
        missing = sorted(set(expected_features) - set(actual["features"]))
        if missing:
            errors.append(f"features missing {missing}")
    expected_metric = expected.get("metric") or expected.get("feature")
    actual_metric = actual.get("metric")
    if expected_metric and actual_metric and actual_metric != expected_metric:
        errors.append(f"metric expected {expected_metric} got {actual_metric}")
    return {"passed": not errors, "errors": errors, "actual_dt_query": actual}


def result_payload(agent_result: dict[str, Any]) -> dict[str, Any]:
    dt_result = agent_result.get("dt_result") or {}
    payload = dt_result.get("result") if isinstance(dt_result, dict) else {}
    return payload if isinstance(payload, dict) else {}


def match_snapshot(expected: dict[str, Any], actual: dict[str, Any], tolerance: float, answer_text: str) -> list[str]:
    errors = []
    for key in ["experiment", "bearing_id", "health_state", "sequence_index"]:
        ev = expected.get(key)
        if ev is not None and actual.get(key) != ev:
            errors.append(f"{key} expected {ev} got {actual.get(key)}")
    if expected.get("timestamp") and actual.get("timestamp") and actual.get("timestamp") != expected.get("timestamp"):
        errors.append(f"timestamp expected {expected.get('timestamp')} got {actual.get('timestamp')}")
    if "health_index" in expected and not close_enough(actual.get("health_index"), expected.get("health_index"), tolerance):
        if not text_contains_value(answer_text, expected.get("health_index"), tolerance):
            errors.append(f"health_index expected {expected.get('health_index')} got {actual.get('health_index')}")
    for key, ev in (expected.get("features") or {}).items():
        av = None
        if isinstance(actual.get("features"), dict):
            av = actual["features"].get(key)
        if av is None:
            av = actual.get(key)
        if not close_enough(av, ev, tolerance) and not text_contains_value(answer_text, ev, tolerance):
            errors.append(f"feature {key} expected {ev} got {av}")
    return errors


def find_record(records: list[dict[str, Any]], experiment: Optional[str], bearing_id: Optional[str]) -> Optional[dict[str, Any]]:
    for row in records:
        if experiment is not None and row.get("experiment") != experiment:
            continue
        if bearing_id is not None and row.get("bearing_id") != bearing_id:
            continue
        return row
    return None


def compare_answer(item: dict[str, Any], agent_result: dict[str, Any]) -> dict[str, Any]:
    expected_query = item["expected_dt_query"]
    expected = item["expected_result"]
    tolerance = float(item.get("grading", {}).get("numeric_tolerance", 1e-3))
    actual = result_payload(agent_result)
    answer_text = str(agent_result.get("answer") or "")
    action = expected_query.get("action")
    errors: list[str] = []

    if not actual:
        return {"passed": False, "errors": ["missing actual dt_result.result"]}

    if action in {"features", "latest_state", "feature_abnormality", "state_check"}:
        errors.extend(match_snapshot(expected, actual, tolerance, answer_text))
        if "answer_boolean" in expected:
            ev = expected["answer_boolean"]
            answer_lower = answer_text.lower()
            if ev and "yes" not in answer_lower and actual.get("health_state") not in answer_lower:
                errors.append("answer should affirm expected boolean result")
            if not ev and "no" not in answer_lower and actual.get("health_state") not in answer_lower:
                errors.append("answer should negate expected boolean result")
        if "is_abnormal" in expected:
            if expected["is_abnormal"] and "abnormal" not in answer_text.lower() and "degradation" not in answer_text.lower():
                errors.append("answer should indicate abnormality/degradation")

    elif action == "compare_feature":
        bearings = actual.get("bearings") or []
        target = find_record(bearings, expected["target"]["experiment"], expected["target"]["bearing_id"])
        if target is None:
            errors.append("target bearing missing in compare result")
        else:
            metric = expected["feature"]
            if not close_enough(target.get(metric), expected["target"].get(metric), tolerance):
                errors.append(f"target {metric} expected {expected['target'].get(metric)} got {target.get(metric)}")
        if expected.get("is_highest") and expected["target"]["bearing_id"] not in answer_text:
            errors.append("answer missing highest target bearing")

    elif action in {"rank_feature", "rank_health"}:
        selected = expected.get("winner") or expected.get("selected")
        if selected is None:
            errors.append("expected selected/winner missing")
        else:
            if action == "rank_health" and actual.get("most_anomalous"):
                actual_selected = actual.get("most_anomalous")
            elif actual.get("anomalies"):
                actual_selected = actual["anomalies"][0]
            elif actual.get("bearings"):
                metric = expected_query.get("feature") or "health_index"
                reverse = expected_query.get("order", "desc") == "desc"
                actual_selected = sorted(actual["bearings"], key=lambda row: float(row.get(metric) or 0.0), reverse=reverse)[0]
            else:
                actual_selected = {}
            for key in ["experiment", "bearing_id"]:
                if actual_selected.get(key) != selected.get(key):
                    errors.append(f"selected {key} expected {selected.get(key)} got {actual_selected.get(key)}")
            metric = expected_query.get("feature") or "health_index"
            if metric in selected and not close_enough(actual_selected.get(metric), selected.get(metric), tolerance):
                errors.append(f"selected {metric} expected {selected.get(metric)} got {actual_selected.get(metric)}")

    elif action == "features_all_bearings":
        actual_bearings = actual.get("bearings") or []
        for expected_bearing in expected.get("bearings", []):
            actual_bearing = find_record(actual_bearings, expected_bearing.get("experiment"), expected_bearing.get("bearing_id"))
            if actual_bearing is None:
                errors.append(f"missing bearing {expected_bearing.get('bearing_id')}")
                continue
            errors.extend(match_snapshot(expected_bearing, actual_bearing, tolerance, answer_text))

    elif action == "filter_state":
        state = expected.get("state")
        text = answer_text.lower()
        for match in expected.get("matches", []):
            if match["bearing_id"] not in text or match["experiment"] not in text:
                errors.append(f"answer missing state match {match['experiment']}/{match['bearing_id']}")
        if state and state not in text:
            errors.append(f"answer missing state label {state}")

    elif action == "states_all_bearings":
        text = answer_text.lower()
        for row in expected.get("bearings", []):
            if row["bearing_id"] not in text or row["health_state"] not in text:
                errors.append(f"answer missing {row['bearing_id']}={row['health_state']}")

    elif action in {"trend", "near_end_trend"}:
        if actual.get("experiment") != expected.get("experiment") or actual.get("bearing_id") != expected.get("bearing_id"):
            errors.append("trend entity mismatch")
        expected_direction = expected.get("trend") or expected.get("near_end_trend")
        actual_direction = actual.get("trend")
        if action == "trend" and expected_query.get("metric") == "health_index" and actual_direction != expected_direction:
            errors.append(f"trend expected {expected_direction} got {actual_direction}")
        metric = expected_query.get("metric")
        if metric == "health_index":
            for key, actual_key in [("early_mean", "early_health_index_mean"), ("late_mean", "late_health_index_mean"), ("delta", "health_index_delta")]:
                if key in expected and not close_enough(actual.get(actual_key), expected.get(key), tolerance):
                    errors.append(f"{key} expected {expected.get(key)} got {actual.get(actual_key)}")
        else:
            # Current high-level DT trend tool returns health-index trends only. For non-health metrics,
            # require the final answer to at least state the expected direction/metric entity.
            if metric and metric.replace("_", " ") not in answer_text.lower().replace("_", " "):
                errors.append(f"answer missing metric {metric}")
            delta = expected.get("delta") or expected.get("near_end_delta")
            if delta is not None:
                direction_word = "increase" if delta > 0 else "decrease" if delta < 0 else "stable"
                if direction_word not in answer_text.lower() and expected_direction not in answer_text:
                    errors.append(f"answer missing expected trend direction {direction_word}")

    elif action == "compare_trends":
        text = answer_text.lower()
        for row in expected.get("trend_comparison", []):
            if row["bearing_id"] not in text:
                errors.append(f"answer missing trend bearing {row['bearing_id']}")

    elif action == "rank_trend":
        selected = expected.get("selected") or {}
        if selected.get("bearing_id") not in answer_text:
            errors.append(f"answer missing strongest trend bearing {selected.get('bearing_id')}")

    elif action == "compare_metric_trends":
        text = answer_text.lower()
        for row in expected.get("trends", []):
            metric = row["metric"]
            if metric not in text and metric.replace("_", " ") not in text:
                errors.append(f"answer missing metric trend {metric}")

    elif action == "first_degradation":
        text = answer_text.lower()
        if expected.get("found"):
            if str(expected.get("sequence_index")) not in text and expected.get("timestamp", "").lower() not in text:
                errors.append("answer missing first degradation sequence/timestamp")
            if expected.get("health_state") not in text:
                errors.append("answer missing first degradation state")

    elif action == "earliest_feature_change":
        earliest = expected.get("earliest_feature_change") or {}
        text = answer_text.lower()
        if earliest.get("feature") not in text and earliest.get("feature", "").replace("_", " ") not in text:
            errors.append(f"answer missing earliest feature {earliest.get('feature')}")
        if str(earliest.get("sequence_index")) not in text and earliest.get("timestamp", "").lower() not in text:
            errors.append("answer missing earliest feature sequence/timestamp")

    else:
        errors.append(f"unsupported action in answer evaluator: {action}")

    return {"passed": not errors, "errors": errors}


def evaluate_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = answer_question(
        question=item["question"],
        db_path=args.db,
        skip_rag=True,
        generate_answer=args.answer,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        router_mode=args.router_mode,
        router_model=args.router_model,
        router_confidence_threshold=args.router_confidence_threshold,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key,
    )
    actual_query = actual_query_from_result(result)
    query_eval = compare_query(item["expected_dt_query"], actual_query)
    answer_eval = compare_answer(item, result)
    return {
        "id": item["id"],
        "question": item["question"],
        "subcategory": item.get("subcategory"),
        "expected_dt_query": item["expected_dt_query"],
        "actual_dt_query": query_eval["actual_dt_query"],
        "query_passed": query_eval["passed"],
        "query_errors": query_eval["errors"],
        "answer_passed": answer_eval["passed"],
        "answer_errors": answer_eval["errors"],
        "joint_passed": query_eval["passed"] and answer_eval["passed"],
        "reference_answer": item.get("reference_answer"),
        "actual_answer": result.get("answer"),
        "actual_route": (result.get("route") or {}).get("route"),
        "actual_dt_result": result.get("dt_result"),
        "agent_errors": result.get("errors") or [],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sub.setdefault(row.get("subcategory") or "unknown", []).append(row)
    summary = {
        "total": total,
        "query_accuracy": sum(1 for row in rows if row["query_passed"]) / total if total else 0.0,
        "answer_accuracy": sum(1 for row in rows if row["answer_passed"]) / total if total else 0.0,
        "joint_accuracy": sum(1 for row in rows if row["joint_passed"]) / total if total else 0.0,
        "by_subcategory": {},
    }
    for sub, sub_rows in by_sub.items():
        n = len(sub_rows)
        summary["by_subcategory"][sub] = {
            "total": n,
            "query_accuracy": sum(1 for row in sub_rows if row["query_passed"]) / n,
            "answer_accuracy": sum(1 for row in sub_rows if row["answer_passed"]) / n,
            "joint_accuracy": sum(1 for row in sub_rows if row["joint_passed"]) / n,
        }
    return summary


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "total", "query_accuracy", "answer_accuracy", "joint_accuracy"])
        writer.writerow(["overall", summary["total"], summary["query_accuracy"], summary["answer_accuracy"], summary["joint_accuracy"]])
        for sub, values in summary["by_subcategory"].items():
            writer.writerow([sub, values["total"], values["query_accuracy"], values["answer_accuracy"], values["joint_accuracy"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DT-only benchmark on query accuracy and answer factual accuracy.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--router-model", default="gpt-4o-mini")
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--answer", action="store_true", help="Use LLM natural-language answer instead of deterministic answer.")
    parser.add_argument("--llm-model", default="llama3.3:70b")
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    args = parser.parse_args()

    if args.llm_model.lower().startswith("llama"):
        args.llm_api_base = args.llm_api_base or resolve_ollama_openai_base()
        args.llm_api_key = args.llm_api_key or os.environ.get("OLLAMA_API_KEY")

    data = load_json(args.benchmark)
    items = data.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]

    rows = []
    for idx, item in enumerate(items, start=1):
        print(f"[DT BENCH] {idx}/{len(items)} {item['id']}: {item['question']}", flush=True)
        rows.append(evaluate_item(item, args))

    summary = summarize(rows)
    payload = {
        "summary": summary,
        "rows": rows,
        "benchmark": str(args.benchmark),
        "db": str(args.db),
        "router_mode": args.router_mode,
        "answer_mode": "llm" if args.answer else "deterministic",
        "llm_model": args.llm_model if args.answer else None,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary, args.summary)
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] Results -> {args.results}", flush=True)
    print(f"[OK] Summary -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
