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
DT_DIR = PROJECT_ROOT / "DT"
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))

import dt_llm_planner  # noqa: E402
from build_dt_benchmark import (  # noqa: E402
    CORE_FEATURES,
    DEFAULT_DB,
    DEFAULT_OUTPUT as DEFAULT_BENCHMARK,
    DEGRADATION_FEATURES,
    TREND_WINDOW,
    Z_THRESHOLD,
    build_expected,
    connect,
    round_float,
)

DEFAULT_RESULTS = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_results_llm_planner.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "DT" / "eval" / "dt_benchmark_summary_llm_planner.csv"

VALID_EXPERIMENTS = {"1st_test", "2nd_test", "3rd_test"}
VALID_BEARINGS = {"bearing_1", "bearing_2", "bearing_3", "bearing_4"}
VALID_ACTIONS = {
    "features",
    "latest_state",
    "compare_feature",
    "rank_feature",
    "feature_abnormality",
    "features_all_bearings",
    "rank_health",
    "state_check",
    "filter_state",
    "states_all_bearings",
    "trend",
    "near_end_trend",
    "rank_trend",
    "compare_trends",
    "first_degradation",
    "earliest_feature_change",
    "compare_metric_trends",
}

FEATURE_ALIASES = {
    "health index": "health_index",
    "health_index": "health_index",
    "rms": "rms",
    "root mean square": "rms",
    "kurtosis": "kurtosis",
    "crest factor": "crest_factor",
    "crest_factor": "crest_factor",
    "impulse factor": "impulse_factor",
    "impulse_factor": "impulse_factor",
    "clearance factor": "clearance_factor",
    "clearance_factor": "clearance_factor",
    "high band energy ratio": "high_band_energy_ratio",
    "high_band_energy_ratio": "high_band_energy_ratio",
    "spectral entropy": "spectral_entropy",
    "spectral_entropy": "spectral_entropy",
    "dominant frequency": "dominant_frequency",
    "dominant_frequency": "dominant_frequency",
}

ACTION_ALIASES = {
    "get_features": "features",
    "feature_query": "features",
    "query_features": "features",
    "state": "latest_state",
    "get_state": "latest_state",
    "latest_health_state": "latest_state",
    "compare": "compare_feature",
    "feature_compare": "compare_feature",
    "rank": "rank_feature",
    "ranking": "rank_feature",
    "abnormality": "feature_abnormality",
    "all_features": "features_all_bearings",
    "health_rank": "rank_health",
    "state_filter": "filter_state",
    "all_states": "states_all_bearings",
    "trend_query": "trend",
    "near_end": "near_end_trend",
    "trend_rank": "rank_trend",
    "trend_compare": "compare_trends",
    "first_degradation_time": "first_degradation",
    "early_feature_change": "earliest_feature_change",
    "metric_trend_compare": "compare_metric_trends",
}

NUMERIC_TOLERANCE = 1e-3


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_base_url(base: Optional[str], model: str) -> Optional[str]:
    if base:
        cleaned = base.rstrip("/")
    elif model.lower().startswith("llama"):
        raw = os.environ.get("OLLAMA_OPENAI_BASE") or os.environ.get("OLLAMA_BASE_URL")
        cleaned = raw.rstrip("/") if raw else None
    else:
        raw = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        cleaned = raw.rstrip("/") if raw else None
    if cleaned and model.lower().startswith("llama") and not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def resolve_api_key(explicit: Optional[str], model: str) -> str:
    if explicit:
        return explicit
    if model.lower().startswith("llama"):
        return os.environ.get("OLLAMA_API_KEY") or "ollama"
    return os.environ.get("OPENAI_API_KEY") or ""


def make_client(api_base: Optional[str], api_key: str, timeout: float):
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY", "timeout": timeout}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def strip_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a JSON object: {text[:300]}")
    return json.loads(text[start : end + 1])


def normalize_feature(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    key = re.sub(r"\s+", " ", key.replace("_", " "))
    return FEATURE_ALIASES.get(key, key.replace(" ", "_"))


def normalize_feature_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = re.split(r"[,;/]", values)
    elif isinstance(values, list):
        parts = values
    else:
        parts = [values]
    normalized = []
    for part in parts:
        feature = normalize_feature(part)
        if feature and feature not in normalized:
            normalized.append(feature)
    return normalized


def normalize_experiment(value: Any, question: str = "") -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw or raw in {"none", "null", "all"}:
        return None
    if "1st" in raw or raw == "first_test" or raw == "first":
        return "1st_test"
    if "2nd" in raw or raw == "second_test" or raw == "second":
        return "2nd_test"
    if "3rd" in raw or raw == "third_test" or raw == "third":
        return "3rd_test"
    return raw if raw in VALID_EXPERIMENTS else None


def normalize_bearing(value: Any, question: str = "") -> Optional[str]:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in {"none", "null", "all"}:
        return None
    match = re.search(r"bearing[_\s]*(\d)", raw)
    if match:
        candidate = f"bearing_{match.group(1)}"
        if candidate in VALID_BEARINGS:
            return candidate
    return raw if raw in VALID_BEARINGS else None


def normalize_action(value: Any, question: str, query: dict[str, Any]) -> Optional[str]:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in {"none", "null"}:
        return None
    action = ACTION_ALIASES.get(raw, raw)
    if action == "rank_feature" and (query.get("metric") == "health_index" or normalize_feature(query.get("feature")) == "health_index"):
        return "rank_health"
    return action if action in VALID_ACTIONS else action


def normalize_planned_query(raw: dict[str, Any], question: str) -> dict[str, Any]:
    query = dict(raw)
    action = normalize_action(query.get("action"), question, query)
    experiment = normalize_experiment(query.get("experiment"), question)
    bearing_id = normalize_bearing(query.get("bearing_id"), question)
    feature = normalize_feature(query.get("feature") or query.get("metric"))
    metric = normalize_feature(query.get("metric") or query.get("feature"))
    features = normalize_feature_list(query.get("features"))
    metrics = normalize_feature_list(query.get("metrics"))

    if not features and feature and action in {"features", "features_all_bearings"}:
        features = [feature]
    if not feature and features and action in {"rank_feature", "compare_feature", "feature_abnormality"}:
        feature = features[0]
    if not metric and features and action in {"trend", "near_end_trend", "rank_trend", "compare_trends"}:
        metric = features[0]

    order = str(query.get("order") or "").strip().lower()
    if order not in {"asc", "desc"}:
        order = None

    state = query.get("state")
    if state:
        state = str(state).strip().lower()

    scope = query.get("scope")
    if scope:
        scope = str(scope).strip()

    normalized = {
        "action": action,
        "experiment": experiment,
        "bearing_id": bearing_id,
        "features": features,
        "feature": feature,
        "metric": metric,
        "metrics": metrics,
        "scope": scope,
        "order": order,
        "window": int(query["window"]) if query.get("window") is not None else None,
        "baseline_window": int(query["baseline_window"]) if query.get("baseline_window") is not None else None,
        "z_threshold": float(query["z_threshold"]) if query.get("z_threshold") is not None else None,
        "state": state,
        "positive_states": query.get("positive_states"),
        "state_order": query.get("state_order"),
        "time_policy": "full_available_sequence" if action and ("trend" in action or action in {"first_degradation", "earliest_feature_change", "compare_metric_trends"}) else "latest_available_record",
        "reason": query.get("reason"),
    }
    return complete_schema_defaults({k: v for k, v in normalized.items() if v not in (None, [], "")})


def complete_schema_defaults(query: dict[str, Any]) -> dict[str, Any]:
    """Fill deterministic DT tool defaults without changing the model's semantic choices."""
    completed = dict(query)
    action = completed.get("action")

    if action in {"rank_feature", "rank_health", "filter_state"} and not completed.get("scope"):
        completed["scope"] = "latest_bearings_in_experiment" if completed.get("experiment") else "latest_bearings_all_experiments"

    if action in {"trend", "near_end_trend", "rank_trend", "compare_trends", "compare_metric_trends"}:
        completed.setdefault("window", TREND_WINDOW)

    if action == "earliest_feature_change":
        completed.setdefault("baseline_window", TREND_WINDOW)
        completed.setdefault("z_threshold", Z_THRESHOLD)

    if action == "state_check" and not completed.get("positive_states"):
        state = completed.get("state")
        if state in {"degradation", "failure_near"}:
            completed["positive_states"] = ["degradation", "failure_near"]
        else:
            completed["positive_states"] = ["early_degradation", "degradation", "failure_near"]

    if action == "first_degradation" and not completed.get("state_order"):
        completed["state_order"] = ["early_degradation", "degradation", "failure_near"]

    return completed


def planner_prompt(question: str) -> list[dict[str, str]]:
    valid_features = ", ".join(CORE_FEATURES)
    actions = ", ".join(sorted(VALID_ACTIONS))
    system = (
        "You are a digital-twin query planner for an IMS bearing SQLite database. "
        "Convert the user question into one structured DT query. Do not answer the question. "
        "Return strict JSON only. The benchmark uses offline IMS data: words like current, latest, or now mean latest_available_record, not today's date."
    )
    user = f"""
Valid experiments: 1st_test, 2nd_test, 3rd_test.
Valid bearings: bearing_1, bearing_2, bearing_3, bearing_4.
Valid features/metrics: {valid_features}.
Valid actions: {actions}.

DT health semantics:
- health_index is a degradation-severity indicator: larger health_index means more severe degradation and closer to failure.
- closest to failure / most degraded should rank health_index in descending order.
- healthiest / least degraded should rank health_index in ascending order.
- degraded means any non-normal degradation state: early_degradation, degradation, or failure_near.
- obvious degradation means degradation or failure_near, not early_degradation.
- first/begin to show clear degradation means the first timestamp/sequence where health_state enters early_degradation, degradation, or failure_near.
- feature trend change should monitor degradation-related vibration features: rms, kurtosis, crest_factor, high_band_energy_ratio, spectral_entropy.

Output JSON schema:
{{
  "action": "one valid action",
  "experiment": "1st_test | 2nd_test | 3rd_test | null",
  "bearing_id": "bearing_1 | bearing_2 | bearing_3 | bearing_4 | null",
  "features": ["feature names, for features/features_all_bearings"],
  "feature": "single feature, for compare/rank/abnormality",
  "metric": "single metric, for trend actions",
  "metrics": ["metrics, for compare_metric_trends"],
  "scope": "latest_bearings_in_experiment | latest_bearings_all_experiments | null",
  "order": "desc | asc | null",
  "state": "failure_near | early_degradation | degradation | normal | null",
  "window": 100,
  "time_policy": "latest_available_record | full_available_sequence",
  "reason": "short reason"
}}

Examples:
Question: What is the latest RMS of bearing_3 in 1st_test?
JSON: {{"action":"features","experiment":"1st_test","bearing_id":"bearing_3","features":["rms"],"time_policy":"latest_available_record","reason":"latest numeric feature query"}}

Question: Which bearing has the highest latest health index in 2nd_test?
JSON: {{"action":"rank_health","experiment":"2nd_test","scope":"latest_bearings_in_experiment","order":"desc","time_policy":"latest_available_record","reason":"rank latest bearing health states"}}

Question: Does the health index of bearing_3 in 1st_test increase over the run?
JSON: {{"action":"trend","experiment":"1st_test","bearing_id":"bearing_3","metric":"health_index","window":100,"time_policy":"full_available_sequence","reason":"trend over full available sequence"}}

Question: {question}
JSON:
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_llm_json(question: str, model: str, api_base: Optional[str], api_key: str, timeout: float) -> dict[str, Any]:
    client = make_client(api_base, api_key, timeout)
    response = client.chat.completions.create(
        model=model,
        messages=planner_prompt(question),
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return strip_json_object(content)


def execute_planned_query(conn, planned_query: dict[str, Any]) -> tuple[Optional[dict[str, Any]], list[str]]:
    action = planned_query.get("action")
    if action not in VALID_ACTIONS:
        return None, [f"invalid action: {action}"]
    spec = {k: v for k, v in planned_query.items() if k not in {"time_policy", "reason"}}
    try:
        result = build_expected(conn, spec)
        return round_float(result), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"execution failed: {type(exc).__name__}: {exc}"]


def answer_prompt(question: str, dt_query: dict[str, Any], dt_result: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You answer IMS bearing digital-twin questions. Use only the DT query and DT result. "
        "Do not invent data. Include the exact experiment, bearing, health state, sequence index, and relevant numeric values when present. "
        "For trend results, explicitly include early_mean, late_mean, and delta/near_end_delta if present. "
        "For state-filter results, explicitly include the requested state label. Answer in concise English."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"DT query JSON:\n{json.dumps(dt_query, ensure_ascii=False, indent=2)}\n\n"
        f"DT result JSON:\n{json.dumps(dt_result, ensure_ascii=False, indent=2)}\n\n"
        "Final answer:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(question: str, dt_query: dict[str, Any], dt_result: dict[str, Any], model: str, api_base: Optional[str], api_key: str, timeout: float) -> str:
    client = make_client(api_base, api_key, timeout)
    response = client.chat.completions.create(
        model=model,
        messages=answer_prompt(question, dt_query, dt_result),
        temperature=0,
    )
    return response.choices[0].message.content or ""


def judge_answer_prompt(item: dict[str, Any], answer: str, rule_errors: list[str]) -> list[dict[str, str]]:
    system = (
        "You are a strict but fair evaluator for IMS bearing digital-twin QA. "
        "Decide whether the candidate answer correctly answers the question using the reference answer and expected DT result. "
        "Accept equivalent wording, yes/no paraphrases, and reasonable numerical rounding. "
        "Fail answers that use the wrong experiment, bearing, feature, state, direction, ranking, or materially wrong value. "
        "Return strict JSON only."
    )
    user = (
        f"Question:\n{item['question']}\n\n"
        f"Expected DT query:\n{json.dumps(item.get('expected_dt_query'), ensure_ascii=False, indent=2)}\n\n"
        f"Expected DT result:\n{json.dumps(item.get('expected_result'), ensure_ascii=False, indent=2)}\n\n"
        f"Reference answer:\n{item.get('reference_answer') or ''}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Rule-based answer errors, if any:\n{json.dumps(rule_errors, ensure_ascii=False)}\n\n"
        "Output JSON schema:\n"
        "{\n"
        "  \"passed\": true or false,\n"
        "  \"score\": number from 0 to 1,\n"
        "  \"reason\": \"short explanation\",\n"
        "  \"missing_key_facts\": [\"...\"],\n"
        "  \"unsupported_or_wrong_claims\": [\"...\"]\n"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_answer_equivalence(
    item: dict[str, Any],
    answer: str,
    rule_errors: list[str],
    model: str,
    api_base: Optional[str],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    client = make_client(api_base, api_key, timeout)
    response = client.chat.completions.create(
        model=model,
        messages=judge_answer_prompt(item, answer, rule_errors),
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    verdict = strip_json_object(content)
    verdict["passed"] = bool(verdict.get("passed"))
    try:
        verdict["score"] = float(verdict.get("score", 0.0))
    except (TypeError, ValueError):
        verdict["score"] = 1.0 if verdict["passed"] else 0.0
    return verdict


def normalize_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    av = normalize_float(actual)
    ev = normalize_float(expected)
    return av is not None and ev is not None and abs(av - ev) <= tolerance


def text_contains_value(text: str, value: Any, tolerance: float) -> bool:
    expected = normalize_float(value)
    if expected is None:
        return False
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    rounded = round(expected, 3)
    return any(abs(num - expected) <= tolerance or abs(num - rounded) <= max(tolerance, 0.001) for num in nums)


def text_contains_feature(text: str, feature: str) -> bool:
    normalized_text = text.lower().replace("_", " ").replace("-", " ")
    normalized_feature = feature.lower().replace("_", " ").replace("-", " ")
    compact_text = re.sub(r"[\s_-]+", "", text.lower())
    compact_feature = re.sub(r"[\s_-]+", "", feature.lower())
    return normalized_feature in normalized_text or compact_feature in compact_text


def compare_query(expected: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if planned.get("action") != expected.get("action"):
        errors.append(f"action expected {expected.get('action')} got {planned.get('action')}")
    for key in ["experiment", "bearing_id", "feature", "metric", "scope", "order", "state"]:
        ev = expected.get(key)
        if ev is not None and planned.get(key) != ev:
            errors.append(f"{key} expected {ev} got {planned.get(key)}")
    for key in ["features", "metrics", "positive_states"]:
        ev = expected.get(key)
        if ev:
            pv = planned.get(key) or []
            if set(pv) != set(ev):
                errors.append(f"{key} expected {ev} got {pv}")
    if expected.get("window") is not None and int(planned.get("window", -1)) != int(expected["window"]):
        errors.append(f"window expected {expected.get('window')} got {planned.get('window')}")
    return {"passed": not errors, "errors": errors}


def require_text(text: str, value: Any, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if str(value).lower() not in text.lower():
        errors.append(f"answer missing {label}: {value}")


def require_value(text: str, value: Any, label: str, tolerance: float, errors: list[str]) -> None:
    if value is None:
        return
    if not text_contains_value(text, value, tolerance):
        errors.append(f"answer missing numeric {label}: {value}")


def require_snapshot_text(
    text: str,
    row: dict[str, Any],
    tolerance: float,
    errors: list[str],
    features: Optional[list[str]] = None,
    require_health_state: bool = True,
) -> None:
    require_text(text, row.get("experiment"), "experiment", errors)
    require_text(text, row.get("bearing_id"), "bearing_id", errors)
    if require_health_state:
        require_text(text, row.get("health_state"), "health_state", errors)
    require_text(text, row.get("sequence_index"), "sequence_index", errors)
    feature_map = row.get("features") if isinstance(row.get("features"), dict) else row
    for feature in features or []:
        if feature in feature_map:
            if not text_contains_feature(text, feature):
                errors.append(f"answer missing feature name {feature}")
            require_value(text, feature_map.get(feature), feature, tolerance, errors)


def compare_answer_text(item: dict[str, Any], answer: str) -> dict[str, Any]:
    expected_query = item["expected_dt_query"]
    expected = item["expected_result"]
    action = expected_query["action"]
    tolerance = float(item.get("grading", {}).get("numeric_tolerance", NUMERIC_TOLERANCE))
    text = answer or ""
    lower = text.lower()
    errors: list[str] = []

    if not text.strip():
        return {"passed": False, "errors": ["empty final answer"]}

    if action == "features":
        features = list((expected.get("features") or {}).keys())
        require_snapshot_text(text, expected, tolerance, errors, features, require_health_state=False)
    elif action == "latest_state":
        require_snapshot_text(text, expected, tolerance, errors, [])
        require_value(text, expected.get("health_index"), "health_index", tolerance, errors)
    elif action == "feature_abnormality":
        require_snapshot_text(text, expected, tolerance, errors, [expected.get("feature")])
        if expected.get("is_abnormal") and not any(word in lower for word in ["abnormal", "degradation", "non-normal"]):
            errors.append("answer should state abnormality/degradation")
    elif action == "state_check":
        require_snapshot_text(text, expected, tolerance, errors, [])
        if expected.get("answer_boolean") and "yes" not in lower:
            errors.append("answer should explicitly answer yes")
        if expected.get("answer_boolean") is False and "no" not in lower:
            errors.append("answer should explicitly answer no")
    elif action == "compare_feature":
        metric = expected["feature"]
        target = expected["target"]
        leader = expected["rankings"][0]
        require_text(text, target.get("bearing_id"), "target bearing", errors)
        require_value(text, target.get(metric), f"target {metric}", tolerance, errors)
        require_text(text, leader.get("bearing_id"), "leader bearing", errors)
        if expected.get("is_highest") and "yes" not in lower:
            errors.append("answer should say the target is the highest")
    elif action == "rank_feature":
        winner = expected.get("winner") or {}
        metric = expected.get("feature")
        require_text(text, winner.get("experiment"), "winner experiment", errors)
        require_text(text, winner.get("bearing_id"), "winner bearing", errors)
        require_value(text, winner.get(metric), f"winner {metric}", tolerance, errors)
    elif action == "features_all_bearings":
        feature = (expected.get("features") or [None])[0]
        for row in expected.get("bearings", []):
            require_text(text, row.get("bearing_id"), "bearing", errors)
            if feature:
                require_value(text, (row.get("features") or {}).get(feature), f"{row.get('bearing_id')} {feature}", tolerance, errors)
    elif action == "rank_health":
        selected = expected.get("selected") or {}
        require_text(text, selected.get("experiment"), "selected experiment", errors)
        require_text(text, selected.get("bearing_id"), "selected bearing", errors)
        require_text(text, selected.get("health_state"), "selected state", errors)
        require_value(text, selected.get("health_index"), "selected health_index", tolerance, errors)
    elif action == "filter_state":
        require_text(text, expected.get("state"), "state label", errors)
        for row in expected.get("matches", []):
            require_text(text, row.get("experiment"), "matched experiment", errors)
            require_text(text, row.get("bearing_id"), "matched bearing", errors)
    elif action == "states_all_bearings":
        for row in expected.get("bearings", []):
            require_text(text, row.get("bearing_id"), "bearing", errors)
            require_text(text, row.get("health_state"), f"{row.get('bearing_id')} state", errors)
    elif action in {"trend", "near_end_trend"}:
        require_text(text, expected.get("experiment"), "experiment", errors)
        require_text(text, expected.get("bearing_id"), "bearing", errors)
        metric = expected.get("metric")
        if metric and not text_contains_feature(text, metric):
            errors.append(f"answer missing metric {metric}")
        trend = expected.get("trend") or expected.get("near_end_trend")
        direction_words = ["increase", "increasing", "strongly"] if (expected.get("delta") or expected.get("near_end_delta") or 0) > 0 else ["decrease", "decreasing", "stable"]
        if trend and trend not in lower and not any(word in lower for word in direction_words):
            errors.append(f"answer missing trend direction {trend}")
        if "delta" in expected:
            require_value(text, expected.get("delta"), "delta", tolerance, errors)
        if "near_end_delta" in expected:
            require_value(text, expected.get("near_end_delta"), "near_end_delta", tolerance, errors)
    elif action == "rank_trend":
        selected = expected.get("selected") or {}
        require_text(text, selected.get("bearing_id"), "selected bearing", errors)
        require_value(text, selected.get("delta"), "selected trend delta", tolerance, errors)
    elif action == "compare_trends":
        for row in expected.get("trend_comparison", []):
            require_text(text, row.get("bearing_id"), "trend bearing", errors)
            require_value(text, row.get("delta"), f"{row.get('bearing_id')} delta", tolerance, errors)
    elif action == "first_degradation":
        if expected.get("found"):
            require_text(text, expected.get("sequence_index"), "first degradation sequence_index", errors)
            require_text(text, expected.get("health_state"), "first degradation state", errors)
            require_value(text, expected.get("health_index"), "first degradation health_index", tolerance, errors)
    elif action == "earliest_feature_change":
        earliest = expected.get("earliest_feature_change") or {}
        require_text(text, earliest.get("feature"), "earliest feature", errors)
        require_text(text, earliest.get("sequence_index"), "earliest feature sequence_index", errors)
        require_value(text, earliest.get("value"), "earliest feature value", tolerance, errors)
    elif action == "compare_metric_trends":
        for row in expected.get("trends", []):
            metric = row.get("metric")
            if metric and not text_contains_feature(text, metric):
                errors.append(f"answer missing metric {metric}")
            require_value(text, row.get("delta"), f"{metric} delta", tolerance, errors)
    else:
        errors.append(f"unsupported answer action {action}")

    return {"passed": not errors, "errors": errors}


def evaluate_item(
    item: dict[str, Any],
    args: argparse.Namespace,
    planner_base: Optional[str],
    planner_key: str,
    answer_base: Optional[str],
    answer_key: str,
    judge_base: Optional[str],
    judge_key: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": item["id"],
        "question": item["question"],
        "subcategory": item.get("subcategory"),
        "expected_dt_query": item["expected_dt_query"],
        "expected_result": item["expected_result"],
        "reference_answer": item.get("reference_answer"),
    }
    try:
        plan = dt_llm_planner.plan_dt_query(
            question=item["question"],
            model=args.planner_model,
            api_base=planner_base,
            api_key=planner_key,
            timeout=args.request_timeout,
        )
        row["raw_planner_output"] = plan["raw_planner_output"]
        row["planned_dt_query"] = plan["planned_dt_query"]
    except Exception as exc:  # noqa: BLE001
        row.update({
            "planned_dt_query": {},
            "query_passed": False,
            "query_errors": [f"planner failed: {type(exc).__name__}: {exc}"],
            "execution_passed": False,
            "execution_errors": ["planner failed"],
            "answer_passed": False,
            "answer_errors": ["planner failed"],
            "joint_passed": False,
            "actual_answer": "",
        })
        return row

    query_eval = compare_query(item["expected_dt_query"], row["planned_dt_query"])
    with connect(args.db) as conn:
        dt_result, execution_errors = dt_llm_planner.execute_planned_query(conn, row["planned_dt_query"])
    row["query_passed"] = query_eval["passed"]
    row["query_errors"] = query_eval["errors"]
    row["execution_passed"] = not execution_errors
    row["execution_errors"] = execution_errors
    row["actual_dt_result"] = dt_result

    if dt_result is None:
        row.update({
            "answer_passed": False,
            "answer_errors": ["no DT result because query execution failed"],
            "joint_passed": False,
            "actual_answer": "",
        })
        return row

    try:
        answer = generate_answer(item["question"], row["planned_dt_query"], dt_result, args.llm_model, answer_base, answer_key, args.request_timeout)
    except Exception as exc:  # noqa: BLE001
        row.update({
            "answer_passed": False,
            "answer_errors": [f"answer generation failed: {type(exc).__name__}: {exc}"],
            "joint_passed": False,
            "actual_answer": "",
        })
        return row

    answer_eval = compare_answer_text(item, answer)
    row["actual_answer"] = answer
    row["rule_answer_passed"] = answer_eval["passed"]
    row["rule_answer_errors"] = answer_eval["errors"]
    row["answer_eval_method"] = "rule"

    if answer_eval["passed"]:
        row["answer_passed"] = True
        row["answer_errors"] = []
    elif args.judge_answer:
        try:
            judge_eval = judge_answer_equivalence(
                item=item,
                answer=answer,
                rule_errors=answer_eval["errors"],
                model=args.judge_model,
                api_base=judge_base,
                api_key=judge_key,
                timeout=args.request_timeout,
            )
            row["judge_answer_eval"] = judge_eval
            row["answer_eval_method"] = "judge_fallback"
            row["answer_passed"] = bool(judge_eval.get("passed"))
            row["answer_errors"] = [] if row["answer_passed"] else answer_eval["errors"]
        except Exception as exc:  # noqa: BLE001
            row["judge_answer_eval"] = {"error": f"{type(exc).__name__}: {exc}"}
            row["answer_passed"] = False
            row["answer_errors"] = answer_eval["errors"] + [f"judge failed: {type(exc).__name__}: {exc}"]
    else:
        row["answer_passed"] = False
        row["answer_errors"] = answer_eval["errors"]

    row["joint_passed"] = row["query_passed"] and row["execution_passed"] and row["answer_passed"]
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("subcategory") or "unknown", []).append(row)

    def avg(key: str, subset: list[dict[str, Any]]) -> float:
        return sum(1 for row in subset if row.get(key)) / len(subset) if subset else 0.0

    summary = {
        "total": total,
        "query_accuracy": avg("query_passed", rows),
        "execution_accuracy": avg("execution_passed", rows),
        "answer_accuracy": avg("answer_passed", rows),
        "joint_accuracy": avg("joint_passed", rows),
        "by_subcategory": {},
    }
    for name, subset in groups.items():
        summary["by_subcategory"][name] = {
            "total": len(subset),
            "query_accuracy": avg("query_passed", subset),
            "execution_accuracy": avg("execution_passed", subset),
            "answer_accuracy": avg("answer_passed", subset),
            "joint_accuracy": avg("joint_passed", subset),
        }
    return summary


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "total", "query_accuracy", "execution_accuracy", "answer_accuracy", "joint_accuracy"])
        writer.writerow(["overall", summary["total"], summary["query_accuracy"], summary["execution_accuracy"], summary["answer_accuracy"], summary["joint_accuracy"]])
        for sub, values in summary["by_subcategory"].items():
            writer.writerow([sub, values["total"], values["query_accuracy"], values["execution_accuracy"], values["answer_accuracy"], values["joint_accuracy"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DT benchmark with an LLM planner plus SQLite DT execution and LLM answer generation.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner-model", default="llama3.3:70b")
    parser.add_argument("--planner-api-base", default=None)
    parser.add_argument("--planner-api-key", default=None)
    parser.add_argument("--llm-model", default="llama3.3:70b")
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--judge-answer", action="store_true", help="Use LLM judge as fallback when rule-based answer_accuracy fails.")
    parser.add_argument("--judge-model", default=None, help="Judge model for answer semantic equivalence. Defaults to --llm-model.")
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()

    planner_base = normalize_base_url(args.planner_api_base, args.planner_model)
    planner_key = resolve_api_key(args.planner_api_key, args.planner_model)
    answer_base = normalize_base_url(args.llm_api_base, args.llm_model)
    answer_key = resolve_api_key(args.llm_api_key, args.llm_model)
    args.judge_model = args.judge_model or args.llm_model
    judge_base = normalize_base_url(args.judge_api_base, args.judge_model)
    judge_key = resolve_api_key(args.judge_api_key, args.judge_model)

    data = load_json(args.benchmark)
    items = data.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]

    rows = []
    for idx, item in enumerate(items, start=1):
        print(f"[DT LLM BENCH] {idx}/{len(items)} {item['id']}: {item['question']}", flush=True)
        rows.append(evaluate_item(item, args, planner_base, planner_key, answer_base, answer_key, judge_base, judge_key))

    summary = summarize(rows)
    payload = {
        "summary": summary,
        "rows": rows,
        "benchmark": str(args.benchmark),
        "db": str(args.db),
        "planner_model": args.planner_model,
        "planner_api_base": planner_base,
        "llm_model": args.llm_model,
        "llm_api_base": answer_base,
        "judge_answer": args.judge_answer,
        "judge_model": args.judge_model if args.judge_answer else None,
        "judge_api_base": judge_base if args.judge_answer else None,
        "evaluation_design": {
            "planner": "LLM converts question to structured DT query parameters.",
            "executor": "Independent SQLite execution via benchmark SQL logic.",
            "answerer": "LLM generates final answer from DT query/result only.",
            "metrics": ["query_accuracy", "execution_accuracy", "answer_accuracy", "joint_accuracy"],
            "answer_accuracy": "Rule-based key fact checking; if --judge-answer is enabled, failed rule checks are passed to an LLM judge for semantic equivalence fallback.",
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary, args.summary)
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] Results -> {args.results}", flush=True)
    print(f"[OK] Summary -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
