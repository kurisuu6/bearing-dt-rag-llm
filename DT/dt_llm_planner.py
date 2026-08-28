#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from build_dt_benchmark import (  # type: ignore
    CORE_FEATURES,
    DEGRADATION_FEATURES,
    TREND_WINDOW,
    Z_THRESHOLD,
    build_expected,
    connect,
    round_float,
)

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
    "diagnose_features": "features",
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


def resolve_ollama_openai_base() -> Optional[str]:
    base = os.environ.get("OLLAMA_OPENAI_BASE") or os.environ.get("OLLAMA_BASE_URL")
    if not base:
        return None
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def normalize_base_url(base: Optional[str], model: str) -> Optional[str]:
    if base:
        cleaned = base.rstrip("/")
    elif "llama" in model.lower():
        cleaned = resolve_ollama_openai_base()
    else:
        raw = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        cleaned = raw.rstrip("/") if raw else None
    if cleaned and "llama" in model.lower() and not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def resolve_api_key(explicit: Optional[str], model: str) -> str:
    if explicit:
        return explicit
    if "llama" in model.lower():
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


def complete_schema_defaults(query: dict[str, Any]) -> dict[str, Any]:
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
- diagnostic or DT+RAG questions about a specific bearing should usually use action=features so the downstream RAG step receives the bearing state, evidence string, and abnormal vibration features.

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

Question: Diagnose bearing_3 in 1st_test and explain the mechanical meaning of its abnormal features.
JSON: {{"action":"features","experiment":"1st_test","bearing_id":"bearing_3","features":["health_index","rms","kurtosis","crest_factor","impulse_factor","clearance_factor","high_band_energy_ratio","spectral_entropy"],"time_policy":"latest_available_record","reason":"diagnostic DT+RAG question needs latest state, feature values, and evidence"}}

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


def plan_dt_query(
    question: str,
    model: str = "llama3.3:70b",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    resolved_base = normalize_base_url(api_base, model)
    resolved_key = resolve_api_key(api_key, model)
    raw = call_llm_json(question, model, resolved_base, resolved_key, timeout)
    planned = normalize_planned_query(raw, question)
    return {
        "raw_planner_output": raw,
        "planned_dt_query": planned,
        "planner_model": model,
        "planner_api_base": resolved_base,
    }


def execute_planned_query(conn_or_db_path: Any, planned_query: dict[str, Any]) -> tuple[Optional[dict[str, Any]], list[str]]:
    action = planned_query.get("action")
    if action not in VALID_ACTIONS:
        return None, [f"invalid action: {action}"]
    spec = {k: v for k, v in planned_query.items() if k not in {"time_policy", "reason"}}

    def _execute(conn: Any) -> dict[str, Any]:
        return round_float(build_expected(conn, spec))

    try:
        if isinstance(conn_or_db_path, (str, Path)):
            with connect(Path(conn_or_db_path)) as conn:
                return _execute(conn), []
        return _execute(conn_or_db_path), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"execution failed: {type(exc).__name__}: {exc}"]


def run_dt_with_llm_planner(
    question: str,
    db_path: Path,
    model: str = "llama3.3:70b",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    plan = plan_dt_query(question, model=model, api_base=api_base, api_key=api_key, timeout=timeout)
    result, errors = execute_planned_query(db_path, plan["planned_dt_query"])
    payload = {
        "action": plan["planned_dt_query"].get("action"),
        "planner": {
            "mode": "llm",
            "model": plan["planner_model"],
            "api_base": plan["planner_api_base"],
            "raw_planner_output": plan["raw_planner_output"],
            "planned_dt_query": plan["planned_dt_query"],
        },
        "result": result,
    }
    if errors:
        payload["errors"] = errors
    return payload
