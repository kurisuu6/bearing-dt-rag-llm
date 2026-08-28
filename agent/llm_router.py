#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from router import (
    EXPLANATION_TERMS,
    RAG_TERMS,
    UNRELATED_TERMS,
    extract_bearing_id,
    extract_experiment,
    extract_metric,
    find_terms,
    route_question as rule_route_question,
)

ROUTES = {"dt_only", "rag_only", "dt_rag", "unrelated"}
DEFAULT_ROUTER_MODEL = "gpt-4o-mini"


def model_uses_ollama(model: str) -> bool:
    return "llama" in model.lower()


def resolve_ollama_openai_base() -> Optional[str]:
    base = os.environ.get("OLLAMA_BASE_URL")
    if not base:
        return None
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def resolve_llm_api(model: str, api_base: Optional[str], api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if model_uses_ollama(model) and not api_key:
        # Router calls often receive the OpenAI embedding base via api_base. For local
        # llama models, prefer Ollama unless the caller also provides an explicit key.
        base = resolve_ollama_openai_base() or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    elif api_base:
        base = api_base
    elif model_uses_ollama(model):
        base = resolve_ollama_openai_base() or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    else:
        base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    if api_key:
        key = api_key
    elif model_uses_ollama(model):
        key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    else:
        key = os.environ.get("OPENAI_API_KEY")
    return base, key


def parse_router_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_llm_route(raw: dict[str, Any], question: str, model: str, api_base: Optional[str]) -> dict[str, Any]:
    route = str(raw.get("route", "")).strip().lower()
    if route not in ROUTES:
        route = "unrelated"
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    dt_entities = raw.get("dt_entities") if isinstance(raw.get("dt_entities"), dict) else {}
    dt_entities = {
        "experiment": dt_entities.get("experiment") or extract_experiment(question),
        "bearing_id": dt_entities.get("bearing_id") or extract_bearing_id(question),
        "metric": dt_entities.get("metric") or extract_metric(question),
    }
    rag_query = raw.get("rag_query") or (question if route in {"rag_only", "dt_rag"} else "")
    return {
        "route": route,
        "confidence": confidence,
        "reason": str(raw.get("reason") or "LLM router decision."),
        "dt_entities": dt_entities,
        "rag_query": rag_query,
        "matched_signals": raw.get("matched_signals") if isinstance(raw.get("matched_signals"), dict) else {},
        "router": {
            "mode": "llm",
            "model": model,
            "api_base": api_base,
        },
    }


def build_router_messages(question: str, rule_hint: Optional[dict[str, Any]] = None) -> list[dict[str, str]]:
    system = (
        "You are the router for a bearing diagnosis QA system. Classify the user question into exactly one route.\n"
        "Routes:\n"
        "- unrelated: outside bearing diagnosis, IMS data, SKF manual, or maintenance scope.\n"
        "- dt_only: asks for IMS digital-twin data such as experiments, bearing_1..bearing_4, health_index, health_state, RMS, kurtosis, crest factor, features, anomalies, or trends.\n"
        "For IMS DT questions, current/latest/now means the latest available record in the offline IMS DT database, not the real-world current date.\n"
        "- rag_only: asks for SKF manual or general bearing knowledge without needing IMS DT measurements.\n"
        "- dt_rag: needs both IMS DT evidence and bearing/SKF explanation, diagnosis, maintenance guidance, or failure-mechanism interpretation.\n"
        "Return strict JSON only. No markdown.\n"
        "JSON schema: {\"route\":\"...\",\"confidence\":0.0,\"reason\":\"...\",\"dt_entities\":{\"experiment\":null,\"bearing_id\":null,\"metric\":null},\"rag_query\":\"...\"}.\n"
        "Valid experiments are 1st_test, 2nd_test, 3rd_test. Valid bearings are bearing_1, bearing_2, bearing_3, bearing_4."
    )
    payload = {"question": question}
    if rule_hint:
        payload["rule_router_hint"] = rule_hint
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def llm_route_question(
    question: str,
    model: str = DEFAULT_ROUTER_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    rule_hint: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("The openai package is required for LLM routing. Install it with: pip install openai") from exc

    base, key = resolve_llm_api(model, api_base, api_key)
    if not key:
        raise RuntimeError("OPENAI_API_KEY or OLLAMA_API_KEY is required for LLM routing.")

    kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
    if base:
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_router_messages(question, rule_hint),
        "temperature": 0.0,
    }
    if not model_uses_ollama(model):
        request_kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or "{}"
    try:
        raw = parse_router_json(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM router returned invalid JSON: {content}") from exc
    return normalize_llm_route(raw, question, model, base)


def has_ims_entity(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> bool:
    rule_entities = rule_result.get("dt_entities") or {}
    llm_entities = llm_result.get("dt_entities") or {}
    return bool(
        extract_experiment(question)
        or extract_bearing_id(question)
        or rule_entities.get("experiment")
        or rule_entities.get("bearing_id")
        or llm_entities.get("experiment")
        or llm_entities.get("bearing_id")
    )


def has_dt_measurement_intent(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> bool:
    metric = extract_metric(question)
    if metric:
        return True
    text = question.lower()
    dt_words = [
        "health state",
        "health index",
        "feature",
        "features",
        "rms",
        "kurtosis",
        "crest factor",
        "impulse factor",
        "clearance factor",
        "spectral entropy",
        "dominant frequency",
        "high-band",
        "high band",
        "trend",
        "over time",
        "latest",
        "current",
        "classified",
        "anomal",
    ]
    if any(word in text for word in dt_words):
        return True
    return bool((rule_result.get("matched_signals") or {}).get("dt") or (llm_result.get("matched_signals") or {}).get("dt"))


def has_explanation_or_maintenance_intent(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> bool:
    text = question.lower()
    extra_terms = [
        "fault",
        "failure",
        "damage",
        "maintenance",
        "inspect",
        "inspection",
        "corrective",
        "action",
        "continue operating",
        "stop the machine",
        "replace",
        "mechanical meaning",
        "what kind of bearing problem",
    ]
    if find_terms(question, EXPLANATION_TERMS) or any(term in text for term in extra_terms):
        return True
    return bool(
        (rule_result.get("matched_signals") or {}).get("explanation")
        or (llm_result.get("matched_signals") or {}).get("explanation")
    )


def has_manual_intent(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> bool:
    text = question.lower()
    if find_terms(question, RAG_TERMS):
        return True
    manual_words = [
        "skf",
        "manual",
        "bearing",
        "mounting",
        "lubrication",
        "alignment",
        "inspection",
        "troubleshooting",
        "dismounting",
        "seal",
        "housing",
    ]
    if any(word in text for word in manual_words):
        return True
    return bool((rule_result.get("matched_signals") or {}).get("rag") or (llm_result.get("matched_signals") or {}).get("rag"))


def is_clearly_unrelated(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> bool:
    if find_terms(question, UNRELATED_TERMS) and not has_ims_entity(question, rule_result, llm_result) and not has_manual_intent(
        question, rule_result, llm_result
    ):
        return True
    return rule_result.get("route") == "unrelated" and llm_result.get("route") == "unrelated"


def merge_dt_entities(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> dict[str, Any]:
    rule_entities = rule_result.get("dt_entities") or {}
    llm_entities = llm_result.get("dt_entities") or {}
    return {
        "experiment": llm_entities.get("experiment") or rule_entities.get("experiment") or extract_experiment(question),
        "bearing_id": llm_entities.get("bearing_id") or rule_entities.get("bearing_id") or extract_bearing_id(question),
        "metric": llm_entities.get("metric") or rule_entities.get("metric") or extract_metric(question),
    }


def select_result_template(route: str, question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> dict[str, Any]:
    if route == llm_result.get("route"):
        selected = dict(llm_result)
    elif route == rule_result.get("route"):
        selected = dict(rule_result)
    else:
        selected = dict(rule_result)
    selected["route"] = route
    selected["dt_entities"] = merge_dt_entities(question, rule_result, llm_result)
    selected["rag_query"] = question if route in {"rag_only", "dt_rag"} else ""
    return selected


def constraint_arbiter(question: str, rule_result: dict[str, Any], llm_result: dict[str, Any]) -> tuple[str, str]:
    ims_entity = has_ims_entity(question, rule_result, llm_result)
    dt_intent = has_dt_measurement_intent(question, rule_result, llm_result)
    explanation_intent = has_explanation_or_maintenance_intent(question, rule_result, llm_result)
    manual_intent = has_manual_intent(question, rule_result, llm_result)

    if is_clearly_unrelated(question, rule_result, llm_result):
        return "unrelated", "clearly_unrelated_no_domain_signal"

    if ims_entity and explanation_intent:
        return "dt_rag", "ims_entity_with_explanation_or_maintenance_intent"

    if ims_entity and dt_intent and not explanation_intent:
        return "dt_only", "ims_entity_with_measurement_state_or_trend_intent"

    if not ims_entity and manual_intent:
        return "rag_only", "manual_or_bearing_knowledge_without_ims_entity"

    if not ims_entity and llm_result.get("route") in {"dt_only", "dt_rag"}:
        return rule_result.get("route", "unrelated"), "llm_dt_route_blocked_without_ims_entity"

    return rule_result.get("route", "unrelated"), "fallback_to_rule_route"


def hybrid_route_question(
    question: str,
    model: str = DEFAULT_ROUTER_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    confidence_threshold: float = 0.85,
    timeout: float = 30.0,
) -> dict[str, Any]:
    rule_result = rule_route_question(question)
    rule_result.setdefault("router", {"mode": "rule"})
    llm_result = llm_route_question(
        question=question,
        model=model,
        api_base=api_base,
        api_key=api_key,
        rule_hint=rule_result,
        timeout=timeout,
    )

    if rule_result.get("route") == llm_result.get("route"):
        final_route = rule_result.get("route", "unrelated")
        decision_type = "agreement"
        selected_source = "agreement"
        selected = select_result_template(final_route, question, rule_result, llm_result)
        reason = f"Rule and LLM agree on {final_route}."
    else:
        final_route, decision_type = constraint_arbiter(question, rule_result, llm_result)
        selected_source = "constraint_arbiter"
        selected = select_result_template(final_route, question, rule_result, llm_result)
        reason = (
            f"Rule route {rule_result.get('route')} and LLM route {llm_result.get('route')} differ; "
            f"constraint arbiter selected {final_route} by {decision_type}."
        )

    selected["reason"] = reason
    selected["confidence"] = max(float(rule_result.get("confidence", 0.0)), float(llm_result.get("confidence", 0.0)))
    selected["matched_signals"] = rule_result.get("matched_signals", {})
    selected["router"] = {
        "mode": "hybrid",
        "strategy": "constraint_guided_rule_llm",
        "selected": selected_source,
        "decision_type": decision_type,
        "legacy_confidence_threshold": confidence_threshold,
        "model": model,
        "api_base": llm_result.get("router", {}).get("api_base"),
        "rule_result": rule_result,
        "llm_result": llm_result,
    }
    return selected


def route_question(
    question: str,
    mode: str = "hybrid",
    model: str = DEFAULT_ROUTER_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    confidence_threshold: float = 0.85,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if mode == "rule":
        result = rule_route_question(question)
        result["router"] = {"mode": "rule"}
        return result
    if mode == "llm":
        return llm_route_question(question, model=model, api_base=api_base, api_key=api_key, timeout=timeout)
    if mode == "hybrid":
        return hybrid_route_question(
            question,
            model=model,
            api_base=api_base,
            api_key=api_key,
            confidence_threshold=confidence_threshold,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported router mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule/LLM/hybrid router for bearing DT + SKF RAG agent.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--router-timeout", type=float, default=30.0)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    args = parser.parse_args()
    result = route_question(
        args.question,
        mode=args.router_mode,
        model=args.router_model,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key,
        confidence_threshold=args.router_confidence_threshold,
        timeout=args.router_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
