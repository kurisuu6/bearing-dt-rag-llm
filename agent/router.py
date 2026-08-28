#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

ROUTES = {"dt_only", "rag_only", "dt_rag", "unrelated"}

DT_EXPERIMENT_RE = re.compile(r"\b(?:1st|2nd|3rd)_test\b", re.IGNORECASE)
DT_BEARING_RE = re.compile(r"\b(?:bearing[_\s-]?[1-4]|bearing\s+(?:one|two|three|four))\b", re.IGNORECASE)
DT_TERMS = {
    "ims", "digital twin", "dt", "health index", "health_index", "health state", "state",
    "rms", "kurtosis", "crest factor", "impulse factor", "clearance factor",
    "spectral entropy", "dominant frequency", "high-band", "high band", "energy ratio",
    "feature", "features", "trend", "degradation", "anomaly", "anomalies",
    "snapshot", "sequence", "latest", "current", "failure_near", "early_degradation",
}
RAG_TERMS = {
    "skf", "manual", "handbook", "bearing basics", "radial bearing", "radial bearings",
    "ball bearing", "roller bearing", "rolling bearing", "rolling bearings", "bearing arrangement", "mounting", "mounted", "dismounting",
    "lubrication", "lubricant", "grease", "oil", "seal", "seals", "alignment",
    "inspection", "maintenance", "troubleshooting", "misalignment", "bearing damage", "damage", "cause", "causes",
    "clearance", "preload", "load", "life", "selection", "designation", "housing", "bearing unit",
}
EXPLANATION_TERMS = {
    "why", "explain", "reason", "mechanism", "indicate", "indicates", "mean", "means",
    "interpret", "diagnose", "diagnosis", "because", "according to", "what does", "how should",
}
UNRELATED_TERMS = {
    "weather", "stock", "movie", "recipe", "football", "basketball", "joke", "poem",
    "translate", "capital of", "travel", "restaurant", "music", "email",
}

NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
}


@dataclass
class RouteResult:
    route: str
    confidence: float
    reason: str
    dt_entities: dict[str, Any]
    rag_query: str
    matched_signals: dict[str, list[str]]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_terms(text: str, terms: set[str]) -> list[str]:
    found = []
    normalized = normalize_text(text)
    for term in sorted(terms, key=len, reverse=True):
        pattern = r"(?<![a-z0-9_])" + re.escape(term.lower()) + r"(?![a-z0-9_])"
        if re.search(pattern, normalized):
            found.append(term)
    return found


def extract_experiment(question: str) -> Optional[str]:
    match = DT_EXPERIMENT_RE.search(question)
    return match.group(0).lower() if match else None


def extract_bearing_id(question: str) -> Optional[str]:
    match = DT_BEARING_RE.search(question)
    if not match:
        return None
    text = match.group(0).lower().replace("-", "_").replace(" ", "_")
    for word, digit in NUMBER_WORDS.items():
        text = text.replace(word, digit)
    text = text.replace("bearing_", "bearing_")
    digit_match = re.search(r"([1-4])", text)
    return f"bearing_{digit_match.group(1)}" if digit_match else None


def extract_metric(question: str) -> Optional[str]:
    metrics = [
        "health_index", "health index", "rms", "kurtosis", "crest factor", "impulse factor",
        "clearance factor", "spectral entropy", "dominant frequency", "high-band energy ratio",
        "high band energy ratio", "health state", "state",
    ]
    normalized = normalize_text(question).replace("-", " ")
    for metric in metrics:
        if metric.replace("_", " ") in normalized:
            return metric.replace(" ", "_")
    return None


def has_dt_entity(question: str) -> bool:
    return bool(DT_EXPERIMENT_RE.search(question) or DT_BEARING_RE.search(question))


def rule_based_route(question: str) -> RouteResult:
    normalized = normalize_text(question)
    dt_terms = find_terms(question, DT_TERMS)
    rag_terms = find_terms(question, RAG_TERMS)
    explanation_terms = find_terms(question, EXPLANATION_TERMS)
    unrelated_terms = find_terms(question, UNRELATED_TERMS)

    experiment = extract_experiment(question)
    bearing_id = extract_bearing_id(question)
    metric = extract_metric(question)
    dt_entity = has_dt_entity(question)
    dt_score = len(dt_terms) + (2 if experiment else 0) + (2 if bearing_id else 0)
    rag_score = len(rag_terms)
    explanation_score = len(explanation_terms)

    dt_entities = {
        "experiment": experiment,
        "bearing_id": bearing_id,
        "metric": metric,
    }

    if unrelated_terms and not dt_entity and rag_score == 0 and metric is None:
        return RouteResult(
            route="unrelated",
            confidence=0.90,
            reason="The question contains unrelated-domain terms and no IMS DT or SKF bearing-manual signals.",
            dt_entities=dt_entities,
            rag_query="",
            matched_signals={"dt": dt_terms, "rag": rag_terms, "explanation": explanation_terms, "unrelated": unrelated_terms},
        )

    if dt_score > 0 and (rag_score > 0 or explanation_score > 0 and any(t in normalized for t in ["why", "explain", "indicate", "mean", "diagnose"])):
        return RouteResult(
            route="dt_rag",
            confidence=0.86 if rag_score else 0.78,
            reason="The question needs IMS DT values/status and also asks for interpretation or bearing-domain explanation.",
            dt_entities=dt_entities,
            rag_query=question,
            matched_signals={"dt": dt_terms, "rag": rag_terms, "explanation": explanation_terms, "unrelated": unrelated_terms},
        )

    if dt_score > 0:
        confidence = 0.90 if dt_entity else 0.76
        return RouteResult(
            route="dt_only",
            confidence=confidence,
            reason="The question asks about IMS experiment, bearing state, features, anomaly, or trend data.",
            dt_entities=dt_entities,
            rag_query="",
            matched_signals={"dt": dt_terms, "rag": rag_terms, "explanation": explanation_terms, "unrelated": unrelated_terms},
        )

    if rag_score > 0:
        return RouteResult(
            route="rag_only",
            confidence=0.84,
            reason="The question asks about SKF manual knowledge or general bearing maintenance/domain concepts without IMS DT entities.",
            dt_entities=dt_entities,
            rag_query=question,
            matched_signals={"dt": dt_terms, "rag": rag_terms, "explanation": explanation_terms, "unrelated": unrelated_terms},
        )

    return RouteResult(
        route="unrelated",
        confidence=0.62,
        reason="No clear IMS DT or SKF RAG signal was found by the rule-based router.",
        dt_entities=dt_entities,
        rag_query="",
        matched_signals={"dt": dt_terms, "rag": rag_terms, "explanation": explanation_terms, "unrelated": unrelated_terms},
    )


def route_question(question: str) -> dict[str, Any]:
    return asdict(rule_based_route(question))


def evaluate_questions(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    correct = 0
    total = 0
    for item in data:
        question = item["question"]
        expected = item.get("expected_route")
        routed = route_question(question)
        ok = expected == routed["route"] if expected else None
        if ok is not None:
            total += 1
            correct += int(ok)
        results.append({"question": question, "expected_route": expected, "ok": ok, **routed})
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based router for bearing DT + SKF RAG agent questions.")
    parser.add_argument("--question", default="", help="Question to route.")
    parser.add_argument("--eval", type=Path, default=None, help="Path to evaluation question JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for eval results.")
    args = parser.parse_args()

    if args.eval:
        result = evaluate_questions(args.eval)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return

    if not args.question:
        raise SystemExit("Provide --question or --eval")
    print(json.dumps(route_question(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
