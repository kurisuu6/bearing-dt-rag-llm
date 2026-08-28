#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from llm_router import DEFAULT_ROUTER_MODEL, route_question

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_RESULTS = AGENT_DIR / "eval" / "router_eval_results_full.json"
DEFAULT_SUMMARY = AGENT_DIR / "eval" / "router_eval_summary_full.csv"


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if not isinstance(data, list):
        raise ValueError("Router evaluation questions must be a JSON list.")
    return data


def evaluate_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = item.get("question")
    expected = item.get("expected_route")
    errors = []
    routed = None
    try:
        routed = route_question(
            question,
            mode=args.router_mode,
            model=args.router_model,
            api_base=args.openai_api_base,
            api_key=args.openai_api_key,
            confidence_threshold=args.router_confidence_threshold,
            timeout=args.router_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - kept in eval output
        errors.append(str(exc))
        routed = {
            "route": None,
            "confidence": 0.0,
            "reason": "router failed",
            "dt_entities": {},
            "rag_query": "",
            "matched_signals": {},
        }
    actual = routed.get("route")
    return {
        "id": item.get("id"),
        "question": question,
        "expected_route": expected,
        "actual_route": actual,
        "route_ok": expected == actual,
        "confidence": routed.get("confidence"),
        "router": routed.get("router"),
        "reason": routed.get("reason"),
        "dt_entities": routed.get("dt_entities"),
        "rag_query": routed.get("rag_query"),
        "errors": errors,
        "raw_route": routed,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_expected: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row.get("expected_route") or "unknown"
        bucket = by_expected.setdefault(expected, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(row.get("route_ok")))
    return {
        "total": total,
        "accuracy": sum(1 for row in rows if row.get("route_ok")) / total if total else 0.0,
        "error_count": sum(len(row.get("errors", [])) for row in rows),
        "by_expected_route": {
            route: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for route, stats in sorted(by_expected.items())
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "expected_route", "actual_route", "route_ok", "confidence", "question", "reason", "errors"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        writer.writerow({})
        for key, value in summary.items():
            writer.writerow({"id": key, "expected_route": json.dumps(value, ensure_ascii=False)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule, LLM, or hybrid router accuracy.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--router-timeout", type=float, default=30.0)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    rows = []
    for idx, item in enumerate(questions, start=1):
        print(f"[ROUTE] {idx}/{len(questions)} {item.get('id')}: {item.get('question')}", flush=True)
        rows.append(evaluate_one(item, args))

    summary = summarize(rows)
    payload = {"summary": summary, "rows": rows}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.summary, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Results -> {args.results}")
    print(f"[OK] Summary -> {args.summary}")


if __name__ == "__main__":
    main()
