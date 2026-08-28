#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[0]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from bearing_agent import DEFAULT_DB, DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR, DEFAULT_MANIFEST, DEFAULT_RERANKER_URL, RETRIEVAL_MODES, answer_question

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_RESULTS = AGENT_DIR / "eval" / "agent_eval_results.json"
DEFAULT_SUMMARY = AGENT_DIR / "eval" / "agent_eval_summary.csv"


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Agent evaluation questions must be a JSON list.")
    return data


def has_tool(result: dict[str, Any], tool: str, skip_rag: bool) -> bool | None:
    if tool == "dt":
        return result.get("dt_result") is not None
    if tool == "rag":
        if skip_rag:
            return None
        return result.get("rag_result") is not None
    raise ValueError(f"Unknown expected tool: {tool}")


def normalize_keyword_text(text: str) -> str:
    normalized = str(text).lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def keyword_is_hit(answer_normalized: str, keyword: str) -> bool:
    keyword_normalized = normalize_keyword_text(keyword)
    if not keyword_normalized:
        return True
    if keyword_normalized in answer_normalized:
        return True

    keyword_tokens = keyword_normalized.split()
    if len(keyword_tokens) >= 2 and all(token in answer_normalized for token in keyword_tokens):
        return True

    # Accept common natural-language renderings of internal DT labels.
    aliases = {
        "strongly increasing degradation": ["strongly increasing trend", "strongly increasing", "severe increasing degradation"],
        "high band energy ratio": ["high frequency energy ratio", "high-band energy ratio"],
        "health index": ["health_index"],
        "crest factor": ["crest_factor"],
        "spectral entropy": ["spectral_entropy"],
    }
    return any(alias in answer_normalized for alias in aliases.get(keyword_normalized, []))


def keyword_hits(answer: str, keywords: list[str]) -> tuple[int, list[str]]:
    answer_normalized = normalize_keyword_text(answer)
    missing = []
    hits = 0
    for keyword in keywords:
        if keyword_is_hit(answer_normalized, keyword):
            hits += 1
        else:
            missing.append(keyword)
    return hits, missing


def evaluate_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = answer_question(
        question=item["question"],
        db_path=args.db,
        index_dir=args.index_dir,
        manifest_path=args.manifest,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        skip_rag=args.skip_rag,
        generate_answer=args.answer,
        llm_model=args.llm_model,
        router_mode=args.router_mode,
        router_model=args.router_model,
        router_confidence_threshold=args.router_confidence_threshold,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_mode=args.retrieval_mode,
        reranker_url=getattr(args, "effective_reranker_url", None) or args.reranker_url,
        reranker_top_k=args.reranker_top_k,
        reranker_timeout=args.reranker_timeout,
        vector_keep_top_k=args.vector_keep_top_k,
        dt_planner_mode=args.dt_planner_mode,
        dt_planner_model=args.dt_planner_model,
        dt_planner_api_base=args.dt_planner_api_base,
        dt_planner_api_key=args.dt_planner_api_key,
        dt_planner_timeout=args.dt_planner_timeout,
    )
    expected_route = item.get("expected_route")
    actual_route = result.get("route", {}).get("route")
    route_ok = expected_route == actual_route

    expected_tools = item.get("expected_tools", [])
    tool_status = {}
    considered_tools = 0
    passed_tools = 0
    for tool in expected_tools:
        ok = has_tool(result, tool, args.skip_rag)
        tool_status[tool] = "skipped" if ok is None else bool(ok)
        if ok is not None:
            considered_tools += 1
            passed_tools += int(bool(ok))
    unexpected_dt = "dt" not in expected_tools and result.get("dt_result") is not None
    unexpected_rag = "rag" not in expected_tools and result.get("rag_result") is not None
    unexpected_tool_used = unexpected_dt or unexpected_rag
    tools_ok = (passed_tools == considered_tools) and not unexpected_tool_used

    answer = result.get("answer") or ""
    expected_keywords = item.get("expected_keywords", [])
    hits, missing = keyword_hits(answer, expected_keywords)
    keyword_rate = hits / len(expected_keywords) if expected_keywords else 1.0
    keywords_ok = keyword_rate >= args.keyword_threshold

    errors = result.get("errors", [])
    row = {
        "id": item.get("id"),
        "question": item.get("question"),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_ok": route_ok,
        "expected_tools": expected_tools,
        "tool_status": tool_status,
        "tools_ok": tools_ok,
        "unexpected_tool_used": unexpected_tool_used,
        "expected_keywords": expected_keywords,
        "keyword_hits": hits,
        "keyword_total": len(expected_keywords),
        "keyword_rate": keyword_rate,
        "keywords_ok": keywords_ok,
        "error_count": len(errors),
        "errors": errors,
        "answer": answer,
        "raw_result": result,
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    def ratio(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / total if total else 0.0
    return {
        "total": total,
        "route_accuracy": ratio("route_ok"),
        "tool_call_accuracy": ratio("tools_ok"),
        "keyword_pass_rate": ratio("keywords_ok"),
        "average_keyword_hit_rate": sum(row.get("keyword_rate", 0.0) for row in rows) / total if total else 0.0,
        "error_count": sum(row.get("error_count", 0) for row in rows),
        "rows_with_errors": sum(1 for row in rows if row.get("error_count", 0) > 0),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "expected_route", "actual_route", "route_ok", "tools_ok",
        "keyword_hits", "keyword_total", "keyword_rate", "keywords_ok", "error_count", "question", "answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        writer.writerow({})
        for key, value in summary.items():
            writer.writerow({"id": key, "expected_route": value})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the integrated bearing DT + SKF RAG agent.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--router-model", default="gpt-4o-mini")
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default="dense")
    parser.add_argument("--dt-planner-mode", choices=["legacy", "llm"], default="legacy")
    parser.add_argument("--dt-planner-model", default="llama3.3:70b")
    parser.add_argument("--dt-planner-api-base", default=None)
    parser.add_argument("--dt-planner-api-key", default=None)
    parser.add_argument("--dt-planner-timeout", type=float, default=120.0)
    parser.add_argument("--reranker-url", default=None)
    parser.add_argument("--reranker-top-k", type=int, default=None)
    parser.add_argument("--reranker-timeout", type=float, default=30.0)
    parser.add_argument("--vector-keep-top-k", type=int, default=0)
    parser.add_argument("--answer", action="store_true", help="Generate LLM final answers during evaluation.")
    parser.add_argument("--skip-rag", action="store_true", help="Skip SKF RAG retrieval for offline router/DT checks.")
    parser.add_argument("--keyword-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.effective_reranker_url = args.reranker_url or DEFAULT_RERANKER_URL

    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    rows = []
    for idx, item in enumerate(questions, start=1):
        print(f"[RUN] {idx}/{len(questions)} {item.get('id')}: {item.get('question')}", flush=True)
        rows.append(evaluate_one(item, args))

    summary = summarize(rows)
    payload = {"summary": summary, "rows": rows}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(args.summary, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Results -> {args.results}")
    print(f"[OK] Summary -> {args.summary}")


if __name__ == "__main__":
    main()
