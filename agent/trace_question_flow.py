#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[0]
DT_DIR = PROJECT_ROOT / "DT"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))

from bearing_agent import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_RERANKER_URL,
    DEFAULT_ROUTER_MODEL,
    RETRIEVAL_MODES,
    build_dt_rag_retrieval_plan,
    build_evidence_plan,
    build_llm_prompt,
    generate_llm_answer,
    infer_dt_action,
    make_deterministic_answer,
    run_dt,
    run_rag,
)
from llm_router import route_question  # noqa: E402
from dt_llm_planner import run_dt_with_llm_planner  # noqa: E402

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_OUTPUT_JSON = AGENT_DIR / "eval" / "trace_dt_rag_explain_002_single_query_v2.json"
DEFAULT_OUTPUT_MD = AGENT_DIR / "eval" / "trace_dt_rag_explain_002_single_query_v2.md"


def load_eval_item(path: Path, item_id: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("id") == item_id:
            return item
    raise ValueError(f"Question id not found: {item_id}")


def compact_for_trace(obj: Any, max_text_chars: int = 900) -> Any:
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "text" and isinstance(value, str):
                result[key] = value[:max_text_chars] + (" ... [truncated]" if len(value) > max_text_chars else "")
            elif key in {"raw_result", "tables"}:
                continue
            else:
                result[key] = compact_for_trace(value, max_text_chars)
        return result
    if isinstance(obj, list):
        return [compact_for_trace(item, max_text_chars) for item in obj]
    return obj


def make_markdown(trace: dict[str, Any]) -> str:
    lines = []
    item = trace["input"]
    lines.append("# Question Flow Trace")
    lines.append("")
    lines.append(f"Question ID: `{item.get('id')}`")
    lines.append("")
    lines.append(f"Question: {item.get('question')}")
    lines.append("")
    lines.append("## 1. Expected Annotation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: item.get(k) for k in ["expected_route", "expected_tools", "expected_keywords"]}, ensure_ascii=False, indent=2))
    lines.append("```")

    route = trace.get("route", {})
    lines.append("")
    lines.append("## 2. Router Decision")
    lines.append("")
    lines.append(f"Route: `{route.get('route')}`")
    lines.append(f"Confidence: `{route.get('confidence')}`")
    lines.append(f"Reason: {route.get('reason')}")
    lines.append("")
    lines.append("DT entities:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(route.get("dt_entities"), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Router metadata:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(route.get("router"), ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append("")
    lines.append("## 3. Tool Plan")
    lines.append("")
    plan = trace.get("tool_plan", {})
    lines.append("```json")
    lines.append(json.dumps(plan, ensure_ascii=False, indent=2))
    lines.append("```")

    if trace.get("dt_rag_retrieval_plan"):
        lines.append("")
        lines.append("DT+RAG retrieval plan:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(compact_for_trace(trace["dt_rag_retrieval_plan"], max_text_chars=1200), ensure_ascii=False, indent=2))
        lines.append("```")

    lines.append("")
    lines.append("## 4. DT Tool Result")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(compact_for_trace(trace.get("dt_result")), ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append("")
    lines.append("## 5. RAG Retrieval Result")
    lines.append("")
    rag_result = trace.get("rag_result")
    if rag_result is None:
        lines.append("RAG was skipped or unavailable.")
    else:
        chunks = rag_result.get("chunks", [])
        lines.append(f"Retrieved chunks: `{len(chunks)}`")
        for chunk in chunks[:5]:
            lines.append("")
            lines.append(f"### Chunk {chunk.get('rank')} | score={chunk.get('score')} | id={chunk.get('chunk_id')}")
            lines.append(f"Section: `{chunk.get('section')}` | Pages: `{chunk.get('page_start')}-{chunk.get('page_end')}`")
            if chunk.get("selection_reason"):
                lines.append(
                    f"Selection: `{chunk.get('selection_reason')}` | "
                    f"Query index: `{chunk.get('selection_query_index')}` | "
                    f"Matched queries: `{chunk.get('matched_queries')}`"
                )
            text = (chunk.get("text") or "").replace("\n", " ")
            lines.append("")
            lines.append(text[:900] + (" ..." if len(text) > 900 else ""))

    if trace.get("evidence_plan"):
        lines.append("")
        lines.append("## 6. Evidence Plan")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(compact_for_trace(trace["evidence_plan"], max_text_chars=1200), ensure_ascii=False, indent=2))
        lines.append("```")

    lines.append("")
    lines.append("## 7. Deterministic Answer")
    lines.append("")
    lines.append(trace.get("deterministic_answer") or "")

    if trace.get("llm_prompt"):
        lines.append("")
        lines.append("## 8. LLM Prompt Preview")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(compact_for_trace(trace["llm_prompt"], max_text_chars=1200), ensure_ascii=False, indent=2))
        lines.append("```")

    if trace.get("llm_answer"):
        lines.append("")
        lines.append("## 9. LLM Final Answer")
        lines.append("")
        lines.append(trace["llm_answer"].get("answer") or "")

    if trace.get("errors"):
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(trace["errors"], ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines) + "\n"


def trace_question_flow(
    item: dict[str, Any],
    db_path: Path,
    index_dir: Path,
    manifest_path: Path,
    top_k: int,
    embedding_model: str,
    router_mode: str,
    router_model: str,
    router_confidence_threshold: float,
    llm_model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    llm_api_base: Optional[str],
    llm_api_key: Optional[str],
    skip_rag: bool,
    answer: bool,
    rag_per_query_top_k: int,
    rag_min_chunks_per_query: int,
    retrieval_top_k: Optional[int],
    reranker_url: Optional[str],
    reranker_top_k: Optional[int],
    reranker_timeout: float,
    vector_keep_top_k: int,
    retrieval_mode: str,
    dt_planner_mode: str = "legacy",
    dt_planner_model: str = "llama3.3:70b",
    dt_planner_api_base: Optional[str] = None,
    dt_planner_api_key: Optional[str] = None,
    dt_planner_timeout: float = 120.0,
    dt_rag_dense_keep_top_k: int = 0,
    dt_rag_bm25_keep_top_k: int = 0,
) -> dict[str, Any]:
    question = item["question"]
    errors = []
    effective_reranker_url = reranker_url or DEFAULT_RERANKER_URL

    trace: dict[str, Any] = {
        "input": item,
        "config": {
            "db_path": str(db_path),
            "index_dir": str(index_dir),
            "manifest_path": str(manifest_path),
            "top_k": top_k,
            "embedding_model": embedding_model,
            "router_mode": router_mode,
            "router_model": router_model,
            "router_confidence_threshold": router_confidence_threshold,
            "llm_model": llm_model,
            "llm_api_base": llm_api_base,
            "skip_rag": skip_rag,
            "answer": answer,
            "rag_per_query_top_k": rag_per_query_top_k,
            "rag_min_chunks_per_query": rag_min_chunks_per_query,
            "retrieval_top_k": retrieval_top_k,
            "retrieval_mode": retrieval_mode,
            "dt_planner_mode": dt_planner_mode,
            "dt_planner_model": dt_planner_model if dt_planner_mode == "llm" else None,
            "reranker_url": effective_reranker_url,
            "reranker_top_k": reranker_top_k,
            "vector_keep_top_k": vector_keep_top_k,
            "dt_rag_dense_keep_top_k": dt_rag_dense_keep_top_k,
            "dt_rag_bm25_keep_top_k": dt_rag_bm25_keep_top_k,
        },
        "errors": errors,
    }

    try:
        route = route_question(
            question,
            mode=router_mode,
            model=router_model,
            api_base=api_base,
            api_key=api_key,
            confidence_threshold=router_confidence_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        route = {
            "route": "unrelated",
            "confidence": 0.0,
            "reason": "Router failed; see errors.",
            "dt_entities": {"experiment": None, "bearing_id": None, "metric": None},
            "rag_query": "",
            "router": {"mode": router_mode, "error": str(exc)},
        }
        errors.append({"component": "router", "error": str(exc)})
    trace["route"] = route

    route_name = route.get("route")
    dt_action = infer_dt_action(question, route) if route_name in {"dt_only", "dt_rag"} else None
    trace["tool_plan"] = {
        "route": route_name,
        "dt_required": route_name in {"dt_only", "dt_rag"},
        "rag_required": route_name in {"rag_only", "dt_rag"},
        "dt_action": dt_action,
        "dt_planner_mode": dt_planner_mode,
        "original_rag_query": route.get("rag_query") or question if route_name in {"rag_only", "dt_rag"} else "",
        "rag_query": route.get("rag_query") or question if route_name in {"rag_only", "dt_rag"} else "",
        "rag_query_rewrite": "none",
        "rag_queries": [],
    }

    dt_result = None
    if trace["tool_plan"]["dt_required"]:
        try:
            if dt_planner_mode == "llm":
                dt_result = run_dt_with_llm_planner(
                    question=question,
                    db_path=db_path,
                    model=dt_planner_model,
                    api_base=dt_planner_api_base,
                    api_key=dt_planner_api_key,
                    timeout=dt_planner_timeout,
                )
                planner = dt_result.get("planner") or {}
                trace["tool_plan"]["dt_action"] = dt_result.get("action")
                trace["tool_plan"]["planned_dt_query"] = planner.get("planned_dt_query")
                trace["tool_plan"]["raw_planner_output"] = planner.get("raw_planner_output")
                if dt_result.get("errors"):
                    errors.append({"component": "dt_planner", "error": "; ".join(dt_result.get("errors") or [])})
            else:
                dt_result = run_dt(question, route, db_path)
        except Exception as exc:  # noqa: BLE001
            errors.append({"component": "dt", "error": str(exc)})
    trace["dt_result"] = dt_result

    retrieval_plan = None
    if route_name == "dt_rag" and dt_result:
        retrieval_plan = build_dt_rag_retrieval_plan(question, dt_result)
        trace["tool_plan"]["rag_query"] = retrieval_plan.get("rag_query") or " || ".join(item["query"] for item in retrieval_plan["rag_queries"])
        trace["tool_plan"]["rag_queries"] = retrieval_plan["rag_queries"]
        trace["tool_plan"]["rag_query_rewrite"] = retrieval_plan.get("retrieval_strategy", "single_dt_augmented_query")
    trace["dt_rag_retrieval_plan"] = retrieval_plan

    rag_result = None
    if trace["tool_plan"]["rag_required"] and not skip_rag:
        try:
            if route_name == "dt_rag" and retrieval_plan:
                rag_result = run_rag(
                    retrieval_plan["rag_query"],
                    index_dir=index_dir,
                    manifest_path=manifest_path,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    api_base=api_base,
                    api_key=api_key,
                    retrieval_top_k=retrieval_top_k,
                    reranker_url=effective_reranker_url,
                    reranker_top_k=reranker_top_k,
                    reranker_timeout=reranker_timeout,
                    vector_keep_top_k=vector_keep_top_k,
                    retrieval_mode=retrieval_mode,
                    dense_keep_top_k=dt_rag_dense_keep_top_k,
                    bm25_keep_top_k=dt_rag_bm25_keep_top_k,
                )
            else:
                rag_result = run_rag(
                    trace["tool_plan"]["rag_query"],
                    index_dir=index_dir,
                    manifest_path=manifest_path,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    api_base=api_base,
                    api_key=api_key,
                    retrieval_top_k=retrieval_top_k,
                    reranker_url=effective_reranker_url,
                    reranker_top_k=reranker_top_k,
                    reranker_timeout=reranker_timeout,
                    vector_keep_top_k=vector_keep_top_k,
                    retrieval_mode=retrieval_mode,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"component": "rag", "error": str(exc)})
    trace["rag_result"] = rag_result

    deterministic_answer = make_deterministic_answer(question, route, dt_result, rag_result)
    evidence_plan = build_evidence_plan(question, dt_result, rag_result, retrieval_plan)
    trace["deterministic_answer"] = deterministic_answer
    trace["evidence_plan"] = evidence_plan
    trace["llm_prompt"] = build_llm_prompt(
        question, route, dt_result, rag_result, deterministic_answer, retrieval_plan, evidence_plan
    )

    llm_answer = None
    if answer:
        try:
            llm_answer = generate_llm_answer(
                question=question,
                route=route,
                dt_result=dt_result,
                rag_result=rag_result,
                deterministic_answer=deterministic_answer,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
                retrieval_plan=retrieval_plan,
                evidence_plan=evidence_plan,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"component": "llm", "error": str(exc)})
    trace["llm_answer"] = llm_answer
    trace["final_answer"] = llm_answer.get("answer") if llm_answer and llm_answer.get("answer") else deterministic_answer
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one question through router, DT, RAG, and optional LLM answer generation.")
    parser.add_argument("--question-id", default="dt_rag_005")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--question", default="", help="Use a custom question instead of --question-id.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--llm-api-base", default=None, help="Optional OpenAI-compatible base URL for final answer model, e.g. Ollama /v1.")
    parser.add_argument("--llm-api-key", default=None, help="Optional API key for final answer model.")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--rag-per-query-top-k", type=int, default=3)
    parser.add_argument("--rag-min-chunks-per-query", type=int, default=1)
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
    parser.add_argument("--dt-rag-dense-keep-top-k", type=int, default=0)
    parser.add_argument("--dt-rag-bm25-keep-top-k", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    if args.question:
        item = {
            "id": "custom",
            "question": args.question,
            "expected_route": None,
            "expected_tools": [],
            "expected_keywords": [],
        }
    else:
        item = load_eval_item(args.questions, args.question_id)

    trace = trace_question_flow(
        item=item,
        db_path=args.db,
        index_dir=args.index_dir,
        manifest_path=args.manifest,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        router_mode=args.router_mode,
        router_model=args.router_model,
        router_confidence_threshold=args.router_confidence_threshold,
        llm_model=args.llm_model,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        skip_rag=args.skip_rag,
        answer=args.answer,
        rag_per_query_top_k=args.rag_per_query_top_k,
        rag_min_chunks_per_query=args.rag_min_chunks_per_query,
        retrieval_top_k=args.retrieval_top_k,
        reranker_url=args.reranker_url,
        reranker_top_k=args.reranker_top_k,
        reranker_timeout=args.reranker_timeout,
        vector_keep_top_k=args.vector_keep_top_k,
        retrieval_mode=args.retrieval_mode,
        dt_planner_mode=args.dt_planner_mode,
        dt_planner_model=args.dt_planner_model,
        dt_planner_api_base=args.dt_planner_api_base,
        dt_planner_api_key=args.dt_planner_api_key,
        dt_planner_timeout=args.dt_planner_timeout,
        dt_rag_dense_keep_top_k=args.dt_rag_dense_keep_top_k,
        dt_rag_bm25_keep_top_k=args.dt_rag_bm25_keep_top_k,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(trace), encoding="utf-8")
    print(json.dumps({
        "question_id": trace["input"].get("id"),
        "route": trace.get("route", {}).get("route"),
        "dt_action": trace.get("tool_plan", {}).get("dt_action"),
        "dt_planner_mode": trace.get("tool_plan", {}).get("dt_planner_mode"),
        "rag_chunk_count": len((trace.get("rag_result") or {}).get("chunks", [])),
        "has_llm_answer": trace.get("llm_answer") is not None,
        "error_count": len(trace.get("errors", [])),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
