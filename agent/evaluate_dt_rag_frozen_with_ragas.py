#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import nest_asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from bearing_agent import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_RERANKER_URL,
    RETRIEVAL_MODES,
    build_dt_rag_reranker_query,
    build_evidence_plan,
    generate_llm_answer,
    make_deterministic_answer,
    resolve_ollama_openai_base,
    run_rag,
)
from evaluate_dt_rag_with_ragas import (  # noqa: E402
    attach_gold_retrieval_metrics,
    context_from_chunk,
    dt_context_text,
    get_openai_api_key,
    get_retrieval_embedding_metadata,
    load_json,
    load_jsonl,
    load_reference_map,
    make_ragas_embedding_model,
    make_ragas_llm,
    resolve_api_base,
    resolve_judge_api_base,
    resolve_judge_api_key,
    run_ragas_safe,
    save_jsonl,
    summarize_metric_columns,
    write_summary_csv,
)

nest_asyncio.apply()

DEFAULT_FROZEN_INPUTS = AGENT_DIR / "eval" / "dt_rag_frozen_inputs_v1.jsonl"
DEFAULT_REFERENCES = AGENT_DIR / "eval" / "dt_rag_reference_answers_gold_v1.json"
DEFAULT_DATASET = AGENT_DIR / "eval" / "dt_rag_frozen_ragas_dataset.jsonl"
DEFAULT_RESULTS = AGENT_DIR / "eval" / "dt_rag_frozen_ragas_results.json"
DEFAULT_SUMMARY = AGENT_DIR / "eval" / "dt_rag_frozen_ragas_summary.csv"


def model_uses_ollama(model: str) -> bool:
    return "llama" in model.lower()


def resolve_answer_api_base(cli_value: Optional[str], model: str) -> Optional[str]:
    if cli_value:
        return cli_value
    if model_uses_ollama(model):
        return resolve_ollama_openai_base()
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")


def resolve_answer_api_key(cli_value: Optional[str], model: str) -> Optional[str]:
    if cli_value:
        return cli_value
    if model_uses_ollama(model):
        return os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def select_frozen_rows(path: Path, limit: Optional[int]) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    rows = [row for row in rows if row.get("expected_route") == "dt_rag" or row.get("forced_route") == "dt_rag"]
    if limit is not None:
        rows = rows[:limit]
    return rows


def make_forced_route(question: str, rag_query: str) -> dict[str, Any]:
    return {
        "route": "dt_rag",
        "confidence": 1.0,
        "reason": "Forced DT+RAG route with frozen DT evidence and frozen augmented query.",
        "dt_entities": {},
        "rag_query": rag_query,
        "router": {
            "mode": "frozen_dt_rag",
            "forced_route": "dt_rag",
            "question": question,
        },
    }


def source_chunk_ids_from_reference(reference: dict[str, Any]) -> list[str]:
    source_chunk_ids = reference.get("source_chunk_ids") or []
    if not source_chunk_ids and reference.get("gold_chunk_id"):
        source_chunk_ids = [reference["gold_chunk_id"]]
    return [str(chunk_id) for chunk_id in source_chunk_ids if chunk_id]


def run_one_frozen_dt_rag(
    frozen: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    api_base: Optional[str],
    api_key: Optional[str],
    answer_api_base: Optional[str],
    answer_api_key: Optional[str],
) -> dict[str, Any]:
    question = str(frozen.get("question") or "")
    retrieval_plan = frozen.get("dt_rag_retrieval_plan") or {}
    rag_query = str(frozen.get("rag_query") or retrieval_plan.get("rag_query") or question)
    dt_result = frozen.get("dt_result") or (frozen.get("raw_result") or {}).get("dt_result")
    route = make_forced_route(question, rag_query)

    plan_for_reranker = dict(retrieval_plan)
    reranker_query = None
    if args.dt_rag_reranker_query_mode == "intent":
        reranker_query = build_dt_rag_reranker_query(question, plan_for_reranker)
        plan_for_reranker["reranker_query"] = reranker_query
        plan_for_reranker["reranker_query_mode"] = "intent"
    else:
        plan_for_reranker["reranker_query_mode"] = "full"

    errors: list[dict[str, Any]] = []
    try:
        rag_result = run_rag(
            query=rag_query,
            index_dir=args.index_dir,
            manifest_path=args.manifest,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            api_base=api_base,
            api_key=api_key,
            retrieval_top_k=args.retrieval_top_k,
            retrieval_mode=args.retrieval_mode,
            reranker_url=args.effective_reranker_url,
            reranker_top_k=args.reranker_top_k,
            reranker_timeout=args.reranker_timeout,
            vector_keep_top_k=args.vector_keep_top_k,
            reranker_query=reranker_query,
            dense_keep_top_k=args.dt_rag_dense_keep_top_k,
            bm25_keep_top_k=args.dt_rag_bm25_keep_top_k,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append({"component": "rag", "error": str(exc)})
        rag_result = {
            "query": rag_query,
            "top_k": args.top_k,
            "retrieval_mode": args.retrieval_mode,
            "chunks": [],
            "error": str(exc),
        }

    deterministic_answer = make_deterministic_answer(question, route, dt_result, rag_result)
    evidence_plan = build_evidence_plan(question, dt_result, rag_result, plan_for_reranker)
    answer = deterministic_answer
    llm_answer: dict[str, Any] = {"used_llm": False}
    if args.answer:
        try:
            print(f"[ANSWER] {frozen.get('id')} model={args.llm_model}", flush=True)
            llm_answer = generate_llm_answer(
                question=question,
                route=route,
                dt_result=dt_result,
                rag_result=rag_result,
                deterministic_answer=deterministic_answer,
                model=args.llm_model,
                api_base=answer_api_base,
                api_key=answer_api_key,
                timeout=args.llm_timeout,
                retrieval_plan=plan_for_reranker,
                evidence_plan=evidence_plan,
            )
            answer = llm_answer.get("answer") or deterministic_answer
        except Exception as exc:  # noqa: BLE001
            errors.append({"component": "llm", "error": str(exc)})
            llm_answer = {"used_llm": False, "error": str(exc)}

    contexts = [dt_context_text({"dt_result": dt_result})]
    contexts.extend(context_from_chunk(chunk) for chunk in rag_result.get("chunks", []) if chunk.get("text"))
    source_chunk_ids = source_chunk_ids_from_reference(reference)

    raw_result = {
        "question": question,
        "route": route,
        "dt_result": dt_result,
        "dt_rag_retrieval_plan": plan_for_reranker,
        "rag_result": rag_result,
        "deterministic_answer": deterministic_answer,
        "llm_answer": llm_answer,
        "answer": answer,
        "evidence_plan": evidence_plan,
        "errors": errors,
        "frozen_input_id": frozen.get("id"),
    }

    return {
        "id": frozen.get("id"),
        "question": question,
        "answer": answer,
        "answer_source": "llm" if llm_answer.get("used_llm") else "deterministic",
        "llm_used": bool(llm_answer.get("used_llm")),
        "llm_error": llm_answer.get("error"),
        "contexts": contexts,
        "ground_truth": reference.get("ground_truth", ""),
        "expected_route": "dt_rag",
        "actual_route": "dt_rag",
        "retrieval_strategy": retrieval_plan.get("retrieval_strategy"),
        "rag_query": rag_query,
        "reranker_query": reranker_query,
        "source_chunk_ids": source_chunk_ids,
        "gold_chunk_id": reference.get("gold_chunk_id") or (source_chunk_ids[0] if source_chunk_ids else None),
        "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in rag_result.get("chunks", [])],
        "retrieved_sources": [
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "section": chunk.get("section"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
            for chunk in rag_result.get("chunks", [])
        ],
        "raw_result": raw_result,
    }


def build_eval_rows(
    frozen_rows: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_base: Optional[str],
    api_key: Optional[str],
    answer_api_base: Optional[str],
    answer_api_key: Optional[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, frozen in enumerate(frozen_rows, start=1):
        item_id = str(frozen.get("id"))
        if item_id not in reference_map:
            raise KeyError(f"Missing DT+RAG reference answer for frozen input id: {item_id}")
        print(f"[FROZEN DT+RAG] {index}/{len(frozen_rows)} {item_id}: {frozen.get('question')}", flush=True)
        rows.append(
            run_one_frozen_dt_rag(
                frozen=frozen,
                reference=reference_map[item_id],
                args=args,
                api_base=api_base,
                api_key=api_key,
                answer_api_base=answer_api_base,
                answer_api_key=answer_api_key,
            )
        )
    return attach_gold_retrieval_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DT+RAG retrieval methods from frozen DT evidence and frozen augmented RAG queries."
    )
    parser.add_argument("--frozen-inputs", type=Path, default=DEFAULT_FROZEN_INPUTS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--judge-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--ragas-embedding-provider", choices=["openai", "http"], default="openai")
    parser.add_argument("--ragas-embedding-api-url", default=None)
    parser.add_argument("--ragas-embedding-api-key", default=None)
    parser.add_argument("--ragas-embedding-batch-size", type=int, default=16)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default="dense")
    parser.add_argument("--reranker-url", default=None)
    parser.add_argument("--reranker-top-k", type=int, default=None)
    parser.add_argument("--reranker-timeout", type=float, default=60.0)
    parser.add_argument("--vector-keep-top-k", type=int, default=0)
    parser.add_argument("--dt-rag-dense-keep-top-k", type=int, default=0)
    parser.add_argument("--dt-rag-bm25-keep-top-k", type=int, default=0)
    parser.add_argument("--dt-rag-reranker-query-mode", choices=["full", "intent"], default="full")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--ragas-workers", type=int, default=1)
    parser.add_argument("--metric-retry-attempts", type=int, default=3)
    parser.add_argument("--metric-retry-sleep", type=float, default=5.0)
    parser.add_argument("--skip-context-recall", action="store_true")
    parser.add_argument("--skip-hit-at-10", action="store_true")
    parser.add_argument("--unsafe-batch-ragas", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = get_openai_api_key(args.openai_api_key)
    api_base = resolve_api_base(args.openai_api_base, args.manifest)
    answer_api_base = resolve_answer_api_base(args.llm_api_base, args.llm_model)
    answer_api_key = resolve_answer_api_key(args.llm_api_key, args.llm_model)
    judge_api_base = None if args.skip_ragas else resolve_judge_api_base(args.judge_api_base, args.judge_model, api_base)
    judge_api_key = None if args.skip_ragas else resolve_judge_api_key(args.judge_api_key, args.judge_model, api_key)
    args.effective_reranker_url = args.reranker_url or DEFAULT_RERANKER_URL

    frozen_rows = select_frozen_rows(args.frozen_inputs, args.limit)
    reference_map = load_reference_map(args.references)
    retrieval_embedding_metadata = get_retrieval_embedding_metadata(args.manifest, args.embedding_model, api_base)

    print(f"[INFO] Frozen inputs: {args.frozen_inputs}", flush=True)
    print(f"[INFO] DT+RAG questions: {len(frozen_rows)}", flush=True)
    print(f"[INFO] Retrieval mode: {args.retrieval_mode}", flush=True)
    print(f"[INFO] Reranker query mode: {args.dt_rag_reranker_query_mode}", flush=True)
    if not args.skip_ragas:
        print(f"[INFO] Judge model: {args.judge_model}", flush=True)

    rows = build_eval_rows(
        frozen_rows=frozen_rows,
        reference_map=reference_map,
        args=args,
        api_base=api_base,
        api_key=api_key,
        answer_api_base=answer_api_base,
        answer_api_key=answer_api_key,
    )
    save_jsonl(rows, args.dataset_output)
    print(f"[OK] Saved frozen DT+RAG RAGAS dataset -> {args.dataset_output}", flush=True)

    if args.skip_ragas:
        payload = {
            "summary": {"total": len(rows), "ragas_skipped": True},
            "rows": rows,
            "dataset_output": str(args.dataset_output),
            "frozen_inputs": str(args.frozen_inputs),
            "top_k": args.top_k,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_mode": args.retrieval_mode,
            "reranker_top_k": args.reranker_top_k,
            "vector_keep_top_k": args.vector_keep_top_k,
            "dt_rag_dense_keep_top_k": args.dt_rag_dense_keep_top_k,
            "dt_rag_bm25_keep_top_k": args.dt_rag_bm25_keep_top_k,
            "dt_rag_reranker_query_mode": args.dt_rag_reranker_query_mode,
            "llm_model": args.llm_model,
            "judge_model": None,
            **retrieval_embedding_metadata,
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Results -> {args.results}", flush=True)
        return

    metrics = [faithfulness, answer_relevancy, context_precision]
    if not args.skip_context_recall:
        metrics.append(context_recall)
    ragas_llm = make_ragas_llm(judge_api_key, judge_api_base, args.judge_model, args.request_timeout, args.max_retries)
    ragas_embeddings = make_ragas_embedding_model(args, api_key, api_base)
    run_config = RunConfig(
        timeout=int(args.request_timeout),
        max_retries=args.max_retries,
        max_workers=args.ragas_workers,
    )

    if args.unsafe_batch_ragas:
        dataset = Dataset.from_list(
            [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "contexts": row["contexts"],
                    "ground_truth": row["ground_truth"],
                }
                for row in rows
            ]
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        summary = dict(result)
    else:
        print("[INFO] Running RAGAS in safe sequential mode.", flush=True)
        df = run_ragas_safe(rows, metrics, ragas_llm, ragas_embeddings, run_config, args)
        summary = summarize_metric_columns(df, metrics, include_hit_at_10=not args.skip_hit_at_10)

    write_summary_csv(args.summary, df, summary)
    payload = {
        "summary": summary,
        "rows": df.to_dict(orient="records"),
        "dataset_output": str(args.dataset_output),
        "frozen_inputs": str(args.frozen_inputs),
        "top_k": args.top_k,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_mode": args.retrieval_mode,
        "reranker_top_k": args.reranker_top_k,
        "vector_keep_top_k": args.vector_keep_top_k,
        "dt_rag_dense_keep_top_k": args.dt_rag_dense_keep_top_k,
        "dt_rag_bm25_keep_top_k": args.dt_rag_bm25_keep_top_k,
        "dt_rag_reranker_query_mode": args.dt_rag_reranker_query_mode,
        "llm_model": args.llm_model,
        "llm_timeout": args.llm_timeout,
        "judge_model": args.judge_model,
        **retrieval_embedding_metadata,
        "ragas_embedding_provider": args.ragas_embedding_provider,
        "safe_sequential_mode": not args.unsafe_batch_ragas,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RAGAS SUMMARY]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] Saved RAGAS row CSV -> {args.summary}", flush=True)
    print(f"[OK] Saved RAGAS JSON -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
