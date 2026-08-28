#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import nest_asyncio
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[0]
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
    answer_question,
    resolve_ollama_openai_base,
)
from http_embedding import make_langchain_http_embeddings  # noqa: E402

nest_asyncio.apply()

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_REFERENCES = AGENT_DIR / "eval" / "dt_rag_reference_answers_gold_v1.json"
DEFAULT_DATASET = AGENT_DIR / "eval" / "dt_rag_ragas_dataset.jsonl"
DEFAULT_RESULTS = AGENT_DIR / "eval" / "dt_rag_ragas_results.json"
DEFAULT_SUMMARY = AGENT_DIR / "eval" / "dt_rag_ragas_summary.csv"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be a JSON object: {path}")
            rows.append(row)
    return rows


def load_dt_rag_questions(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Question file must be a JSON list.")
    return [row for row in rows if row.get("expected_route") == "dt_rag"]


def load_reference_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Reference file must be a JSON list.")
    return {str(row["id"]): row for row in rows}


def resolve_api_base(cli_value: Optional[str], manifest_path: Optional[Path]) -> Optional[str]:
    manifest = {}
    if manifest_path and manifest_path.exists():
        manifest = load_json(manifest_path)
    embedding = manifest.get("embedding", {}) if isinstance(manifest, dict) else {}
    return (
        cli_value
        or embedding.get("api_base")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )


def get_openai_api_key(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or os.environ.get("OPENAI_API_KEY")


def model_uses_ollama(model: str) -> bool:
    return "llama" in model.lower()


def resolve_judge_api_base(cli_value: Optional[str], model: str, fallback_api_base: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value
    if model_uses_ollama(model):
        return resolve_ollama_openai_base() or fallback_api_base
    return fallback_api_base


def resolve_judge_api_key(cli_value: Optional[str], model: str, fallback_api_key: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value
    if model_uses_ollama(model):
        return os.environ.get("OLLAMA_API_KEY") or fallback_api_key
    return fallback_api_key


def make_ragas_llm(api_key: Optional[str], api_base: Optional[str], model: str, timeout: float, max_retries: int):
    if not api_key:
        raise RuntimeError(f"API key is required for RAGAS judge model: {model}")
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.0,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if api_base:
        kwargs["base_url"] = api_base
    return ChatOpenAI(**kwargs)


def make_ragas_embeddings(api_key: Optional[str], api_base: Optional[str], model: str, timeout: float, max_retries: int):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --ragas-embedding-provider openai.")
    kwargs = {
        "model": model,
        "api_key": api_key,
        "chunk_size": 16,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAIEmbeddings(**kwargs)


def make_ragas_embedding_model(args: argparse.Namespace, api_key: Optional[str], api_base: Optional[str]):
    if args.ragas_embedding_provider == "http":
        return make_langchain_http_embeddings(
            api_url=args.ragas_embedding_api_url,
            api_key=args.ragas_embedding_api_key,
            batch_size=args.ragas_embedding_batch_size,
            timeout=args.request_timeout,
        )
    return make_ragas_embeddings(api_key, api_base, args.embedding_model, args.request_timeout, args.max_retries)


def get_retrieval_embedding_metadata(manifest_path: Path, cli_embedding_model: str, api_base: Optional[str]) -> dict[str, Any]:
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    embedding = manifest.get("embedding", {}) if isinstance(manifest, dict) else {}
    provider = embedding.get("provider") or "openai"
    return {
        "retrieval_embedding": {
            "provider": provider,
            "model": embedding.get("model") or cli_embedding_model,
            "api_base": embedding.get("api_url") or embedding.get("api_base") or api_base,
            "dimensions": embedding.get("dimensions"),
            "manifest": str(manifest_path),
        }
    }


def dt_context_text(raw_result: dict[str, Any]) -> str:
    dt_result = raw_result.get("dt_result") or {}
    action = dt_result.get("action")
    result = dt_result.get("result") if isinstance(dt_result, dict) else {}
    result = result if isinstance(result, dict) else {}
    fields = {
        "action": action,
        "experiment": result.get("experiment"),
        "bearing_id": result.get("bearing_id"),
        "timestamp": result.get("timestamp"),
        "sequence_index": result.get("sequence_index"),
        "health_state": result.get("health_state"),
        "features": result.get("features"),
        "evidence": result.get("evidence") or result.get("latest_evidence"),
        "time_policy": result.get("time_policy"),
        "time_policy_description": result.get("time_policy_description"),
        "resolved_time_mode": result.get("resolved_time_mode"),
    }
    return "DT context:\n" + json.dumps(fields, ensure_ascii=False, indent=2)


def context_from_chunk(chunk: dict[str, Any]) -> str:
    prefix = []
    if chunk.get("section"):
        prefix.append(f"SKF section: {chunk.get('section')}")
    if chunk.get("page_start"):
        prefix.append(f"SKF pages: {chunk.get('page_start')}-{chunk.get('page_end')}")
    text = str(chunk.get("text") or "").strip()
    return "\n".join(prefix + [text]).strip()



def run_one_dt_rag(
    item: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    api_base: Optional[str],
    api_key: str,
) -> dict[str, Any]:
    raw_result = answer_question(
        question=item["question"],
        db_path=args.db,
        index_dir=args.index_dir,
        manifest_path=args.manifest,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        api_base=api_base,
        api_key=api_key,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        generate_answer=args.answer,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
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
        dt_rag_dense_keep_top_k=args.dt_rag_dense_keep_top_k,
        dt_rag_bm25_keep_top_k=args.dt_rag_bm25_keep_top_k,
        dt_rag_reranker_query_mode=args.dt_rag_reranker_query_mode,
        force_route=args.force_route,
    )
    rag_result = raw_result.get("rag_result") or {"chunks": []}
    contexts = [dt_context_text(raw_result)]
    contexts.extend(context_from_chunk(chunk) for chunk in rag_result.get("chunks", []) if chunk.get("text"))
    answer = raw_result.get("answer") or raw_result.get("deterministic_answer") or ""
    retrieval_plan = raw_result.get("dt_rag_retrieval_plan") or {}
    llm_answer = raw_result.get("llm_answer") or {}
    llm_errors = [
        item.get("error")
        for item in (raw_result.get("errors") or [])
        if item.get("component") == "llm"
    ]
    llm_used = bool(llm_answer.get("used_llm"))
    answer_source = "llm" if llm_used else ("deterministic_fallback_after_llm_error" if llm_errors else "deterministic")
    source_chunk_ids = reference.get("source_chunk_ids", [])
    if not source_chunk_ids and reference.get("gold_chunk_id"):
        source_chunk_ids = [reference.get("gold_chunk_id")]
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": answer,
        "answer_source": answer_source,
        "llm_used": llm_used,
        "llm_error": "; ".join(str(error) for error in llm_errors) if llm_errors else None,
        "contexts": contexts,
        "ground_truth": reference.get("ground_truth", ""),
        "expected_route": item.get("expected_route"),
        "actual_route": raw_result.get("route", {}).get("route"),
        "retrieval_strategy": retrieval_plan.get("retrieval_strategy"),
        "rag_query": retrieval_plan.get("rag_query") or rag_result.get("query"),
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


def compute_gold_retrieval_metrics(row: dict[str, Any]) -> dict[str, Any]:
    gold_chunk_id = row.get("gold_chunk_id")
    if not gold_chunk_id:
        source_chunk_ids = row.get("source_chunk_ids") or []
        gold_chunk_id = source_chunk_ids[0] if source_chunk_ids else None
    retrieved_chunk_ids = [chunk_id for chunk_id in (row.get("retrieved_chunk_ids") or []) if chunk_id]
    rank = None
    if gold_chunk_id:
        for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id == gold_chunk_id:
                rank = index
                break
    return {
        "gold_chunk_id": gold_chunk_id,
        "gold_rank": rank,
        "hit_at_1": 1.0 if rank == 1 else 0.0,
        "hit_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "hit_at_10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def attach_gold_retrieval_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item.update(compute_gold_retrieval_metrics(item))
        output.append(item)
    return output


def build_eval_rows(
    questions: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_base: Optional[str],
    api_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(questions, start=1):
        item_id = str(item.get("id"))
        if item_id not in reference_map:
            raise KeyError(f"Missing DT+RAG reference answer for question id: {item_id}")
        print(f"[DT+RAG] {idx}/{len(questions)} {item_id}: {item.get('question')}", flush=True)
        rows.append(run_one_dt_rag(item, reference_map[item_id], args, api_base, api_key))
    return attach_gold_retrieval_metrics(rows)


def run_ragas_safe(rows: list[dict[str, Any]], metrics, ragas_llm, ragas_embeddings, run_config: RunConfig, args: argparse.Namespace) -> pd.DataFrame:
    output_rows = []
    for row_index, row in enumerate(rows, start=1):
        record = {
            "id": row.get("id"),
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
            "retrieval_strategy": row.get("retrieval_strategy"),
            "rag_query": row.get("rag_query"),
            "retrieved_chunk_ids": row.get("retrieved_chunk_ids"),
            "source_chunk_ids": row.get("source_chunk_ids"),
            "gold_chunk_id": row.get("gold_chunk_id"),
            "gold_rank": row.get("gold_rank"),
            "hit_at_1": row.get("hit_at_1"),
            "hit_at_5": row.get("hit_at_5"),
            "hit_at_10": row.get("hit_at_10"),
            "mrr": row.get("mrr"),
            "answer_source": row.get("answer_source"),
            "llm_used": row.get("llm_used"),
            "llm_error": row.get("llm_error"),
        }
        for metric in metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__)
            print(f"[RAGAS] sample {row_index}/{len(rows)} metric={metric_name}", flush=True)
            single_dataset = Dataset.from_list(
                [
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                        "contexts": row["contexts"],
                        "ground_truth": row["ground_truth"],
                    }
                ]
            )
            last_error = None
            for attempt in range(1, max(1, args.metric_retry_attempts) + 1):
                try:
                    metric_result = evaluate(
                        single_dataset,
                        metrics=[metric],
                        llm=ragas_llm,
                        embeddings=ragas_embeddings,
                        run_config=run_config,
                        raise_exceptions=False,
                    )
                    metric_df = metric_result.to_pandas()
                    if metric_name in metric_df.columns:
                        record[metric_name] = metric_df.loc[0, metric_name]
                    else:
                        metric_columns = [
                            col
                            for col in metric_df.columns
                            if col not in {"question", "answer", "contexts", "ground_truth"}
                        ]
                        record[metric_name] = metric_df.loc[0, metric_columns[0]] if metric_columns else None
                    record[f"{metric_name}_attempts"] = attempt
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    if attempt < max(1, args.metric_retry_attempts):
                        print(
                            f"[RAGAS] sample {row_index}/{len(rows)} metric={metric_name} retry {attempt}/{args.metric_retry_attempts} after error: {last_error}",
                            flush=True,
                        )
                        time.sleep(max(0.0, args.metric_retry_sleep))
            if last_error is not None:
                record[metric_name] = None
                record[f"{metric_name}_error"] = last_error
                record[f"{metric_name}_attempts"] = max(1, args.metric_retry_attempts)
        output_rows.append(record)
    return pd.DataFrame(output_rows)


def summarize_metric_columns(df: pd.DataFrame, metrics, include_hit_at_10: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": int(len(df))}
    if "llm_used" in df.columns:
        used = df["llm_used"].astype(bool)
        summary["llm_success_count"] = int(used.sum())
        summary["llm_success_rate"] = float(used.mean()) if len(used) else None
    if "llm_error" in df.columns:
        summary["llm_error_count"] = int(df["llm_error"].notna().sum())
    retrieval_metrics = ["hit_at_1", "hit_at_5", "mrr"]
    if include_hit_at_10:
        retrieval_metrics.insert(2, "hit_at_10")
    for metric_name in retrieval_metrics:
        if metric_name in df.columns:
            values = pd.to_numeric(df[metric_name], errors="coerce")
            summary[metric_name] = None if values.dropna().empty else float(values.mean())
    for metric in metrics:
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        if metric_name not in df.columns:
            summary[metric_name] = None
            continue
        values = pd.to_numeric(df[metric_name], errors="coerce")
        summary[metric_name] = None if values.dropna().empty else float(values.mean())
    return summary


def write_summary_csv(path: Path, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    summary_path = path.with_name(path.stem + "_means.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DT+RAG questions with RAGAS.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--reuse-dataset", type=Path, default=None, help="Reuse an existing DT+RAG JSONL dataset and rerun only RAGAS metrics.")
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Model used to generate final agent answers.")
    parser.add_argument("--llm-timeout", type=float, default=180.0, help="Timeout in seconds for final answer generation.")
    parser.add_argument("--judge-model", default=DEFAULT_LLM_MODEL, help="Model used by RAGAS metrics.")
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--router-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--force-route", choices=["unrelated", "dt_only", "rag_only", "dt_rag"], default=None, help="Force a route for controlled evaluation/debugging.")
    parser.add_argument("--answer", action="store_true", help="Generate final natural-language answers with an LLM.")
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--llm-api-base", default=None, help="Optional OpenAI-compatible base URL for answer model, e.g. Ollama /v1.")
    parser.add_argument("--llm-api-key", default=None, help="Optional API key for answer model.")
    parser.add_argument("--judge-api-base", default=None, help="Optional OpenAI-compatible base URL for RAGAS judge model.")
    parser.add_argument("--judge-api-key", default=None, help="Optional API key for RAGAS judge model.")
    parser.add_argument("--ragas-embedding-provider", choices=["openai", "http"], default="openai")
    parser.add_argument("--ragas-embedding-api-url", default=None, help="HTTP embedding URL for RAGAS, defaults to EMBEDDING_URL.")
    parser.add_argument("--ragas-embedding-api-key", default=None, help="Optional bearer token for HTTP RAGAS embedding service.")
    parser.add_argument("--ragas-embedding-batch-size", type=int, default=16)
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="Initial vector retrieval count before optional reranking.")
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default="dense")
    parser.add_argument("--dt-planner-mode", choices=["legacy", "llm"], default="legacy", help="DT query planner used inside the agent for dt_rag routes.")
    parser.add_argument("--dt-planner-model", default="llama3.3:70b")
    parser.add_argument("--dt-planner-api-base", default=None)
    parser.add_argument("--dt-planner-api-key", default=None)
    parser.add_argument("--dt-planner-timeout", type=float, default=120.0)
    parser.add_argument("--reranker-url", default=None, help="HTTP reranker service URL.")
    parser.add_argument("--reranker-top-k", type=int, default=None, help="Number of chunks kept after reranking. Defaults to --top-k.")
    parser.add_argument("--reranker-timeout", type=float, default=30.0)
    parser.add_argument("--vector-keep-top-k", type=int, default=0, help="Force-keep this many top vector hits before filling the rest with reranked chunks.")
    parser.add_argument("--dt-rag-dense-keep-top-k", type=int, default=0, help="For DT+RAG hybrid_reranker only, keep this many dense hits before reranking.")
    parser.add_argument("--dt-rag-bm25-keep-top-k", type=int, default=0, help="For DT+RAG hybrid_reranker only, keep this many BM25 hits before reranking.")
    parser.add_argument("--dt-rag-reranker-query-mode", choices=["full", "intent"], default="full", help="For DT+RAG hybrid_reranker, use the full augmented query or a shorter intent-focused query for reranking.")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--ragas-workers", type=int, default=1)
    parser.add_argument("--metric-retry-attempts", type=int, default=3, help="Retry each failed RAGAS metric with the same full input before recording an error/NaN.")
    parser.add_argument("--metric-retry-sleep", type=float, default=5.0, help="Seconds to wait between per-metric RAGAS retries.")
    parser.add_argument("--skip-context-recall", action="store_true", help="Skip RAGAS context_recall to speed up gold-chunk DT+RAG evaluation.")
    parser.add_argument("--skip-hit-at-10", action="store_true", help="Do not include hit_at_10 in summaries; hit_at_1/hit_at_5/MRR are still computed.")
    parser.add_argument("--unsafe-batch-ragas", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = get_openai_api_key(args.openai_api_key)
    api_base = resolve_api_base(args.openai_api_base, args.manifest)
    judge_api_base = None if args.skip_ragas else resolve_judge_api_base(args.judge_api_base, args.judge_model, api_base)
    retrieval_embedding_metadata = get_retrieval_embedding_metadata(args.manifest, args.embedding_model, api_base)
    judge_api_key = None if args.skip_ragas else resolve_judge_api_key(args.judge_api_key, args.judge_model, api_key)
    args.effective_reranker_url = args.reranker_url or DEFAULT_RERANKER_URL
    rows = None
    questions = []
    reference_map = {}
    if args.reuse_dataset:
        rows = attach_gold_retrieval_metrics(load_jsonl(args.reuse_dataset))
        if args.limit is not None:
            rows = rows[: args.limit]
        questions = [{"id": row.get("id"), "question": row.get("question"), "expected_route": row.get("expected_route")} for row in rows]
    else:
        questions = load_dt_rag_questions(args.questions)
        if args.limit is not None:
            questions = questions[: args.limit]
        reference_map = load_reference_map(args.references)

    print(f"[INFO] API base: {api_base}", flush=True)
    if args.skip_ragas:
        print("[INFO] Judge model: skipped because --skip-ragas is enabled.", flush=True)
    else:
        print(f"[INFO] Judge model: {args.judge_model}", flush=True)
        print(f"[INFO] Judge API base: {judge_api_base}", flush=True)
    print(f"[INFO] Reranker URL: {args.effective_reranker_url}", flush=True)
    print(f"[INFO] DT+RAG questions: {len(questions)}", flush=True)
    if rows is None:
        rows = build_eval_rows(questions, reference_map, args, api_base, api_key)
        save_jsonl(rows, args.dataset_output)
        print(f"[OK] Saved DT+RAG RAGAS dataset -> {args.dataset_output}", flush=True)
    else:
        if args.dataset_output != args.reuse_dataset:
            save_jsonl(rows, args.dataset_output)
            print(f"[OK] Copied reused DT+RAG RAGAS dataset -> {args.dataset_output}", flush=True)
        else:
            print(f"[OK] Reusing DT+RAG RAGAS dataset -> {args.reuse_dataset}", flush=True)

    if args.skip_ragas:
        payload = {
            "summary": {"total": len(rows), "ragas_skipped": True},
            "rows": rows,
            "dataset_output": str(args.dataset_output),
            "top_k": args.top_k,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_mode": args.retrieval_mode,
            "force_route": args.force_route,
        "dt_planner_mode": args.dt_planner_mode,
        "dt_planner_model": args.dt_planner_model if args.dt_planner_mode == "llm" else None,
            "reranker_url": args.effective_reranker_url,
            "reranker_top_k": args.reranker_top_k,
            "vector_keep_top_k": args.vector_keep_top_k,
            "dt_rag_dense_keep_top_k": args.dt_rag_dense_keep_top_k,
            "dt_rag_bm25_keep_top_k": args.dt_rag_bm25_keep_top_k,
            "dt_rag_reranker_query_mode": args.dt_rag_reranker_query_mode,
            "llm_model": args.llm_model,
            "llm_timeout": args.llm_timeout,
            "judge_model": None,
            "judge_api_base": None,
            "ragas_embedding_provider": None,
            "ragas_embedding_api_base": None,
            **retrieval_embedding_metadata,
            "api_base": api_base,
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[OK] Skipped RAGAS metrics.", flush=True)
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
        print(f"[INFO] Running RAGAS in batch mode with max_workers={args.ragas_workers}.", flush=True)
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
        "top_k": args.top_k,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_mode": args.retrieval_mode,
        "force_route": args.force_route,
        "dt_planner_mode": args.dt_planner_mode,
        "dt_planner_model": args.dt_planner_model if args.dt_planner_mode == "llm" else None,
        "reranker_url": args.effective_reranker_url,
        "reranker_top_k": args.reranker_top_k,
        "vector_keep_top_k": args.vector_keep_top_k,
        "dt_rag_dense_keep_top_k": args.dt_rag_dense_keep_top_k,
        "dt_rag_bm25_keep_top_k": args.dt_rag_bm25_keep_top_k,
        "dt_rag_reranker_query_mode": args.dt_rag_reranker_query_mode,
        "llm_model": args.llm_model,
        "llm_timeout": args.llm_timeout,
        "judge_model": args.judge_model,
        **retrieval_embedding_metadata,
        "api_base": api_base,
        "judge_api_base": judge_api_base,
        "ragas_embedding_provider": args.ragas_embedding_provider,
        "ragas_embedding_api_base": api_base if args.ragas_embedding_provider == "openai" else (args.ragas_embedding_api_url or os.environ.get("EMBEDDING_URL") or os.environ.get("BGE_EMBEDDING_URL")),
        "metric_retry_attempts": args.metric_retry_attempts,
        "metric_retry_sleep": args.metric_retry_sleep,
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
