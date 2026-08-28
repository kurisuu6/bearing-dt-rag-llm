#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_RERANKER_URL,
    RETRIEVAL_MODES,
    answer_question,
    generate_llm_answer,
    make_deterministic_answer,
    resolve_ollama_openai_base,
    run_rag,
)
from http_embedding import make_langchain_http_embeddings  # noqa: E402

nest_asyncio.apply()

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_REFERENCES = AGENT_DIR / "eval" / "rag_only_reference_answers_v3_reviewed.json"
DEFAULT_DATASET = AGENT_DIR / "eval" / "rag_only_ragas_dataset.jsonl"
DEFAULT_RESULTS = AGENT_DIR / "eval" / "rag_only_ragas_results.json"
DEFAULT_SUMMARY = AGENT_DIR / "eval" / "rag_only_ragas_summary.csv"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rag_only_questions(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Question file must be a JSON list.")
    return [row for row in rows if row.get("expected_route") == "rag_only"]


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


def context_from_chunk(chunk: dict[str, Any]) -> str:
    prefix = []
    if chunk.get("section"):
        prefix.append(f"Section: {chunk.get('section')}")
    if chunk.get("page_start"):
        prefix.append(f"Pages: {chunk.get('page_start')}-{chunk.get('page_end')}")
    text = str(chunk.get("text") or "").strip()
    return "\n".join(prefix + [text]).strip()


def resolve_gold_chunk_id(reference: dict[str, Any]) -> Optional[str]:
    gold_chunk_id = reference.get("gold_chunk_id")
    if gold_chunk_id:
        return str(gold_chunk_id)
    source_chunk_ids = reference.get("source_chunk_ids") or []
    if source_chunk_ids:
        return str(source_chunk_ids[0])
    return None


def compute_gold_retrieval_metrics(gold_chunk_id: Optional[str], retrieved_chunk_ids: list[Any]) -> dict[str, Any]:
    retrieved_ids = [str(chunk_id) for chunk_id in retrieved_chunk_ids if chunk_id]
    rank = None
    if gold_chunk_id:
        for idx, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id == gold_chunk_id:
                rank = idx
                break
    return {
        "gold_chunk_id": gold_chunk_id,
        "gold_rank": rank,
        "hit_at_1": 1.0 if rank == 1 else 0.0,
        "hit_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "hit_at_10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def make_forced_rag_route(question: str) -> dict[str, Any]:
    return {
        "route": "rag_only",
        "confidence": 1.0,
        "reason": "Forced rag_only route for isolated SKF RAG evaluation.",
        "dt_entities": {"experiment": None, "bearing_id": None, "metric": None},
        "rag_query": question,
        "matched_signals": {},
        "router": {"mode": "forced_rag_only_eval"},
    }


def run_one_rag_only(
    item: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    api_base: Optional[str],
    api_key: str,
) -> dict[str, Any]:
    question = item["question"]

    if args.force_rag_only:
        route = make_forced_rag_route(question)
        rag_result = run_rag(
            query=question,
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
        )
        deterministic_answer = make_deterministic_answer(question, route, None, rag_result)
        llm_answer = None
        if args.answer:
            llm_answer = generate_llm_answer(
                question=question,
                route=route,
                dt_result=None,
                rag_result=rag_result,
                deterministic_answer=deterministic_answer,
                model=args.llm_model,
                api_base=args.llm_api_base,
                api_key=args.llm_api_key,
                timeout=args.request_timeout,
            )
        answer = llm_answer.get("answer") if llm_answer and llm_answer.get("answer") else deterministic_answer
        raw_result = {
            "question": question,
            "route": route,
            "dt_result": None,
            "rag_result": rag_result,
            "errors": [],
            "deterministic_answer": deterministic_answer,
            "llm_answer": llm_answer,
            "answer": answer,
        }
    else:
        raw_result = answer_question(
            question=question,
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
            router_mode=args.router_mode,
            router_model=args.router_model,
            router_confidence_threshold=args.router_confidence_threshold,
            retrieval_top_k=args.retrieval_top_k,
            retrieval_mode=args.retrieval_mode,
            reranker_url=args.effective_reranker_url,
            reranker_top_k=args.reranker_top_k,
            reranker_timeout=args.reranker_timeout,
            vector_keep_top_k=args.vector_keep_top_k,
        )
        rag_result = raw_result.get("rag_result") or {"chunks": []}
        answer = raw_result.get("answer") or raw_result.get("deterministic_answer") or ""

    chunks = rag_result.get("chunks", [])
    contexts = [context_from_chunk(chunk) for chunk in chunks if chunk.get("text")]
    source_chunk_ids = reference.get("source_chunk_ids", [])
    gold_chunk_id = resolve_gold_chunk_id(reference)
    retrieved_chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    row = {
        "id": item.get("id"),
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": reference.get("ground_truth", ""),
        "reference_keywords": reference.get("reference_keywords", []),
        "expected_route": item.get("expected_route"),
        "actual_route": raw_result.get("route", {}).get("route"),
        "source_chunk_ids": source_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_sources": [
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "section": chunk.get("section"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
            for chunk in chunks
        ],
        "raw_result": raw_result,
    }
    row.update(compute_gold_retrieval_metrics(gold_chunk_id, retrieved_chunk_ids))
    return row


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
            raise KeyError(f"Missing reference answer for question id: {item_id}")
        print(f"[RAG] {idx}/{len(questions)} {item_id}: {item.get('question')}", flush=True)
        rows.append(run_one_rag_only(item, reference_map[item_id], args, api_base, api_key))
    return rows


def run_ragas_safe(rows: list[dict[str, Any]], metrics, ragas_llm, ragas_embeddings, run_config: RunConfig) -> pd.DataFrame:
    output_rows = []
    for row_index, row in enumerate(rows, start=1):
        record = {
            "id": row.get("id"),
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
            "retrieved_chunk_ids": row.get("retrieved_chunk_ids"),
            "source_chunk_ids": row.get("source_chunk_ids"),
            "gold_chunk_id": row.get("gold_chunk_id"),
            "gold_rank": row.get("gold_rank"),
            "hit_at_1": row.get("hit_at_1"),
            "hit_at_5": row.get("hit_at_5"),
            "hit_at_10": row.get("hit_at_10"),
            "mrr": row.get("mrr"),
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
            except Exception as exc:  # noqa: BLE001 - store metric errors per sample
                record[metric_name] = None
                record[f"{metric_name}_error"] = repr(exc)
        output_rows.append(record)
    return pd.DataFrame(output_rows)


def summarize_metric_columns(df: pd.DataFrame, metrics, include_hit_at_10: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": int(len(df))}
    retrieval_metric_names = ["hit_at_1", "hit_at_5", "mrr"]
    if include_hit_at_10:
        retrieval_metric_names.insert(2, "hit_at_10")
    for metric_name in retrieval_metric_names:
        if metric_name not in df.columns:
            summary[metric_name] = None
            continue
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
    parser = argparse.ArgumentParser(description="Evaluate 50 SKF rag_only questions with RAGAS.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Model used to generate RAG answers.")
    parser.add_argument("--judge-model", default=DEFAULT_LLM_MODEL, help="Model used by RAGAS metrics.")
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--router-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--force-rag-only", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--reranker-url", default=None, help="HTTP reranker service URL.")
    parser.add_argument("--reranker-top-k", type=int, default=None, help="Number of chunks kept after reranking. Defaults to --top-k.")
    parser.add_argument("--reranker-timeout", type=float, default=30.0)
    parser.add_argument("--vector-keep-top-k", type=int, default=0, help="Force-keep this many top vector hits before filling the rest with reranked chunks.")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--ragas-workers", type=int, default=1)
    parser.add_argument("--skip-context-recall", action="store_true")
    parser.add_argument("--skip-hit-at-10", action="store_true")
    parser.add_argument("--unsafe-batch-ragas", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true", help="Only build the RAGAS dataset; do not run metrics.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = get_openai_api_key(args.openai_api_key)
    api_base = resolve_api_base(args.openai_api_base, args.manifest)
    judge_api_base = resolve_judge_api_base(args.judge_api_base, args.judge_model, api_base)
    retrieval_embedding_metadata = get_retrieval_embedding_metadata(args.manifest, args.embedding_model, api_base)
    judge_api_key = resolve_judge_api_key(args.judge_api_key, args.judge_model, api_key)
    args.effective_reranker_url = args.reranker_url or DEFAULT_RERANKER_URL

    questions = load_rag_only_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    reference_map = load_reference_map(args.references)

    print(f"[INFO] API base: {api_base}", flush=True)
    print(f"[INFO] Judge API base: {judge_api_base}", flush=True)
    print(f"[INFO] Reranker URL: {args.effective_reranker_url}", flush=True)
    print(f"[INFO] RAG-only questions: {len(questions)}", flush=True)
    print(f"[INFO] Force rag_only route: {args.force_rag_only}", flush=True)

    rows = build_eval_rows(questions, reference_map, args, api_base, api_key)
    save_jsonl(rows, args.dataset_output)
    print(f"[OK] Saved RAGAS dataset -> {args.dataset_output}", flush=True)

    if args.skip_ragas:
        rows_df = pd.DataFrame(rows)
        payload = {
            "summary": {
                **summarize_metric_columns(rows_df, [], include_hit_at_10=not args.skip_hit_at_10),
                "ragas_skipped": True,
            },
            "rows": rows,
            "dataset_output": str(args.dataset_output),
            "top_k": args.top_k,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_mode": args.retrieval_mode,
            "reranker_url": args.effective_reranker_url,
            "reranker_top_k": args.reranker_top_k,
            "vector_keep_top_k": args.vector_keep_top_k,
            "llm_model": args.llm_model,
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
        metadata = pd.DataFrame(
            [
                {
                    "id": row.get("id"),
                    "retrieved_chunk_ids": row.get("retrieved_chunk_ids"),
                    "source_chunk_ids": row.get("source_chunk_ids"),
                    "gold_chunk_id": row.get("gold_chunk_id"),
                    "gold_rank": row.get("gold_rank"),
                    "hit_at_1": row.get("hit_at_1"),
                    "hit_at_5": row.get("hit_at_5"),
                    "hit_at_10": row.get("hit_at_10"),
                    "mrr": row.get("mrr"),
                }
                for row in rows
            ]
        )
        for col in reversed(metadata.columns):
            if col not in df.columns:
                df.insert(0, col, metadata[col])
        summary = summarize_metric_columns(df, metrics, include_hit_at_10=not args.skip_hit_at_10)
    else:
        print("[INFO] Running RAGAS in safe sequential mode.", flush=True)
        df = run_ragas_safe(rows, metrics, ragas_llm, ragas_embeddings, run_config)
        summary = summarize_metric_columns(df, metrics, include_hit_at_10=not args.skip_hit_at_10)

    write_summary_csv(args.summary, df, summary)
    payload = {
        "summary": summary,
        "rows": df.to_dict(orient="records"),
        "dataset_output": str(args.dataset_output),
        "top_k": args.top_k,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_mode": args.retrieval_mode,
        "reranker_url": args.effective_reranker_url,
        "reranker_top_k": args.reranker_top_k,
        "vector_keep_top_k": args.vector_keep_top_k,
        "llm_model": args.llm_model,
        "judge_model": args.judge_model,
        **retrieval_embedding_metadata,
        "api_base": api_base,
        "judge_api_base": judge_api_base,
        "ragas_embedding_provider": args.ragas_embedding_provider,
        "ragas_embedding_api_base": api_base if args.ragas_embedding_provider == "openai" else (args.ragas_embedding_api_url or os.environ.get("EMBEDDING_URL") or os.environ.get("BGE_EMBEDDING_URL")),
        "skip_context_recall": args.skip_context_recall,
        "skip_hit_at_10": args.skip_hit_at_10,
        "safe_sequential_mode": not args.unsafe_batch_ragas,
        "force_rag_only": args.force_rag_only,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RAGAS SUMMARY]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] Saved RAGAS row CSV -> {args.summary}", flush=True)
    print(f"[OK] Saved RAGAS JSON -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
