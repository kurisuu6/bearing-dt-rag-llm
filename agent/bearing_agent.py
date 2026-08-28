#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DT_DIR = PROJECT_ROOT / "DT"
SKF_RAG_DIR = PROJECT_ROOT / "SKF-RAG"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DT_DIR) not in sys.path:
    sys.path.insert(0, str(DT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_router import route_question
from dt_query_tools import (  # type: ignore
    DEFAULT_DB,
    compare_bearings,
    get_bearing_features,
    get_health_trend,
    get_latest_state,
    get_top_anomalies,
)
from dt_llm_planner import run_dt_with_llm_planner  # type: ignore

DEFAULT_INDEX_DIR = SKF_RAG_DIR / "chapters_01_11" / "llamaindex_vector_index_bge_m3"
DEFAULT_MANIFEST = SKF_RAG_DIR / "chapters_01_11" / "llamaindex_vector_index_bge_m3_manifest.json"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_ROUTER_MODEL = "gpt-4o-mini"
DEFAULT_RERANKER_URL = os.environ.get("RERANKER_URL")
RETRIEVAL_MODES = {"bm25", "dense", "hybrid", "hybrid_reranker"}
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def resolve_ollama_openai_base() -> Optional[str]:
    base = os.environ.get("OLLAMA_BASE_URL")
    if not base:
        return None
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_dt_action(question: str, route: dict[str, Any]) -> str:
    text = question.lower()
    entities = route.get("dt_entities") or {}
    metric = entities.get("metric")
    experiment = entities.get("experiment")
    bearing_id = entities.get("bearing_id")
    has_specific_bearing = bool(experiment and bearing_id)

    if has_specific_bearing and ("trend" in text or "over time" in text or "increase over time" in text or "changes over time" in text):
        return "trend"
    if "compare" in text or "which bearing" in text or "all four bearings" in text:
        return "compare"
    if not has_specific_bearing and ("highest" in text or "top" in text or "most anomal" in text or "failure_near" in text):
        return "top_anomalies"
    if has_specific_bearing and (
        "feature" in text
        or "features" in text
        or "abnormal" in text
        or "diagnose" in text
        or "explain" in text
        or "fault" in text
        or "failure_near" in text
        or "degradation" in text
    ):
        return "features"
    if "feature" in text or "features" in text or "abnormal" in text:
        return "features"
    if metric and metric not in {"health_state", "state"}:
        return "features"
    return "latest_state"


def require_dt_entity(route: dict[str, Any], key: str) -> str:
    value = (route.get("dt_entities") or {}).get(key)
    if not value:
        raise ValueError(f"The DT route did not identify required entity: {key}")
    return value


def run_dt(question: str, route: dict[str, Any], db_path: Path) -> dict[str, Any]:
    action = infer_dt_action(question, route)
    entities = route.get("dt_entities") or {}
    experiment = entities.get("experiment")
    bearing_id = entities.get("bearing_id")

    if action == "top_anomalies":
        return {"action": action, "result": get_top_anomalies(db_path, experiment=experiment, top_k=5)}

    if action == "compare":
        if not experiment:
            return {"action": "top_anomalies", "result": get_top_anomalies(db_path, top_k=8)}
        return {"action": action, "result": compare_bearings(db_path, experiment)}

    if action == "trend":
        experiment = require_dt_entity(route, "experiment")
        bearing_id = require_dt_entity(route, "bearing_id")
        return {"action": action, "result": get_health_trend(db_path, experiment, bearing_id)}

    if action == "features":
        experiment = require_dt_entity(route, "experiment")
        bearing_id = require_dt_entity(route, "bearing_id")
        return {"action": action, "result": get_bearing_features(db_path, experiment, bearing_id)}

    experiment = require_dt_entity(route, "experiment")
    bearing_id = require_dt_entity(route, "bearing_id")
    return {"action": "latest_state", "result": get_latest_state(db_path, experiment, bearing_id)}


def configure_embedding_from_manifest(manifest_path: Path, embedding_model: str, api_base: Optional[str], api_key: Optional[str]):
    try:
        from llama_index.core import Settings
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LlamaIndex core is not installed. Install llama-index-core."
        ) from exc

    manifest = load_json(manifest_path)
    manifest_embedding = manifest.get("embedding", {}) if manifest else {}
    provider = manifest_embedding.get("provider", "openai")
    if provider == "http" and embedding_model == DEFAULT_EMBED_MODEL:
        model = manifest_embedding.get("model") or embedding_model
    else:
        model = embedding_model or manifest_embedding.get("model") or DEFAULT_EMBED_MODEL

    if provider == "http":
        from http_embedding import make_http_embedding

        embedding_url = manifest_embedding.get("api_url") or os.environ.get("EMBEDDING_URL") or os.environ.get("BGE_EMBEDDING_URL")
        Settings.embed_model = make_http_embedding(
            api_url=embedding_url,
            model=model,
            embed_batch_size=int(manifest_embedding.get("embed_batch_size") or 16),
            timeout=float(manifest_embedding.get("timeout") or 60.0),
            api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("BGE_EMBEDDING_API_KEY"),
        )
        return {"provider": "http", "model": model, "api_url": embedding_url}

    base = api_base or manifest_embedding.get("api_base") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for SKF RAG retrieval because query embedding must be computed.")

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LlamaIndex OpenAI embedding package is not installed. Install llama-index-embeddings-openai."
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": key,
        "embed_batch_size": 4,
        "timeout": 30.0,
        "max_retries": 1,
    }
    if base:
        kwargs["api_base"] = base
    Settings.embed_model = OpenAIEmbedding(**kwargs)
    return {"model": model, "api_base": base}


def resolve_source_chunks_path(manifest_path: Path) -> Path:
    manifest = load_json(manifest_path)
    source_chunks = manifest.get("source_chunks")
    if not source_chunks:
        raise RuntimeError(f"Manifest does not contain source_chunks: {manifest_path}")
    path = Path(source_chunks)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Source chunks JSON not found: {path}")
    return path


def chunk_to_retrieval_chunk(chunk: dict[str, Any], rank: int, score: float, retrieval_source: str) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    return {
        "rank": rank,
        "score": score,
        "chunk_id": chunk.get("id") or metadata.get("chunk_id"),
        "section": metadata.get("section") or chunk.get("section"),
        "page_start": metadata.get("page_start") or chunk.get("page_start"),
        "page_end": metadata.get("page_end") or chunk.get("page_end"),
        "text": chunk.get("embedding_text") or chunk.get("text") or "",
        "retrieval_source": retrieval_source,
    }


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def load_bm25_corpus(manifest_path: Path) -> list[dict[str, Any]]:
    source_path = resolve_source_chunks_path(manifest_path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    corpus = []
    for chunk in chunks:
        text = chunk.get("embedding_text") or chunk.get("text") or ""
        tokens = tokenize_for_bm25(text)
        if not tokens:
            continue
        corpus.append({"chunk": chunk, "tokens": tokens, "term_freq": Counter(tokens), "length": len(tokens)})
    return corpus


def bm25_retrieve(query: str, manifest_path: Path, top_k: int) -> list[dict[str, Any]]:
    corpus = load_bm25_corpus(manifest_path)
    if not corpus:
        return []
    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return []

    doc_count = len(corpus)
    avg_len = sum(item["length"] for item in corpus) / doc_count
    doc_freq: Counter[str] = Counter()
    for item in corpus:
        for term in set(item["tokens"]):
            doc_freq[term] += 1

    scored = []
    for item in corpus:
        score = 0.0
        length = item["length"]
        term_freq = item["term_freq"]
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg_len)
            score += idf * (tf * (BM25_K1 + 1)) / denom
        if score > 0:
            scored.append((score, item["chunk"]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk_to_retrieval_chunk(chunk, rank, score, "bm25") for rank, (score, chunk) in enumerate(scored[:top_k], start=1)]


def dense_retrieve(
    query: str,
    index_dir: Path,
    manifest_path: Path,
    top_k: int,
    embedding_model: str,
    api_base: Optional[str],
    api_key: Optional[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not index_dir.exists():
        raise FileNotFoundError(f"SKF RAG index directory not found: {index_dir}")
    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except ModuleNotFoundError as exc:
        raise RuntimeError("LlamaIndex core is not installed.") from exc

    embedding = configure_embedding_from_manifest(manifest_path, embedding_model, api_base, api_key)
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved = retriever.retrieve(query)
    chunks = []
    for rank, item in enumerate(retrieved, start=1):
        node = item.node
        metadata = node.metadata or {}
        chunks.append(
            {
                "rank": rank,
                "score": item.score,
                "chunk_id": metadata.get("chunk_id"),
                "section": metadata.get("section"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "text": node.get_content(metadata_mode="none"),
                "retrieval_source": "dense",
            }
        )
    return chunks, embedding


def rrf_merge(dense_chunks: list[dict[str, Any]], bm25_chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source, chunks in [("dense", dense_chunks), ("bm25", bm25_chunks)]:
        for rank, chunk in enumerate(chunks, start=1):
            key = chunk.get("chunk_id") or str(chunk.get("text") or "")[:160]
            if key not in merged:
                merged[key] = dict(chunk)
                merged[key]["score"] = 0.0
                merged[key]["retrieval_sources"] = []
                merged[key]["source_ranks"] = {}
                merged[key]["source_scores"] = {}
            merged[key]["score"] += 1.0 / (RRF_K + rank)
            merged[key]["retrieval_sources"].append(source)
            merged[key]["source_ranks"][source] = rank
            merged[key]["source_scores"][source] = chunk.get("score")
    ranked = sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    for rank, chunk in enumerate(ranked[:top_k], start=1):
        chunk["rank"] = rank
        chunk["retrieval_source"] = "hybrid"
    return ranked[:top_k]


def prepend_pre_rerank_kept_chunks(
    chunks: list[dict[str, Any]],
    dense_chunks: list[dict[str, Any]],
    bm25_chunks: list[dict[str, Any]],
    dense_keep_top_k: int = 0,
    bm25_keep_top_k: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def chunk_key(chunk: dict[str, Any]) -> str:
        return str(chunk.get("chunk_id") or chunk.get("text", "")[:160])

    def add(chunk: dict[str, Any], reason: str) -> None:
        key = chunk_key(chunk)
        if key in selected_keys:
            return
        item = dict(chunk)
        item["pre_rerank_keep_reason"] = reason
        selected.append(item)
        selected_keys.add(key)

    for chunk in dense_chunks[: max(0, dense_keep_top_k)]:
        add(chunk, "dense_pre_rerank_keep")
    for chunk in bm25_chunks[: max(0, bm25_keep_top_k)]:
        add(chunk, "bm25_pre_rerank_keep")
    for chunk in chunks:
        add(chunk, str(chunk.get("pre_rerank_keep_reason") or "rrf_candidate"))

    for rank, chunk in enumerate(selected, start=1):
        chunk["rank"] = rank
    return selected


def run_rag(
    query: str,
    index_dir: Path,
    manifest_path: Path,
    top_k: int,
    embedding_model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    retrieval_top_k: Optional[int] = None,
    reranker_url: Optional[str] = None,
    reranker_top_k: Optional[int] = None,
    reranker_timeout: float = 30.0,
    vector_keep_top_k: int = 0,
    retrieval_mode: str = "dense",
    reranker_query: Optional[str] = None,
    dense_keep_top_k: int = 0,
    bm25_keep_top_k: int = 0,
) -> dict[str, Any]:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}. Expected one of {sorted(RETRIEVAL_MODES)}")
    initial_top_k = max(top_k, retrieval_top_k or top_k)

    embedding: Optional[dict[str, Any]] = None
    dense_chunks: list[dict[str, Any]] = []
    bm25_chunks: list[dict[str, Any]] = []
    if retrieval_mode in {"dense", "hybrid", "hybrid_reranker"}:
        dense_chunks, embedding = dense_retrieve(query, index_dir, manifest_path, initial_top_k, embedding_model, api_base, api_key)
    if retrieval_mode in {"bm25", "hybrid", "hybrid_reranker"}:
        bm25_chunks = bm25_retrieve(query, manifest_path, initial_top_k)

    if retrieval_mode == "dense":
        chunks = dense_chunks[:top_k]
    elif retrieval_mode == "bm25":
        chunks = bm25_chunks[:top_k]
    else:
        chunks = rrf_merge(dense_chunks, bm25_chunks, initial_top_k)

    reranker = None
    pre_rerank_candidate_count = None
    if retrieval_mode == "hybrid_reranker" and reranker_url:
        chunks = prepend_pre_rerank_kept_chunks(
            chunks,
            dense_chunks=dense_chunks,
            bm25_chunks=bm25_chunks,
            dense_keep_top_k=dense_keep_top_k,
            bm25_keep_top_k=bm25_keep_top_k,
        )
        pre_rerank_candidate_count = len(chunks)
        chunks, reranker = rerank_chunks(
            query=reranker_query or query,
            chunks=chunks,
            reranker_url=reranker_url,
            top_k=reranker_top_k or top_k,
            timeout=reranker_timeout,
            vector_keep_top_k=vector_keep_top_k,
        )
    else:
        chunks = chunks[:top_k]
    return {
        "query": query,
        "reranker_query": reranker_query,
        "top_k": top_k,
        "initial_top_k": initial_top_k,
        "retrieval_mode": retrieval_mode,
        "embedding": embedding,
        "bm25": {"enabled": retrieval_mode in {"bm25", "hybrid", "hybrid_reranker"}, "candidate_count": len(bm25_chunks)},
        "dense": {"enabled": retrieval_mode in {"dense", "hybrid", "hybrid_reranker"}, "candidate_count": len(dense_chunks)},
        "pre_rerank_keep": {
            "dense_keep_top_k": dense_keep_top_k if retrieval_mode == "hybrid_reranker" else 0,
            "bm25_keep_top_k": bm25_keep_top_k if retrieval_mode == "hybrid_reranker" else 0,
            "candidate_count_after_keep": pre_rerank_candidate_count,
        },
        "reranker": reranker,
        "chunks": chunks,
    }


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker_url: str,
    top_k: int,
    timeout: float = 30.0,
    vector_keep_top_k: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        return [], {"enabled": True, "url": reranker_url, "top_k": top_k, "status": "skipped_empty"}
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("The requests package is required for reranker support. Install it with: pip install requests") from exc

    keep_count = min(max(0, vector_keep_top_k), top_k, len(chunks))
    kept_chunks = []
    kept_keys = set()
    for chunk in chunks[:keep_count]:
        kept = dict(chunk)
        kept["vector_rank"] = kept.get("rank")
        kept["vector_score"] = kept.get("score")
        kept["selection_reason"] = "vector_forced_keep"
        kept_chunks.append(kept)
        kept_keys.add(kept.get("chunk_id") or str(kept.get("text") or "")[:120])

    remaining_chunks = [
        chunk
        for chunk in chunks
        if (chunk.get("chunk_id") or str(chunk.get("text") or "")[:120]) not in kept_keys
    ]
    remaining_slots = max(0, top_k - len(kept_chunks))
    if remaining_slots == 0 or not remaining_chunks:
        for rank, chunk in enumerate(kept_chunks, start=1):
            chunk["rank"] = rank
        return kept_chunks[:top_k], {
            "enabled": True,
            "url": reranker_url,
            "top_k": top_k,
            "vector_keep_top_k": keep_count,
            "status": "skipped_no_remaining_slots",
            "candidate_count": len(chunks),
            "returned_count": len(kept_chunks[:top_k]),
        }

    documents = [str(chunk.get("text") or "") for chunk in remaining_chunks]
    payload = {"query": query, "documents": documents, "top_k": min(remaining_slots, len(documents))}
    start_time = time.time()
    response = requests.post(f"{reranker_url.rstrip('/')}/rerank", json=payload, timeout=timeout)
    elapsed = time.time() - start_time
    response.raise_for_status()
    data = response.json()
    ranked_documents = data.get("ranked_documents") or []
    scores = data.get("scores") or []

    buckets: dict[str, list[dict[str, Any]]] = {}
    for chunk in remaining_chunks:
        buckets.setdefault(str(chunk.get("text") or ""), []).append(chunk)

    ranked_chunks: list[dict[str, Any]] = list(kept_chunks)
    used_keys: set[str] = set(kept_keys)
    for document, score in zip(ranked_documents, scores):
        candidates = buckets.get(str(document)) or []
        if not candidates:
            continue
        chunk = dict(candidates.pop(0))
        chunk["vector_rank"] = chunk.get("rank")
        chunk["vector_score"] = chunk.get("score")
        chunk["rank"] = len(ranked_chunks) + 1
        chunk["rerank_score"] = score
        chunk["score"] = score
        chunk["selection_reason"] = "bge_reranker"
        ranked_chunks.append(chunk)
        used_keys.add(chunk.get("chunk_id") or str(document)[:120])

    if len(ranked_chunks) < min(top_k, len(chunks)):
        for chunk in chunks:
            dedupe_key = chunk.get("chunk_id") or str(chunk.get("text") or "")[:120]
            if dedupe_key in used_keys:
                continue
            fallback = dict(chunk)
            fallback["vector_rank"] = fallback.get("rank")
            fallback["vector_score"] = fallback.get("score")
            fallback["rank"] = len(ranked_chunks) + 1
            fallback["selection_reason"] = "vector_fallback_after_rerank"
            ranked_chunks.append(fallback)
            if len(ranked_chunks) >= top_k:
                break

    reranker = {
        "enabled": True,
        "url": reranker_url,
        "model_name": data.get("model_name"),
        "query": query,
        "top_k": top_k,
        "vector_keep_top_k": keep_count,
        "processing_time": data.get("processing_time"),
        "elapsed_seconds": elapsed,
        "candidate_count": len(chunks),
        "reranked_candidate_count": len(remaining_chunks),
        "returned_count": len(ranked_chunks[:top_k]),
    }
    return ranked_chunks[:top_k], reranker


def node_to_chunk(item: Any, rank: int) -> dict[str, Any]:
    node = item.node
    metadata = node.metadata or {}
    return {
        "rank": rank,
        "score": item.score,
        "chunk_id": metadata.get("chunk_id"),
        "section": metadata.get("section"),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "text": node.get_content(metadata_mode="none"),
    }


def run_rag_queries(
    queries: list[dict[str, str]],
    index_dir: Path,
    manifest_path: Path,
    top_k: int,
    embedding_model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    per_query_top_k: int = 3,
    min_chunks_per_query: int = 1,
) -> dict[str, Any]:
    if not queries:
        return {
            "query": "",
            "queries": [],
            "top_k": top_k,
            "per_query_top_k": per_query_top_k,
            "min_chunks_per_query": min_chunks_per_query,
            "chunks": [],
        }
    if not index_dir.exists():
        raise FileNotFoundError(f"SKF RAG index directory not found: {index_dir}")
    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except ModuleNotFoundError as exc:
        raise RuntimeError("LlamaIndex core is not installed.") from exc

    embedding = configure_embedding_from_manifest(manifest_path, embedding_model, api_base, api_key)
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=per_query_top_k)

    query_results = []
    merged: dict[str, dict[str, Any]] = {}
    query_candidates: dict[int, list[dict[str, Any]]] = {}
    for query_index, query_item in enumerate(queries, start=1):
        query_text = query_item["query"]
        retrieved = retriever.retrieve(query_text)
        chunks = []
        for rank, item in enumerate(retrieved, start=1):
            chunk = node_to_chunk(item, rank)
            chunk["query_index"] = query_index
            chunks.append(chunk)

            dedupe_key = chunk.get("chunk_id") or chunk.get("text", "")[:160]
            chunk["dedupe_key"] = dedupe_key
            if dedupe_key not in merged or (chunk.get("score") or 0) > (merged[dedupe_key].get("score") or 0):
                merged[dedupe_key] = dict(chunk)
                merged[dedupe_key]["matched_queries"] = [query_index]
                merged[dedupe_key]["matched_purposes"] = [query_item.get("purpose", "")]
            else:
                matched_queries = merged[dedupe_key].setdefault("matched_queries", [])
                if query_index not in matched_queries:
                    matched_queries.append(query_index)
                append_unique(merged[dedupe_key].setdefault("matched_purposes", []), [query_item.get("purpose", "")])

        query_candidates[query_index] = chunks
        query_results.append(
            {
                "query_index": query_index,
                "query": query_text,
                "purpose": query_item.get("purpose", ""),
                "chunks": chunks,
            }
        )

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    quota = max(0, min_chunks_per_query)
    if quota and top_k > 0:
        for query_index in range(1, len(queries) + 1):
            if len(selected) >= top_k:
                break
            kept_for_query = 0
            for chunk in query_candidates.get(query_index, []):
                dedupe_key = chunk.get("dedupe_key") or chunk.get("chunk_id") or chunk.get("text", "")[:160]
                if dedupe_key in selected_keys:
                    continue
                selected_chunk = dict(merged.get(dedupe_key, chunk))
                selected_chunk["selection_reason"] = f"quota_for_query_{query_index}"
                selected_chunk["selection_query_index"] = query_index
                selected.append(selected_chunk)
                selected_keys.add(dedupe_key)
                kept_for_query += 1
                if kept_for_query >= quota or len(selected) >= top_k:
                    break

    remaining = sorted(merged.values(), key=lambda chunk: chunk.get("score") or 0, reverse=True)
    for chunk in remaining:
        if len(selected) >= top_k:
            break
        dedupe_key = chunk.get("dedupe_key") or chunk.get("chunk_id") or chunk.get("text", "")[:160]
        if dedupe_key in selected_keys:
            continue
        selected_chunk = dict(chunk)
        selected_chunk["selection_reason"] = "score_fill"
        selected_chunk["selection_query_index"] = None
        selected.append(selected_chunk)
        selected_keys.add(dedupe_key)

    final_chunks = selected[:top_k]
    for rank, chunk in enumerate(final_chunks, start=1):
        chunk["rank"] = rank
        chunk.pop("dedupe_key", None)
    for query_result in query_results:
        for chunk in query_result["chunks"]:
            chunk.pop("dedupe_key", None)

    return {
        "query": " || ".join(item["query"] for item in queries),
        "queries": queries,
        "top_k": top_k,
        "per_query_top_k": per_query_top_k,
        "min_chunks_per_query": min_chunks_per_query,
        "embedding": embedding,
        "chunks": final_chunks,
        "query_results": query_results,
    }


FEATURE_TO_RAG_TERMS = {
    "rms": [
        "high RMS",
        "increased vibration amplitude",
        "abnormal vibration",
        "bearing damage",
        "inspection",
    ],
    "kurtosis": [
        "high kurtosis",
        "shock impulses",
        "impulsive vibration",
        "localized bearing damage",
        "raceway damage",
        "rolling element damage",
    ],
    "crest_factor": [
        "high crest factor",
        "peak vibration",
        "impact damage",
        "shock load",
        "incipient bearing fault",
    ],
    "impulse_factor": [
        "high impulse factor",
        "shock impulses",
        "impact vibration",
        "localized damage",
    ],
    "clearance_factor": [
        "high clearance factor",
        "bearing looseness",
        "clearance problem",
        "mounting condition",
    ],
    "high_band_energy_ratio": [
        "high-frequency vibration",
        "high frequency energy",
        "surface distress",
        "rolling contact fatigue",
        "lubrication film breakdown",
    ],
    "spectral_entropy": [
        "irregular vibration spectrum",
        "complex vibration spectrum",
        "fault progression",
        "bearing degradation",
    ],
}

FEATURE_EVIDENCE_PATTERNS = {
    "rms": ["rms z-score", "rms"],
    "kurtosis": ["kurtosis z-score", "kurtosis"],
    "crest_factor": ["crest factor z-score", "crest factor"],
    "impulse_factor": ["impulse factor z-score", "impulse factor"],
    "clearance_factor": ["clearance factor z-score", "clearance factor"],
    "high_band_energy_ratio": ["high-frequency energy ratio z-score", "high band energy ratio", "high-frequency energy"],
    "spectral_entropy": ["spectral entropy z-score", "spectral entropy"],
}

FEATURE_EVIDENCE_NAMES = {
    "rms": ["rms"],
    "kurtosis": ["kurtosis"],
    "crest_factor": ["crest factor"],
    "impulse_factor": ["impulse factor"],
    "clearance_factor": ["clearance factor"],
    "high_band_energy_ratio": ["high-frequency energy ratio", "high band energy ratio"],
    "spectral_entropy": ["spectral entropy"],
}

FEATURE_NEUTRAL_LABELS = {
    "rms": "RMS vibration level",
    "kurtosis": "kurtosis / impulsiveness indicator",
    "crest_factor": "crest factor / peak vibration indicator",
    "impulse_factor": "impulse factor / shock impulse indicator",
    "clearance_factor": "clearance factor / looseness-related indicator",
    "high_band_energy_ratio": "high-frequency energy ratio",
    "spectral_entropy": "spectral entropy / spectrum complexity indicator",
}

FEATURE_MILD_LABELS = {
    "rms": "mildly elevated RMS / vibration amplitude",
    "kurtosis": "mildly elevated kurtosis / impulsive vibration tendency",
    "crest_factor": "mildly elevated crest factor / peak vibration tendency",
    "impulse_factor": "mildly elevated impulse factor / shock impulse tendency",
    "clearance_factor": "mildly elevated clearance factor / possible looseness indicator",
    "high_band_energy_ratio": "mildly elevated high-frequency vibration energy",
    "spectral_entropy": "mildly elevated spectral entropy / spectrum complexity",
}


def append_unique(terms: list[str], values: list[str] | tuple[str, ...]) -> None:
    seen = {term.lower() for term in terms}
    for value in values:
        value = str(value).strip()
        if value and value.lower() not in seen:
            terms.append(value)
            seen.add(value.lower())


def get_dt_payload(dt_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not dt_result:
        return {}
    result = dt_result.get("result", {})
    return result if isinstance(result, dict) else {}


def get_dt_evidence_text(payload: dict[str, Any]) -> str:
    parts = []
    for key in ("evidence", "latest_evidence"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def get_dt_feature_values(payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    nested = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    for feature in FEATURE_TO_RAG_TERMS:
        if feature in nested:
            values[feature] = nested.get(feature)
        elif feature in payload:
            values[feature] = payload.get(feature)
    return values


def parse_feature_z_scores(evidence_text: str) -> dict[str, float]:
    z_scores: dict[str, float] = {}
    text = str(evidence_text or "").lower()
    for feature, names in FEATURE_EVIDENCE_NAMES.items():
        values: list[float] = []
        for name in names:
            pattern = rf"{re.escape(name)}\s+z-score\s+(-?\d+(?:\.\d+)?)"
            values.extend(float(match) for match in re.findall(pattern, text))
        if values:
            z_scores[feature] = max(values, key=abs)
    return z_scores


def feature_requested_by_question(feature: str, question: str) -> bool:
    question_text = question.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "rms": ["rms", "vibration amplitude", "vibration level"],
        "kurtosis": ["kurtosis"],
        "crest_factor": ["crest factor"],
        "impulse_factor": ["impulse factor"],
        "clearance_factor": ["clearance factor", "clearance", "looseness"],
        "high_band_energy_ratio": ["high frequency", "high-frequency", "high band", "high-frequency energy"],
        "spectral_entropy": ["spectral entropy", "entropy", "complex spectrum", "vibration spectrum"],
    }
    return any(alias in question_text for alias in aliases.get(feature, [feature.replace("_", " ")]))


def feature_level_from_z_score(z_score: Optional[float], health_state: Optional[str]) -> str:
    if z_score is None:
        return "unconfirmed"
    abs_z = abs(z_score)
    if abs_z >= 3.0:
        return "high"
    if abs_z >= 2.0:
        return "mild"
    # Normal-state records should not be called abnormal just because a value exists.
    if str(health_state or "").lower() == "normal":
        return "normal"
    return "normal"


def assess_dt_features(payload: dict[str, Any], question: str) -> dict[str, dict[str, Any]]:
    evidence_text = get_dt_evidence_text(payload)
    z_scores = parse_feature_z_scores(evidence_text)
    features = get_dt_feature_values(payload)
    health_state = payload.get("health_state") or payload.get("latest_state")
    broad_question = any(word in question.lower() for word in ["feature", "features", "abnormal", "diagnose"])
    assessments: dict[str, dict[str, Any]] = {}

    for feature in FEATURE_TO_RAG_TERMS:
        has_value = feature in features
        has_z_score = feature in z_scores
        requested = feature_requested_by_question(feature, question)
        if not (has_value or has_z_score or requested or broad_question):
            continue
        # Broad questions should focus on statistically supported features.
        if broad_question and not (has_z_score or requested):
            continue
        z_score = z_scores.get(feature)
        level = feature_level_from_z_score(z_score, health_state)
        if level == "unconfirmed" and requested and str(health_state or "").lower() != "normal":
            level = "reported"
        elif level == "unconfirmed" and not requested:
            continue
        assessments[feature] = {
            "feature": feature,
            "value": features.get(feature),
            "z_score": z_score,
            "level": level,
            "mentioned_in_question": requested,
            "has_statistical_support": has_z_score,
        }
    return assessments


def detect_abnormal_features(payload: dict[str, Any], question: str) -> list[str]:
    assessments = assess_dt_features(payload, question)
    detected = [
        feature
        for feature, assessment in assessments.items()
        if assessment.get("level") in {"high", "mild"}
    ]
    return list(dict.fromkeys(detected))


def feature_label_from_assessment(feature: str, assessment: dict[str, Any]) -> str:
    level = assessment.get("level")
    if level == "high":
        return FEATURE_LABELS.get(feature, feature.replace("_", " "))
    if level == "mild":
        return FEATURE_MILD_LABELS.get(feature, FEATURE_NEUTRAL_LABELS.get(feature, feature.replace("_", " ")))
    return FEATURE_NEUTRAL_LABELS.get(feature, feature.replace("_", " "))


def feature_interpretation_from_assessment(feature: str, assessment: dict[str, Any]) -> Optional[str]:
    level = assessment.get("level")
    if level == "high":
        return FEATURE_INTERPRETATIONS.get(feature)
    if level == "mild":
        mild_interpretations = {
            "rms": "vibration level may be increasing",
            "kurtosis": "impulsive vibration may be emerging",
            "crest_factor": "peak vibration may be emerging",
            "impulse_factor": "shock impulses may be emerging",
            "clearance_factor": "looseness, clearance, or mounting-related effects should be checked cautiously",
            "high_band_energy_ratio": "high-frequency vibration components may be present",
            "spectral_entropy": "the vibration spectrum may be becoming more complex",
        }
        return mild_interpretations.get(feature)
    return None


FEATURE_LABELS = {
    "rms": "high RMS / increased vibration amplitude",
    "kurtosis": "high kurtosis / impulsive vibration",
    "crest_factor": "high crest factor / peak impact vibration",
    "impulse_factor": "high impulse factor / shock impulses",
    "clearance_factor": "high clearance factor / possible looseness or clearance issue",
    "high_band_energy_ratio": "high-frequency vibration energy",
    "spectral_entropy": "irregular or complex vibration spectrum",
}

FEATURE_INTERPRETATIONS = {
    "rms": "overall vibration energy is increasing",
    "kurtosis": "impulsive impacts are present",
    "crest_factor": "peak vibration shocks are prominent",
    "impulse_factor": "repeated impact events may be present",
    "clearance_factor": "looseness, clearance, or mounting-related effects may contribute",
    "high_band_energy_ratio": "localized contact damage, surface distress, or early bearing defects may be present",
    "spectral_entropy": "the vibration spectrum is complex, which can indicate fault progression",
}


def infer_dt_rag_query_focus(question: str, dt_context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    text = question.lower()
    health_stage = str((dt_context or {}).get("health_stage") or "").lower()
    abnormal_keys = set((dt_context or {}).get("abnormal_feature_keys") or [])
    focus: list[str] = ["vibration monitoring for rolling bearings", "condition monitoring", "inspection during operation"]
    cautions: list[str] = []

    is_maintenance = any(term in text for term in ["maintenance", "action", "inspect", "inspection", "replace", "replacement", "continue operating", "stop", "troubleshooting", "checked first", "corrective"])
    is_fault_type = any(term in text for term in ["fault type", "damage categories", "damage modes", "classify", "classification", "mechanism", "raceway", "rolling element", "localized damage", "imbalance", "lubrication-related", "mounting-related"])
    is_explanation = any(term in text for term in ["why", "explain", "diagnose", "meaning", "suggest", "reflect"])

    if is_fault_type:
        append_unique(focus, ["bearing damage classification", "bearing damage mechanisms", "failure mode classification"])
    if is_explanation:
        append_unique(focus, ["increase in vibration levels", "vibration characteristics", "rolling bearing defect frequency analysis"])
    if is_maintenance:
        append_unique(focus, ["where to take vibration measurements", "when to take measurements", "trouble conditions and their solutions", "corrective action after inspection"])
    if any(term in text for term in ["lubrication", "lubricant", "relubrication"]):
        append_unique(focus, ["monitoring lubrication conditions", "ineffective lubrication"])
    if any(term in text for term in ["mounting", "misalignment", "looseness", "clearance"]) or "clearance_factor" in abnormal_keys:
        append_unique(focus, ["mounting condition", "misalignment", "bearing looseness"])
    if any(term in text for term in ["high-frequency", "high frequency", "high-frequency energy"]) or "high_band_energy_ratio" in abnormal_keys:
        append_unique(focus, ["high-frequency vibration generated by rolling bearing defects"])
    if {"kurtosis", "crest_factor", "impulse_factor"} & abnormal_keys:
        append_unique(focus, ["shock impulses", "peak vibration", "defect-generated vibration peaks"])
    if "rms" in abnormal_keys:
        append_unique(focus, ["increased vibration levels"])
    if health_stage == "normal":
        cautions.append("DT health_state is normal; retrieve SKF evidence for cautious comparison, not for assuming confirmed damage.")
        if not abnormal_keys:
            focus = [
                term
                for term in focus
                if term
                not in {
                    "bearing damage",
                    "bearing damage classification",
                    "bearing damage mechanisms",
                    "failure mode classification",
                    "inspection priority",
                    "maintenance implications",
                    "developing degradation",
                    "early indications of bearing damage",
                }
            ]
    elif health_stage in {"failure_near", "severe_degradation"}:
        append_unique(focus, ["bearing damage", "inspection priority", "maintenance implications"])
    elif health_stage in {"degradation", "early_degradation"}:
        append_unique(focus, ["developing degradation", "early indications of bearing damage"])

    return {
        "intent": "maintenance_decision" if is_maintenance else "fault_type_identification" if is_fault_type else "fault_explanation",
        "focus_terms": focus,
        "cautions": cautions,
    }


def build_dt_context_card(question: str, dt_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload = get_dt_payload(dt_result)
    feature_assessments = assess_dt_features(payload, question)
    detected_features = [
        feature
        for feature, assessment in feature_assessments.items()
        if assessment.get("level") in {"high", "mild"}
    ]
    mentioned_features = [
        feature
        for feature, assessment in feature_assessments.items()
        if assessment.get("mentioned_in_question")
    ]
    features = get_dt_feature_values(payload)
    health_state = payload.get("health_state") or payload.get("latest_state") or payload.get("latest_health_state")

    severity = "unknown"
    if str(health_state) in {"failure_near", "severe_degradation"}:
        severity = "high"
    elif str(health_state) in {"degradation", "early_degradation"}:
        severity = "medium"
    elif health_state:
        severity = "low"

    abnormal_feature_labels = [
        feature_label_from_assessment(feature, feature_assessments[feature])
        for feature in detected_features
    ]
    mentioned_feature_labels = [
        feature_label_from_assessment(feature, feature_assessments[feature])
        for feature in mentioned_features
        if feature not in detected_features
    ]
    engineering_interpretation = [
        interpretation
        for feature in detected_features
        for interpretation in [feature_interpretation_from_assessment(feature, feature_assessments[feature])]
        if interpretation
    ]
    context: dict[str, Any] = {
        "dataset": payload.get("experiment"),
        "bearing_id": payload.get("bearing_id"),
        "timestamp": payload.get("timestamp") or payload.get("latest_timestamp"),
        "sequence_index": payload.get("sequence_index") or payload.get("latest_sequence_index"),
        "health_stage": health_state,
        "severity": severity,
        "abnormal_features": abnormal_feature_labels,
        "abnormal_feature_keys": detected_features,
        "mentioned_features": mentioned_feature_labels,
        "mentioned_feature_keys": mentioned_features,
        "feature_values": {feature: features.get(feature) for feature in sorted(set(detected_features + mentioned_features)) if feature in features},
        "feature_assessments": feature_assessments,
        "evidence": payload.get("evidence") or payload.get("latest_evidence"),
        "engineering_interpretation": engineering_interpretation,
    }
    if payload.get("trend") or payload.get("delta") is not None:
        metric = payload.get("metric")
        trend = payload.get("trend")
        delta = payload.get("delta")
        early_mean = payload.get("early_mean")
        late_mean = payload.get("late_mean")
        context["trend_evidence"] = {
            "metric": metric,
            "trend": trend,
            "early_mean": early_mean,
            "late_mean": late_mean,
            "delta": delta,
            "percent_change": payload.get("percent_change"),
            "samples": payload.get("samples"),
            "window": payload.get("window"),
        }
        trend_parts = []
        if metric:
            trend_parts.append(str(metric).replace("_", " "))
        if trend:
            trend_parts.append(str(trend).replace("_", " "))
        if delta is not None:
            trend_parts.append(f"delta {delta}")
        if early_mean is not None and late_mean is not None:
            trend_parts.append(f"early mean {early_mean} to late mean {late_mean}")
        if trend_parts:
            context["engineering_interpretation"] = list(context["engineering_interpretation"])
            context["engineering_interpretation"].append(
                "DT trend evidence shows " + ", ".join(trend_parts)
            )
    context["query_focus"] = infer_dt_rag_query_focus(question, context)
    return context


def build_single_dt_augmented_query(question: str, dt_context: dict[str, Any]) -> str:
    dataset = dt_context.get("dataset") or "the IMS experiment"
    bearing_id = dt_context.get("bearing_id") or "the bearing"
    health_stage = dt_context.get("health_stage") or "unknown state"
    severity = dt_context.get("severity") or "unknown severity"
    abnormal_features = dt_context.get("abnormal_features") or []
    mentioned_features = dt_context.get("mentioned_features") or []
    interpretations = dt_context.get("engineering_interpretation") or []
    query_focus = dt_context.get("query_focus") or {}
    focus_terms = query_focus.get("focus_terms") or ["vibration monitoring for rolling bearings", "bearing inspection"]
    cautions = query_focus.get("cautions") or []

    health_text = f"{bearing_id} in {dataset} is classified by the DT as {health_stage} with {severity} severity."
    if abnormal_features:
        feature_text = "DT-supported elevated features: " + ", ".join(abnormal_features) + "."
    elif dt_context.get("trend_evidence"):
        trend = dt_context["trend_evidence"]
        metric = str(trend.get("metric") or "health indicator").replace("_", " ")
        trend_name = str(trend.get("trend") or "trend").replace("_", " ")
        delta = trend.get("delta")
        feature_text = f"DT trend evidence: {metric} shows {trend_name}"
        if delta is not None:
            feature_text += f" with delta {delta}"
        feature_text += "."
    elif mentioned_features:
        feature_text = (
            "The question mentions these DT features, but the current DT evidence does not mark them as statistically abnormal: "
            + ", ".join(mentioned_features)
            + "."
        )
    else:
        feature_text = "No statistically elevated DT vibration feature is available in the current DT context."

    interpretation_text = ""
    if interpretations:
        interpretation_text = "Cautious DT interpretation: " + ", ".join(interpretations) + "."
    elif str(health_stage).lower() == "normal":
        interpretation_text = "Do not assume bearing damage; use SKF evidence for comparison and inspection criteria."
    else:
        interpretation_text = "Use SKF evidence to interpret whether the DT state and features indicate bearing degradation."

    focus_text = "SKF retrieval focus: " + "; ".join(focus_terms) + "."
    caution_text = (" Query caution: " + " ".join(cautions)) if cautions else ""

    query = (
        f"{health_text} "
        f"{feature_text} "
        f"{interpretation_text} "
        f"{focus_text}{caution_text} "
        f"Original question: {question}"
    )
    return " ".join(query.split())


def build_dt_rag_retrieval_plan(question: str, dt_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    dt_context = build_dt_context_card(question, dt_result)
    rag_query = build_single_dt_augmented_query(question, dt_context)

    intent = ((dt_context.get("query_focus") or {}).get("intent") or "fault_explanation")
    if intent == "maintenance_decision":
        final_answer_task = (
            "Use DT state/features and SKF inspection/troubleshooting evidence to recommend cautious inspection or maintenance priorities. "
            "Do not claim a specific corrective action unless supported by the supplied evidence."
        )
    elif intent == "fault_type_identification":
        final_answer_task = (
            "Use DT state/features and SKF damage evidence to identify plausible bearing damage mechanisms. "
            "Distinguish likely mechanisms from unconfirmed root causes."
        )
    else:
        final_answer_task = (
            "Explain the mechanical meaning of the DT state/features using SKF vibration monitoring and bearing damage evidence. "
            "Keep interpretations cautious and evidence-grounded."
        )

    return {
        "question_type": "dt_rag",
        "retrieval_strategy": "single_dt_augmented_query",
        "dt_context": dt_context,
        "rag_query": rag_query,
        "rag_queries": [
            {
                "query": rag_query,
                "purpose": "Use one DT-augmented diagnostic query to retrieve SKF evidence for the DT + RAG question.",
            }
        ],
        "final_answer_task": final_answer_task,
    }


def build_dt_rag_reranker_query(question: str, retrieval_plan: Optional[dict[str, Any]]) -> str:
    if not retrieval_plan:
        return question
    dt_context = retrieval_plan.get("dt_context") or {}
    query_focus = (dt_context.get("query_focus") or {})
    intent = query_focus.get("intent") or "fault_explanation"
    focus_terms = query_focus.get("focus_terms") or []
    abnormal_features = dt_context.get("abnormal_features") or []
    interpretations = dt_context.get("engineering_interpretation") or []
    health_stage = dt_context.get("health_stage") or "unknown"

    if intent == "maintenance_decision":
        priority = (
            "Prioritize SKF chunks about inspection during operation, vibration monitoring, "
            "troubleshooting, lubrication checks, sealing checks, and cautious maintenance priorities. "
            "Do not over-prioritize generic damage tables unless they explain why inspection is needed."
        )
    elif intent == "fault_type_identification":
        priority = (
            "Prioritize SKF chunks about bearing damage classification, damage mechanisms, ISO 15243, "
            "raceway or rolling-element damage, surface distress, fatigue, and defect-related vibration."
        )
    else:
        priority = (
            "Prioritize SKF chunks about vibration monitoring, increasing vibration levels, high-frequency "
            "vibration, shock or peak vibration, rolling bearing defects, and mechanical meaning of abnormal features."
        )

    parts = [
        f"DT+RAG intent: {intent}.",
        f"Health state: {health_stage}.",
        priority,
    ]
    if abnormal_features:
        parts.append("DT abnormal features: " + ", ".join(abnormal_features) + ".")
    if interpretations:
        parts.append("DT engineering interpretation: " + ", ".join(interpretations) + ".")
    if focus_terms:
        parts.append("SKF focus terms: " + "; ".join(focus_terms[:10]) + ".")
    parts.append("Original question: " + question)
    return " ".join(" ".join(parts).split())


def classify_dt_rag_evidence_roles(chunk: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(value or "")
        for value in [
            chunk.get("chunk_id"),
            chunk.get("section"),
            chunk.get("text"),
        ]
    ).lower()
    roles: list[str] = []
    if any(term in text for term in [
        "inspection during operation", "inspect", "inspection", "condition monitoring", "vibration monitoring",
        "vibration measurement", "measurements", "monitoring", "noise", "temperature", "lubricant samples",
    ]):
        roles.append("inspection_monitoring")
    if any(term in text for term in [
        "trouble conditions", "troubleshooting", "corrective", "maintenance", "relubrication", "lubrication",
        "seal", "sealing", "contamination", "check", "remedy", "solutions",
    ]):
        roles.append("troubleshooting_maintenance")
    if any(term in text for term in [
        "bearing damage", "damage", "failure mode", "iso 15243", "fatigue", "surface distress", "spalling",
        "false brinelling", "fracture", "crack", "raceway", "rolling element", "defect", "wear",
    ]):
        roles.append("damage_mechanism")
    if "table" in str(chunk.get("chunk_id") or "").lower() or text.lstrip().startswith("table"):
        roles.append("table")
    return roles or ["general"]


def intent_role_quotas(intent: str, top_k: int) -> list[tuple[str, int]]:
    if top_k <= 0:
        return []
    if intent == "maintenance_decision":
        return [
            ("inspection_monitoring", min(3, top_k)),
            ("troubleshooting_maintenance", min(2, max(0, top_k - 1))),
            ("damage_mechanism", min(1, top_k)),
        ]
    if intent == "fault_type_identification":
        return [
            ("damage_mechanism", min(3, top_k)),
            ("inspection_monitoring", min(1, top_k)),
            ("troubleshooting_maintenance", min(1, top_k)),
        ]
    return [
        ("inspection_monitoring", min(2, top_k)),
        ("damage_mechanism", min(2, top_k)),
        ("troubleshooting_maintenance", min(1, top_k)),
    ]


def apply_dt_rag_evidence_role_quotas(
    chunks: list[dict[str, Any]],
    intent: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated = []
    for chunk in chunks:
        item = dict(chunk)
        item["dt_rag_evidence_roles"] = classify_dt_rag_evidence_roles(item)
        annotated.append(item)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    role_counts: dict[str, int] = {}
    max_table_chunks = 1 if top_k > 0 else 0

    def chunk_key(chunk: dict[str, Any]) -> str:
        return str(chunk.get("chunk_id") or chunk.get("text", "")[:160])

    def is_table_chunk(chunk: dict[str, Any]) -> bool:
        return "table" in (chunk.get("dt_rag_evidence_roles") or [])

    def can_select(chunk: dict[str, Any]) -> bool:
        key = chunk_key(chunk)
        if key in selected_keys or len(selected) >= top_k:
            return False
        if is_table_chunk(chunk) and role_counts.get("table", 0) >= max_table_chunks:
            return False
        return True

    def add_chunk(chunk: dict[str, Any], reason: str) -> bool:
        if not can_select(chunk):
            return False
        item = dict(chunk)
        item["previous_selection_reason"] = item.get("selection_reason")
        item["selection_reason"] = reason
        selected.append(item)
        selected_keys.add(chunk_key(item))
        for role in item.get("dt_rag_evidence_roles") or []:
            role_counts[role] = role_counts.get(role, 0) + 1
        return True

    for chunk in annotated:
        if str(chunk.get("selection_reason")) == "vector_forced_keep":
            add_chunk(chunk, "dt_rag_quota_vector_forced_keep")

    for role, quota in intent_role_quotas(intent, top_k):
        while role_counts.get(role, 0) < quota and len(selected) < top_k:
            added = False
            for prefer_table in (False, True):
                for chunk in annotated:
                    if prefer_table != is_table_chunk(chunk):
                        continue
                    if role in (chunk.get("dt_rag_evidence_roles") or []) and add_chunk(chunk, f"dt_rag_quota_{role}"):
                        added = True
                        break
                if added:
                    break
            if not added:
                break

    for prefer_table in (False, True):
        for chunk in annotated:
            if len(selected) >= top_k:
                break
            if prefer_table != is_table_chunk(chunk):
                continue
            add_chunk(chunk, "dt_rag_quota_score_fill")

    for rank, chunk in enumerate(selected, start=1):
        chunk["rank"] = rank

    return selected[:top_k], {
        "enabled": True,
        "intent": intent,
        "top_k": top_k,
        "input_count": len(chunks),
        "returned_count": len(selected[:top_k]),
        "role_counts": role_counts,
        "role_quotas": [{"role": role, "quota": quota} for role, quota in intent_role_quotas(intent, top_k)],
        "max_table_chunks": max_table_chunks,
    }


def build_dt_augmented_rag_query(question: str, dt_result: Optional[dict[str, Any]]) -> str:
    plan = build_dt_rag_retrieval_plan(question, dt_result)
    return str(plan.get("rag_query") or "SKF bearing vibration troubleshooting bearing damage inspection maintenance")



def shorten_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def compact_dt_context(dt_result: Optional[dict[str, Any]]) -> dict[str, Any] | None:
    if not dt_result:
        return None
    action = dt_result.get("action")
    result = dt_result.get("result") or {}
    if action == "latest_state":
        keys = [
            "experiment", "bearing_id", "timestamp", "sequence_index", "health_index",
            "health_state", "rms", "kurtosis", "crest_factor", "impulse_factor",
            "clearance_factor", "high_band_energy_ratio", "spectral_entropy", "evidence", "summary",
            "time_policy", "time_policy_description", "resolved_time_mode",
        ]
        return {"action": action, "result": {k: result.get(k) for k in keys if k in result}}
    if action == "features":
        return {
            "action": action,
            "result": {
                "experiment": result.get("experiment"),
                "bearing_id": result.get("bearing_id"),
                "timestamp": result.get("timestamp"),
                "sequence_index": result.get("sequence_index"),
                "health_state": result.get("health_state"),
                "features": result.get("features"),
                "evidence": result.get("evidence"),
                "time_policy": result.get("time_policy"),
                "time_policy_description": result.get("time_policy_description"),
                "resolved_time_mode": result.get("resolved_time_mode"),
            },
        }
    if action == "trend":
        return {"action": action, "result": result}
    if action in {"compare", "top_anomalies"}:
        compact = dict(result)
        if "bearings" in compact:
            compact["bearings"] = compact["bearings"][:6]
        if "anomalies" in compact:
            compact["anomalies"] = compact["anomalies"][:6]
        return {"action": action, "result": compact}
    return dt_result


def compact_rag_context(rag_result: Optional[dict[str, Any]], max_chunks: int = 5, max_chars_per_chunk: int = 1200) -> dict[str, Any] | None:
    if not rag_result:
        return None
    chunks = []
    for chunk in rag_result.get("chunks", [])[:max_chunks]:
        chunks.append(
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "section": chunk.get("section"),
                "pages": [chunk.get("page_start"), chunk.get("page_end")],
                "matched_queries": chunk.get("matched_queries"),
                "selection_reason": chunk.get("selection_reason"),
                "previous_selection_reason": chunk.get("previous_selection_reason"),
                "dt_rag_evidence_roles": chunk.get("dt_rag_evidence_roles"),
                "selection_query_index": chunk.get("selection_query_index"),
                "text": shorten_text(chunk.get("text", ""), max_chars_per_chunk),
            }
        )
    compact = {"query": rag_result.get("query"), "chunks": chunks}
    if rag_result.get("queries"):
        compact["queries"] = rag_result.get("queries")
    return compact


EVIDENCE_KEYWORDS = {
    "rms": ["vibration", "vibration levels", "amplitude", "mechanical problem"],
    "kurtosis": ["peak", "impact", "shock", "defect", "vibration characteristics"],
    "crest_factor": ["peak", "impact", "shock", "defect", "vibration characteristics"],
    "impulse_factor": ["impact", "shock", "defect", "vibration characteristics"],
    "clearance_factor": ["clearance", "looseness", "misalignment", "mechanical looseness"],
    "high_band_energy_ratio": ["high frequency", "high-frequency", "defect", "rolling bearing", "damage"],
    "spectral_entropy": ["frequency", "vibration characteristics", "condition", "fault"],
}


SKF_EVIDENCE_TERMS = [
    "vibration", "vibration levels", "vibration characteristics", "high frequency", "high-frequency",
    "rolling bearing", "bearing damage", "damage", "defect", "defects", "raceway", "rolling elements",
    "inspection", "condition monitoring", "troubleshooting", "lubrication", "contamination", "maintenance",
    "noise", "impact", "shock", "spalling", "spalls", "false brinelling", "misalignment", "looseness",
]


def split_evidence_sentences(text: str) -> list[str]:
    cleaned = str(text).replace("<br/>", " ").replace("<br>", " ")
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_lc = line.lower()
        if line_lc.startswith(("chapter:", "section:", "subsection:", "subsubsection:", "heading path:", "pages:")):
            continue
        if line.startswith(("#", "|", "---")):
            continue
        if re.fullmatch(r"[:\-|\s]+", line):
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        lines.append(line)
    normalized = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = []
    for part in parts:
        part = part.strip(" -")
        if 30 <= len(part) <= 360 and not part.lower().startswith("table "):
            sentences.append(part)
    if sentences:
        return sentences
    return [normalized[:360]] if normalized else []


def collect_evidence_terms(question: str, dt_context: dict[str, Any] | None) -> list[str]:
    terms = set(SKF_EVIDENCE_TERMS)
    question_lc = question.lower()
    for token in re.findall(r"[a-zA-Z][a-zA-Z_-]{2,}", question_lc):
        if len(token) > 4:
            terms.add(token.replace("-", " "))
    for feature in (dt_context or {}).get("abnormal_feature_keys") or []:
        terms.update(EVIDENCE_KEYWORDS.get(feature, []))
    return sorted(terms, key=len, reverse=True)


def build_dt_evidence_points(dt_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not dt_context:
        return []
    points: list[dict[str, Any]] = []
    dataset = dt_context.get("dataset")
    bearing_id = dt_context.get("bearing_id")
    timestamp = dt_context.get("timestamp")
    sequence_index = dt_context.get("sequence_index")
    health_stage = dt_context.get("health_stage")
    severity = dt_context.get("severity")
    if dataset or bearing_id or health_stage:
        points.append(
            {
                "type": "dt_state",
                "text": (
                    f"{bearing_id or 'The bearing'} in {dataset or 'the IMS dataset'} has health_state={health_stage or 'unknown'} "
                    f"with {severity or 'unknown'} severity at timestamp={timestamp or 'unknown'}, sequence_index={sequence_index or 'unknown'}."
                ),
                "source": "IMS DT database",
            }
        )
    feature_values = dt_context.get("feature_values") or {}
    feature_assessments = dt_context.get("feature_assessments") or {}
    for feature, value in feature_values.items():
        assessment = feature_assessments.get(feature) if isinstance(feature_assessments, dict) else {}
        label = feature_label_from_assessment(feature, assessment) if isinstance(assessment, dict) else FEATURE_NEUTRAL_LABELS.get(feature, feature.replace("_", " "))
        z_score = assessment.get("z_score") if isinstance(assessment, dict) else None
        level = assessment.get("level") if isinstance(assessment, dict) else None
        points.append(
            {
                "type": "dt_feature",
                "feature": feature,
                "value": value,
                "z_score": z_score,
                "level": level,
                "text": (
                    f"DT feature {feature}={value}"
                    + (f" with z_score={z_score}" if z_score is not None else "")
                    + (f" is assessed as {level}" if level else "")
                    + f" and described as {label}."
                ),
                "source": "IMS DT database",
            }
        )
    evidence = dt_context.get("evidence")
    if evidence:
        points.append({"type": "dt_statistical_evidence", "text": str(evidence), "source": "IMS DT database"})
    return points


def build_skf_evidence_points(question: str, rag_result: Optional[dict[str, Any]], dt_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not rag_result:
        return []
    terms = collect_evidence_terms(question, dt_context)
    points: list[dict[str, Any]] = []
    for chunk in rag_result.get("chunks", [])[:5]:
        text = chunk.get("text") or ""
        sentences = split_evidence_sentences(text)
        selected = []
        for sentence in sentences:
            sentence_lc = sentence.lower()
            if any(term in sentence_lc for term in terms):
                selected.append(sentence)
            if len(selected) >= 2:
                break
        if not selected and sentences:
            selected = sentences[:1]
        for sentence in selected:
            points.append(
                {
                    "type": "skf_manual_evidence",
                    "chunk_id": chunk.get("chunk_id"),
                    "section": chunk.get("section"),
                    "pages": [chunk.get("page_start"), chunk.get("page_end")],
                    "text": sentence,
                }
            )
        if len(points) >= 8:
            break
    return points[:8]


def has_skf_term(skf_points: list[dict[str, Any]], *terms: str) -> bool:
    text = " ".join(str(point.get("text", "")).lower() for point in skf_points)
    return any(term.lower() in text for term in terms)


def build_supported_interpretations(dt_context: dict[str, Any] | None, skf_points: list[dict[str, Any]]) -> list[str]:
    if not dt_context or not skf_points:
        return []
    features = set(dt_context.get("abnormal_feature_keys") or [])
    interpretations = []
    if "rms" in features and has_skf_term(skf_points, "vibration levels", "increase in vibration"):
        interpretations.append(
            "The DT high RMS can be interpreted as increased vibration level because the retrieved SKF evidence links mechanical problems with increased vibration levels."
        )
    if ({"kurtosis", "crest_factor", "impulse_factor"} & features) and has_skf_term(skf_points, "peak", "impact", "defect", "vibration characteristics"):
        interpretations.append(
            "The DT impulsive or peak-related features can support a cautious defect-related interpretation when combined with SKF evidence about vibration characteristics and defect-generated peaks."
        )
    if "high_band_energy_ratio" in features and has_skf_term(skf_points, "high frequency", "high-frequency", "defect"):
        interpretations.append(
            "The DT high-frequency energy is consistent with SKF evidence that high-frequency vibration can be generated by rolling bearing defects."
        )
    health_stage = dt_context.get("health_stage")
    if health_stage and health_stage != "normal":
        interpretations.append(
            f"The DT health_state={health_stage} supports treating the bearing condition as abnormal, while SKF evidence is used only to explain possible mechanisms."
        )
    return interpretations


def build_evidence_gaps(question: str, dt_context: dict[str, Any] | None, skf_points: list[dict[str, Any]]) -> list[dict[str, str]]:
    question_lc = question.lower()
    features = set((dt_context or {}).get("abnormal_feature_keys") or [])
    gaps: list[dict[str, str]] = []
    damage_requested = any(term in question_lc for term in ["damage", "fault", "defect", "failure", "diagnose", "localized", "problem", "mechanical meaning"])
    if damage_requested:
        gaps.append(
            {
                "topic": "specific failure mode or defect location",
                "meaning": "The current DT result and retrieved SKF chunks support abnormal vibration interpretation, but they do not by themselves directly determine a specific failure mode or exact defect location.",
            }
        )
    if any(term in question_lc for term in ["cause", "why", "lubrication", "contamination"]):
        if not has_skf_term(skf_points, "lubrication", "contamination", "moisture", "electric current", "misalignment", "looseness"):
            gaps.append(
                {
                    "topic": "specific root cause",
                    "meaning": "The current evidence does not directly confirm a unique physical root cause for the observed DT abnormality.",
                }
            )
    if any(term in question_lc for term in ["maintenance", "replace", "repair", "action", "decision", "should", "diagnose"]):
        if not has_skf_term(skf_points, "replace", "repair", "re-lubricate", "disassemble", "mounting procedure"):
            gaps.append(
                {
                    "topic": "specific maintenance action",
                    "meaning": "The retrieved SKF chunks support inspection, troubleshooting, or condition monitoring context, but they do not directly justify one specific corrective action for this bearing.",
                }
            )
    if features and not skf_points:
        gaps.append(
            {
                "topic": "SKF manual support",
                "meaning": "DT abnormal features are available, but no SKF manual chunks were retrieved to support a mechanical interpretation.",
            }
        )
    return gaps


def build_evidence_plan(
    question: str,
    dt_result: Optional[dict[str, Any]],
    rag_result: Optional[dict[str, Any]],
    retrieval_plan: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    dt_context = None
    if retrieval_plan and isinstance(retrieval_plan.get("dt_context"), dict):
        dt_context = retrieval_plan.get("dt_context")
    elif dt_result:
        dt_context = build_dt_context_card(question, dt_result)
    skf_points = build_skf_evidence_points(question, rag_result, dt_context)
    plan = {
        "definition": "Evidence plan separates directly usable DT facts, directly usable SKF manual evidence, cautious supported interpretations, and evidence gaps that are not directly confirmed by current evidence.",
        "dt_evidence_points": build_dt_evidence_points(dt_context),
        "skf_evidence_points": skf_points,
        "supported_interpretations": build_supported_interpretations(dt_context, skf_points),
        "evidence_gaps": build_evidence_gaps(question, dt_context, skf_points),
    }
    return plan


def build_llm_prompt(
    question: str,
    route: dict[str, Any],
    dt_result: Optional[dict[str, Any]],
    rag_result: Optional[dict[str, Any]],
    deterministic_answer: str,
    retrieval_plan: Optional[dict[str, Any]] = None,
    evidence_plan: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    route_name = route.get("route")
    rag_context_max_chunks = 8 if route_name == "dt_rag" else 5
    payload = {
        "question": question,
        "route": route,
        "dt_rag_retrieval_plan": retrieval_plan,
        "dt_context": compact_dt_context(dt_result),
        "rag_context": compact_rag_context(rag_result, max_chunks=rag_context_max_chunks),
        "fallback_summary": deterministic_answer,
    }
    if route_name == "dt_rag" and evidence_plan:
        payload["evidence_plan"] = evidence_plan
    system = (
        "You are a bearing diagnosis assistant for a digital-twin + SKF manual RAG system. "
        "Answer in clear technical English. Use only the supplied DT and RAG context. "
        "For IMS DT questions, interpret current/latest/now as the latest available timestamp in the offline IMS DT database, not the real-world current date. "
        "If the question is unrelated, politely say it is outside the system scope. "
        "Do not invent measurements, SKF guidance, or failure labels. "
        "When DT data is present, cite the key numerical evidence. "
        "When RAG context is present, mention the relevant manual sections or pages when available. "
        "Keep the answer concise and structured."
    )
    if route_name == "dt_rag":
        system += (
            " For DT+RAG questions, use this answer structure exactly: "
            "1. Direct answer; 2. Supporting DT evidence; 3. Supporting SKF evidence; 4. Evidence boundary. "
            "The Direct answer must answer the user's exact question first in one or two sentences before listing measurements. "
            "For fault-explanation questions, state the mechanical meaning of the requested DT features. "
            "For fault-type questions, state the most plausible mechanism or category cautiously. "
            "For maintenance-decision questions, state the inspection or maintenance priority cautiously. "
            "Supporting DT evidence may only contain values, states, z-scores, timestamps, and trends from the DT context, and should focus on features mentioned in the question or needed for the direct answer. "
            "Supporting SKF evidence may only summarize claims directly supported by retrieved SKF chunks or skf_evidence_points. "
            "Use cautious wording such as may, suggests, or is consistent with when connecting DT facts to SKF evidence. "
            "Do not state a specific root cause, defect location, failure mode, or maintenance action unless it is directly supported by the supplied evidence. "
            "Keep the Evidence boundary to one short sentence. "
            "If evidence_gaps are listed, summarize them briefly without adding examples. "
            "If no evidence_gaps are listed, state only that the interpretation is limited to the supplied DT and SKF evidence."
        )
    user = (
        "Generate the final user-facing answer from this structured context.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_llm_answer(
    question: str,
    route: dict[str, Any],
    dt_result: Optional[dict[str, Any]],
    rag_result: Optional[dict[str, Any]],
    deterministic_answer: str,
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    timeout: float = 60.0,
    retrieval_plan: Optional[dict[str, Any]] = None,
    evidence_plan: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if route.get("route") == "unrelated":
        return {
            "model": None,
            "answer": deterministic_answer,
            "used_llm": False,
            "reason": "Unrelated questions are answered without calling the LLM.",
        }
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("The openai package is required for --answer. Install it with: pip install openai") from exc

    model_lc = model.lower()
    if api_base is None and (model_lc.startswith("llama") or "llama" in model_lc):
        base = resolve_ollama_openai_base() or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        key = api_key or os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY")
        base = (
            api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or resolve_ollama_openai_base()
        )
    if not key:
        raise RuntimeError("OPENAI_API_KEY or OLLAMA_API_KEY is required for --answer.")

    kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
    if base:
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    if evidence_plan is None:
        evidence_plan = build_evidence_plan(question, dt_result, rag_result, retrieval_plan)
    messages = build_llm_prompt(question, route, dt_result, rag_result, deterministic_answer, retrieval_plan, evidence_plan)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""
    return {
        "model": model,
        "api_base": base,
        "answer": answer.strip(),
        "used_llm": True,
        "evidence_plan": evidence_plan,
    }


def make_deterministic_answer(question: str, route: dict[str, Any], dt_result: Optional[dict[str, Any]], rag_result: Optional[dict[str, Any]]) -> str:
    route_name = route["route"]
    if route_name == "unrelated":
        return "This question is outside the bearing DT and SKF RAG system scope."

    parts = []
    if dt_result:
        result = dt_result.get("result") or {}
        summary = result.get("summary")
        if summary:
            parts.append(summary)
        elif dt_result.get("action") == "features":
            features = result.get("features", {})
            feature_text = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in features.items())
            parts.append(f"The latest DT features are: {feature_text}.")
        if result.get("evidence"):
            parts.append(f"Evidence: {result['evidence']}")
        if result.get("latest_evidence"):
            parts.append(f"Latest evidence: {result['latest_evidence']}")

    if rag_result:
        chunks = rag_result.get("chunks", [])
        if chunks:
            sections = [c.get("section") for c in chunks if c.get("section")]
            unique_sections = list(dict.fromkeys(sections))[:3]
            if unique_sections:
                parts.append("Relevant SKF manual sections: " + "; ".join(unique_sections) + ".")
            else:
                parts.append(f"Retrieved {len(chunks)} SKF manual chunks for supporting context.")

    if not parts:
        return f"The question was routed to {route_name}, but no executable result was produced."
    return " ".join(parts)


def answer_question(
    question: str,
    db_path: Path = DEFAULT_DB,
    index_dir: Path = DEFAULT_INDEX_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_api_base: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    skip_rag: bool = False,
    generate_answer: bool = False,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_timeout: float = 180.0,
    router_mode: str = "hybrid",
    router_model: str = DEFAULT_ROUTER_MODEL,
    router_confidence_threshold: float = 0.85,
    rag_per_query_top_k: int = 3,
    rag_min_chunks_per_query: int = 1,
    retrieval_top_k: Optional[int] = None,
    reranker_url: Optional[str] = None,
    reranker_top_k: Optional[int] = None,
    reranker_timeout: float = 30.0,
    vector_keep_top_k: int = 0,
    retrieval_mode: str = "dense",
    dt_planner_mode: str = "legacy",
    dt_planner_model: str = "llama3.3:70b",
    dt_planner_api_base: Optional[str] = None,
    dt_planner_api_key: Optional[str] = None,
    dt_planner_timeout: float = 120.0,
    dt_rag_dense_keep_top_k: int = 0,
    dt_rag_bm25_keep_top_k: int = 0,
    dt_rag_reranker_query_mode: str = "full",
    force_route: Optional[str] = None,
) -> dict[str, Any]:
    errors = []
    effective_reranker_url = reranker_url or DEFAULT_RERANKER_URL
    if force_route:
        try:
            route = route_question(
                question,
                mode="rule",
                model=router_model,
                api_base=api_base,
                api_key=api_key,
                confidence_threshold=router_confidence_threshold,
            )
            base_route = route.get("route")
            base_router = route.get("router", {})
            route["route"] = force_route
            route["confidence"] = 1.0
            route["reason"] = f"Forced route for evaluation; base rule route was {base_route}."
            route["router"] = {
                "mode": "forced",
                "forced_route": force_route,
                "base_route": base_route,
                "base_router": base_router,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced in structured agent output
            route = {
                "route": force_route,
                "confidence": 1.0,
                "reason": "Forced route for evaluation; base route extraction failed.",
                "dt_entities": {"experiment": None, "bearing_id": None, "metric": None},
                "rag_query": question if force_route in {"rag_only", "dt_rag"} else "",
                "matched_signals": {},
                "router": {"mode": "forced", "forced_route": force_route, "error": str(exc)},
            }
            errors.append({"component": "router", "error": str(exc)})
    else:
        try:
            route = route_question(
                question,
                mode=router_mode,
                model=router_model,
                api_base=api_base,
                api_key=api_key,
                confidence_threshold=router_confidence_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in structured agent output
            route = {
                "route": "unrelated",
                "confidence": 0.0,
                "reason": "Router failed; see errors.",
                "dt_entities": {"experiment": None, "bearing_id": None, "metric": None},
                "rag_query": "",
                "matched_signals": {},
                "router": {"mode": router_mode, "error": str(exc)},
            }
            errors.append({"component": "router", "error": str(exc)})

    route_name = route["route"]
    dt_result = None
    rag_result = None
    retrieval_plan = None

    if route_name in {"dt_only", "dt_rag"}:
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
                if dt_result.get("errors"):
                    errors.append({"component": "dt_planner", "error": "; ".join(dt_result.get("errors") or [])})
            else:
                dt_result = run_dt(question, route, db_path)
            if route_name == "dt_rag":
                retrieval_plan = build_dt_rag_retrieval_plan(question, dt_result)
                retrieval_plan["dt_planner_mode"] = dt_planner_mode
        except Exception as exc:  # noqa: BLE001 - surfaced in structured agent output
            errors.append({"component": "dt", "error": str(exc)})
            if route_name == "dt_rag":
                retrieval_plan = build_dt_rag_retrieval_plan(question, None)
                retrieval_plan["dt_status"] = "dt_query_failed_rag_fallback"
                retrieval_plan["dt_error"] = str(exc)
                retrieval_plan["dt_planner_mode"] = dt_planner_mode

    if route_name == "dt_rag" and retrieval_plan is None:
        retrieval_plan = build_dt_rag_retrieval_plan(question, dt_result)
        retrieval_plan["dt_status"] = "dt_missing_rag_fallback"

    if route_name in {"rag_only", "dt_rag"} and not skip_rag:
        try:
            rag_query = route.get("rag_query") or question
            if route_name == "dt_rag" and retrieval_plan:
                rag_query = retrieval_plan.get("rag_query") or question
                retrieval_plan["rag_query"] = rag_query
                if not retrieval_plan.get("rag_queries"):
                    retrieval_plan["rag_queries"] = [{"query": rag_query, "purpose": "Fallback SKF evidence retrieval for DT+RAG."}]
                reranker_query = None
                if dt_rag_reranker_query_mode == "intent":
                    reranker_query = build_dt_rag_reranker_query(question, retrieval_plan)
                    retrieval_plan["reranker_query"] = reranker_query
                rag_result = run_rag(
                    rag_query,
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
                    reranker_query=reranker_query,
                    dense_keep_top_k=dt_rag_dense_keep_top_k,
                    bm25_keep_top_k=dt_rag_bm25_keep_top_k,
                )
            else:
                rag_result = run_rag(
                    rag_query,
                    index_dir,
                    manifest_path,
                    top_k,
                    embedding_model,
                    api_base,
                    api_key,
                    retrieval_top_k=retrieval_top_k,
                    reranker_url=effective_reranker_url,
                    reranker_top_k=reranker_top_k,
                    reranker_timeout=reranker_timeout,
                    vector_keep_top_k=vector_keep_top_k,
                    retrieval_mode=retrieval_mode,
                )
        except Exception as exc:  # noqa: BLE001 - surfaced in structured agent output
            errors.append({"component": "rag", "error": str(exc)})

    deterministic_answer = make_deterministic_answer(question, route, dt_result, rag_result)
    evidence_plan = build_evidence_plan(question, dt_result, rag_result, retrieval_plan)
    llm_answer = None
    if generate_answer:
        try:
            answer_api_base = llm_api_base
            answer_api_key = llm_api_key
            if answer_api_base is None and llm_model.lower().startswith("llama"):
                answer_api_base = resolve_ollama_openai_base()
            if answer_api_key is None and llm_model.lower().startswith("llama"):
                answer_api_key = os.environ.get("OLLAMA_API_KEY")
            llm_answer = generate_llm_answer(
                question=question,
                route=route,
                dt_result=dt_result,
                rag_result=rag_result,
                deterministic_answer=deterministic_answer,
                model=llm_model,
                api_base=answer_api_base or api_base,
                api_key=answer_api_key or api_key,
                timeout=llm_timeout,
                retrieval_plan=retrieval_plan,
                evidence_plan=evidence_plan,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in structured agent output
            errors.append({"component": "llm", "error": str(exc)})

    return {
        "question": question,
        "route": route,
        "dt_result": dt_result,
        "dt_rag_retrieval_plan": retrieval_plan,
        "rag_result": rag_result,
        "evidence_plan": evidence_plan,
        "errors": errors,
        "deterministic_answer": deterministic_answer,
        "llm_answer": llm_answer,
        "answer": llm_answer.get("answer") if llm_answer and llm_answer.get("answer") else deterministic_answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bearing diagnosis agent executor for DT + SKF RAG routing.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=180.0, help="Timeout in seconds for final answer generation.")
    parser.add_argument("--router-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--rag-per-query-top-k", type=int, default=3)
    parser.add_argument("--rag-min-chunks-per-query", type=int, default=1)
    parser.add_argument("--answer", action="store_true", help="Use an OpenAI-compatible LLM to generate the final natural-language answer.")
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--llm-api-base", default=None, help="Optional OpenAI-compatible base URL for final answer LLM, e.g. Ollama /v1.")
    parser.add_argument("--llm-api-key", default=None, help="Optional API key for final answer LLM.")
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="Initial vector retrieval count before optional reranking.")
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default="dense")
    parser.add_argument("--dt-planner-mode", choices=["legacy", "llm"], default="legacy", help="DT query planner used for dt_only/dt_rag routes.")
    parser.add_argument("--dt-planner-model", default="llama3.3:70b")
    parser.add_argument("--dt-planner-api-base", default=None)
    parser.add_argument("--dt-planner-api-key", default=None)
    parser.add_argument("--dt-planner-timeout", type=float, default=120.0)
    parser.add_argument("--reranker-url", default=None, help="HTTP reranker service URL, e.g. http://your-reranker-server:8002.")
    parser.add_argument("--reranker-top-k", type=int, default=None, help="Number of chunks to keep after reranking. Defaults to --top-k.")
    parser.add_argument("--reranker-timeout", type=float, default=30.0)
    parser.add_argument("--vector-keep-top-k", type=int, default=0, help="Force-keep this many top vector hits before filling the rest with reranked chunks.")
    parser.add_argument("--dt-rag-dense-keep-top-k", type=int, default=0, help="For DT+RAG hybrid_reranker only, keep this many dense hits before reranking.")
    parser.add_argument("--dt-rag-bm25-keep-top-k", type=int, default=0, help="For DT+RAG hybrid_reranker only, keep this many BM25 hits before reranking.")
    parser.add_argument("--dt-rag-reranker-query-mode", choices=["full", "intent"], default="full", help="For DT+RAG hybrid_reranker, use the full augmented query or a shorter intent-focused query for reranking.")
    parser.add_argument("--force-route", choices=["unrelated", "dt_only", "rag_only", "dt_rag"], default=None, help="Force a route for controlled evaluation/debugging.")
    parser.add_argument("--skip-rag", action="store_true", help="Do not execute SKF RAG retrieval even for rag routes.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = answer_question(
        question=args.question,
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
        llm_timeout=args.llm_timeout,
        router_mode=args.router_mode,
        router_model=args.router_model,
        router_confidence_threshold=args.router_confidence_threshold,
        rag_per_query_top_k=args.rag_per_query_top_k,
        rag_min_chunks_per_query=args.rag_min_chunks_per_query,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_mode=args.retrieval_mode,
        reranker_url=args.reranker_url,
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
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
