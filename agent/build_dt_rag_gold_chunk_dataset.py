#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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
    build_dt_rag_retrieval_plan,
    compact_dt_context,
)
from dt_llm_planner import run_dt_with_llm_planner  # type: ignore  # noqa: E402

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "SKF-RAG" / "chapters_01_11" / "skf_chapters_01_11_rag_chunks_llamaindex.json"
DEFAULT_CANDIDATES_OUTPUT = AGENT_DIR / "eval" / "dt_rag_gold_chunk_candidates_v1.json"
DEFAULT_DATASET_OUTPUT = AGENT_DIR / "eval" / "dt_rag_gold_chunk_dataset_v1.json"
DEFAULT_REFERENCES_OUTPUT = AGENT_DIR / "eval" / "dt_rag_reference_answers_gold_v1.json"

LABEL_BY_SCORE = {
    0: "irrelevant",
    1: "weakly_relevant",
    2: "relevant",
    3: "highly_relevant",
}

VALID_SUPPORTS = {
    "fault_explanation",
    "fault_type_identification",
    "maintenance_decision",
    "background",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Question file must be a JSON list.")
    return [row for row in rows if row.get("expected_route") == "dt_rag"]


def parse_chunk_header(text: str) -> dict[str, Any]:
    header: dict[str, Any] = {}
    for line in text.splitlines()[:6]:
        if line.startswith("Chapter:"):
            header["chapter"] = line.replace("Chapter:", "", 1).strip()
        elif line.startswith("Section:"):
            header["section"] = line.replace("Section:", "", 1).strip()
        elif line.startswith("Heading path:"):
            header["heading_path"] = line.replace("Heading path:", "", 1).strip()
        elif line.startswith("Pages:"):
            value = line.replace("Pages:", "", 1).strip()
            match = re.match(r"(\d+)(?:-(\d+))?", value)
            if match:
                header["page_start"] = int(match.group(1))
                header["page_end"] = int(match.group(2) or match.group(1))
    return header


def normalize_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    text = str(raw.get("text") or raw.get("embedding_text") or "")
    header = parse_chunk_header(text)
    chunk_id = raw.get("id") or metadata.get("chunk_id")
    return {
        "chunk_id": str(chunk_id),
        "type": raw.get("type") or metadata.get("chunk_type"),
        "chapter": raw.get("chapter") or metadata.get("chapter") or header.get("chapter"),
        "section": raw.get("section") or metadata.get("section") or header.get("section"),
        "heading_path": raw.get("heading_path") or metadata.get("heading_path") or header.get("heading_path"),
        "page_start": raw.get("page_start") or metadata.get("page_start") or header.get("page_start"),
        "page_end": raw.get("page_end") or metadata.get("page_end") or header.get("page_end"),
        "text": text,
    }


def load_chunks(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Chunk file must be a list or a dict with a 'chunks' list.")
    chunks = [normalize_chunk(row) for row in rows]
    return [chunk for chunk in chunks if chunk.get("chunk_id") and chunk.get("text")]


def compact_chunk(chunk: dict[str, Any], max_chars: int) -> dict[str, Any]:
    text = " ".join(str(chunk.get("text") or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 18].rstrip() + " ... [truncated]"
    return {
        "chunk_id": chunk.get("chunk_id"),
        "type": chunk.get("type"),
        "chapter": chunk.get("chapter"),
        "section": chunk.get("section"),
        "heading_path": chunk.get("heading_path"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "text": text,
    }


def chunk_batches(chunks: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [chunks[index : index + batch_size] for index in range(0, len(chunks), batch_size)]


def resolve_api_base(cli_value: Optional[str], model: str) -> Optional[str]:
    if cli_value:
        return cli_value.rstrip("/")
    if "llama" in model.lower():
        base = os.environ.get("OLLAMA_OPENAI_BASE") or os.environ.get("OLLAMA_BASE_URL")
        if not base:
            return None
        base = base.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"
    base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    return base.rstrip("/") if base else None


def resolve_api_key(cli_value: Optional[str], model: str) -> str:
    if cli_value:
        return cli_value
    if "llama" in model.lower():
        return os.environ.get("OLLAMA_API_KEY") or "ollama"
    return os.environ.get("OPENAI_API_KEY") or ""


def make_openai_client(api_base: Optional[str], api_key: str, timeout: float):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("The openai package is required. Install it with: pip install openai") from exc

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
        raise ValueError(f"LLM did not return a JSON object: {text[:500]}")
    return json.loads(text[start : end + 1])


def call_json_llm(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return strip_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"LLM JSON call failed after {max_retries} attempts: {last_error}") from last_error


def build_annotation_prompt(
    question: str,
    dt_evidence: dict[str, Any] | None,
    enhanced_query: str,
    batch: list[dict[str, Any]],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You are annotating gold evidence chunks for an academic DT+RAG benchmark about bearing diagnosis. "
        "For each SKF manual chunk, decide whether it can serve as the single best SKF manual evidence for answering "
        "the question when combined with the supplied DT evidence. Return strict JSON only. "
        "Do not favor chunks because they were retrieved by any system; judge only the supplied text."
    )
    user_payload = {
        "scoring_rubric": {
            "3": "highly_relevant: the chunk can independently serve as the gold SKF evidence for the question",
            "2": "relevant: useful SKF evidence, but may need another chunk or is less direct",
            "1": "weakly_relevant: background only",
            "0": "irrelevant: does not support answering the question",
        },
        "selection_preferences": [
            "Prefer direct fault explanation, vibration monitoring, bearing damage, inspection, or maintenance evidence matching the question intent.",
            "Prefer specific evidence over introductions or broad background.",
            "Prefer text chunks over tables unless the table directly answers the question.",
            "Do not require the chunk to mention IMS feature names such as RMS or kurtosis; SKF mechanism evidence may use vibration, shock, impact, monitoring, damage, or inspection terminology.",
        ],
        "question": question,
        "dt_evidence": dt_evidence,
        "enhanced_query": enhanced_query,
        "chunks": [compact_chunk(chunk, max_chunk_chars) for chunk in batch],
        "required_output_schema": {
            "items": [
                {
                    "chunk_id": "same id as input",
                    "score": "0, 1, 2, or 3",
                    "label": "irrelevant | weakly_relevant | relevant | highly_relevant",
                    "supports": ["fault_explanation | fault_type_identification | maintenance_decision | background"],
                    "reason": "brief reason",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def normalize_annotation_items(payload: dict[str, Any], batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_ids = {str(chunk["chunk_id"]) for chunk in batch}
    raw_items = payload.get("items") or []
    if not isinstance(raw_items, list):
        raw_items = []
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id not in valid_ids:
            continue
        try:
            score = int(item.get("score", 0))
        except Exception:
            score = 0
        score = min(3, max(0, score))
        label = str(item.get("label") or LABEL_BY_SCORE[score])
        raw_supports = item.get("supports")
        if isinstance(raw_supports, str):
            support_parts = re.split(r"[,;/|]", raw_supports)
        elif isinstance(raw_supports, list):
            support_parts = raw_supports
        else:
            support_parts = []
        supports = []
        for part in support_parts:
            normalized = str(part).strip().lower().replace(" ", "_")
            if normalized in VALID_SUPPORTS and normalized not in supports:
                supports.append(normalized)
        by_id[chunk_id] = {
            "chunk_id": chunk_id,
            "score": score,
            "label": label,
            "supports": supports,
            "reason": str(item.get("reason") or "").strip(),
        }
    for chunk_id in valid_ids:
        by_id.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "score": 0,
                "label": "irrelevant",
                "supports": [],
                "reason": "Missing from model output; treated as irrelevant.",
            },
        )
    return [by_id[str(chunk["chunk_id"])] for chunk in batch]


def build_gold_selection_prompt(
    question: str,
    dt_evidence: dict[str, Any] | None,
    enhanced_query: str,
    candidates: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    candidate_payload = []
    for item in candidates:
        chunk = chunks_by_id[item["chunk_id"]]
        candidate_payload.append(
            {
                "annotation": item,
                "chunk": compact_chunk(chunk, max_chunk_chars),
            }
        )
    system = (
        "You are selecting the single gold SKF manual evidence chunk for a DT+RAG benchmark item. "
        "Choose exactly one gold_chunk_id. The gold chunk should be the best single SKF evidence for answering "
        "the question together with the DT evidence. Return strict JSON only."
    )
    user_payload = {
        "question": question,
        "dt_evidence": dt_evidence,
        "enhanced_query": enhanced_query,
        "candidate_chunks": candidate_payload,
        "required_output_schema": {
            "gold_chunk_id": "one chunk_id from candidate_chunks",
            "selection_reason": "why this is the best single gold evidence",
            "secondary_chunk_ids": ["optional 0-2 supporting chunks, not used as main retrieval gold"],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def build_reference_prompt(
    question: str,
    dt_evidence: dict[str, Any] | None,
    enhanced_query: str,
    gold_chunk: dict[str, Any],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You are writing a ground-truth reference answer for a DT+RAG bearing diagnosis benchmark. "
        "Use only the supplied DT evidence and the single gold SKF chunk. Do not invent exact root causes, "
        "defect locations, or maintenance actions that are not supported. Return strict JSON only."
    )
    user_payload = {
        "question": question,
        "dt_evidence": dt_evidence,
        "enhanced_query": enhanced_query,
        "gold_skf_chunk": compact_chunk(gold_chunk, max_chunk_chars),
        "answer_requirements": [
            "Answer the user's question directly.",
            "Use the DT state/features as the observed condition.",
            "Use the SKF chunk as the mechanical or maintenance grounding.",
            "State evidence limits briefly if the exact failure type or action cannot be determined.",
        ],
        "required_output_schema": {
            "reference_answer": "concise English answer",
            "evidence_boundary": "brief limitation statement",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def build_dt_payload(
    item: dict[str, Any],
    db_path: Path,
    dt_planner_model: str,
    dt_planner_api_base: Optional[str],
    dt_planner_api_key: Optional[str],
    dt_planner_timeout: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    dt_result = run_dt_with_llm_planner(
        question=item["question"],
        db_path=db_path,
        model=dt_planner_model,
        api_base=dt_planner_api_base,
        api_key=dt_planner_api_key,
        timeout=dt_planner_timeout,
    )
    dt_evidence = compact_dt_context(dt_result)
    retrieval_plan = build_dt_rag_retrieval_plan(item["question"], dt_result)
    return dt_result, dt_evidence, str(retrieval_plan.get("rag_query") or item["question"])


def select_candidate_annotations(
    annotations: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    relevant = [item for item in annotations if int(item.get("score", 0)) > 0]
    relevant.sort(key=lambda item: (int(item.get("score", 0)), str(item.get("chunk_id"))), reverse=True)
    return relevant[:max_candidates]


def build_existing_candidate_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id")): row for row in rows if row.get("id")}


def build_one_item(
    item: dict[str, Any],
    chunks: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    client: Any,
    args: argparse.Namespace,
    existing: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if existing and args.resume and existing.get("status") == "complete":
        dataset_row = existing["dataset_row"]
        reference_row = existing["reference_row"]
        return existing, dataset_row, reference_row

    print(f"[ITEM] {item['id']} {item['question']}", flush=True)
    dt_result, dt_evidence, enhanced_query = build_dt_payload(
        item=item,
        db_path=args.db,
        dt_planner_model=args.dt_planner_model,
        dt_planner_api_base=args.dt_planner_api_base,
        dt_planner_api_key=args.dt_planner_api_key,
        dt_planner_timeout=args.dt_planner_timeout,
    )

    annotations: list[dict[str, Any]] = []
    batches = chunk_batches(chunks, args.batch_size)
    existing_annotations = {
        str(annotation.get("chunk_id")): annotation
        for annotation in (existing or {}).get("annotations", [])
        if annotation.get("chunk_id")
    }

    for batch_index, batch in enumerate(batches, start=1):
        missing_batch = [chunk for chunk in batch if str(chunk["chunk_id"]) not in existing_annotations]
        if missing_batch:
            print(f"[ANNOTATE] {item['id']} batch {batch_index}/{len(batches)} chunks={len(missing_batch)}", flush=True)
            messages = build_annotation_prompt(
                question=item["question"],
                dt_evidence=dt_evidence,
                enhanced_query=enhanced_query,
                batch=missing_batch,
                max_chunk_chars=args.annotation_chunk_chars,
            )
            payload = call_json_llm(client, args.annotation_model, messages, args.max_retries, args.retry_sleep)
            for annotation in normalize_annotation_items(payload, missing_batch):
                existing_annotations[annotation["chunk_id"]] = annotation
            if args.sleep:
                time.sleep(args.sleep)
        annotations.extend(existing_annotations[str(chunk["chunk_id"])] for chunk in batch)

    selected_candidates = select_candidate_annotations(annotations, args.max_gold_candidates)
    if not selected_candidates:
        annotations.sort(key=lambda item_: (int(item_.get("score", 0)), str(item_.get("chunk_id"))), reverse=True)
        selected_candidates = annotations[: min(args.max_gold_candidates, len(annotations))]

    print(f"[SELECT] {item['id']} candidate_count={len(selected_candidates)}", flush=True)
    selection_messages = build_gold_selection_prompt(
        question=item["question"],
        dt_evidence=dt_evidence,
        enhanced_query=enhanced_query,
        candidates=selected_candidates,
        chunks_by_id=chunks_by_id,
        max_chunk_chars=args.selection_chunk_chars,
    )
    selection = call_json_llm(client, args.annotation_model, selection_messages, args.max_retries, args.retry_sleep)
    gold_chunk_id = str(selection.get("gold_chunk_id") or "")
    if gold_chunk_id not in chunks_by_id:
        gold_chunk_id = selected_candidates[0]["chunk_id"]
        selection["gold_chunk_id"] = gold_chunk_id
        selection["selection_reason"] = (
            str(selection.get("selection_reason") or "")
            + " Fallback: model returned an invalid gold_chunk_id, so the highest-scored candidate was used."
        ).strip()
    gold_chunk = chunks_by_id[gold_chunk_id]

    print(f"[ANSWER] {item['id']} gold={gold_chunk_id}", flush=True)
    reference_messages = build_reference_prompt(
        question=item["question"],
        dt_evidence=dt_evidence,
        enhanced_query=enhanced_query,
        gold_chunk=gold_chunk,
        max_chunk_chars=args.reference_chunk_chars,
    )
    reference_payload = call_json_llm(client, args.annotation_model, reference_messages, args.max_retries, args.retry_sleep)
    reference_answer = str(reference_payload.get("reference_answer") or "").strip()

    annotations_by_id = {annotation["chunk_id"]: annotation for annotation in annotations}
    gold_annotation = annotations_by_id.get(gold_chunk_id, {"chunk_id": gold_chunk_id})
    secondary_chunk_ids = [
        str(chunk_id)
        for chunk_id in selection.get("secondary_chunk_ids", [])
        if str(chunk_id) in chunks_by_id and str(chunk_id) != gold_chunk_id
    ][:2]

    dataset_row = {
        "id": item.get("id"),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "question": item.get("question"),
        "expected_route": item.get("expected_route"),
        "dt_query_params": ((dt_result.get("planner") or {}).get("planned_dt_query") if isinstance(dt_result.get("planner"), dict) else None),
        "dt_result": dt_result,
        "dt_evidence": dt_evidence,
        "enhanced_query": enhanced_query,
        "gold_chunk_id": gold_chunk_id,
        "gold_chunk": compact_chunk(gold_chunk, args.output_chunk_chars),
        "gold_annotation": gold_annotation,
        "secondary_chunk_ids": secondary_chunk_ids,
        "secondary_chunks": [compact_chunk(chunks_by_id[chunk_id], args.output_chunk_chars) for chunk_id in secondary_chunk_ids],
        "gold_selection": selection,
        "annotation_model": args.annotation_model,
        "annotation_summary": {
            "total_chunks": len(chunks),
            "score_3": sum(1 for annotation in annotations if annotation.get("score") == 3),
            "score_2": sum(1 for annotation in annotations if annotation.get("score") == 2),
            "score_1": sum(1 for annotation in annotations if annotation.get("score") == 1),
            "score_0": sum(1 for annotation in annotations if annotation.get("score") == 0),
        },
    }
    reference_row = {
        "id": item.get("id"),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "question": item.get("question"),
        "ground_truth": reference_answer,
        "reference_answer": reference_answer,
        "evidence_boundary": str(reference_payload.get("evidence_boundary") or "").strip(),
        "dt_evidence": dt_evidence,
        "rag_query": enhanced_query,
        "enhanced_query": enhanced_query,
        "source_chunk_ids": [gold_chunk_id],
        "gold_chunk_id": gold_chunk_id,
        "secondary_chunk_ids": secondary_chunk_ids,
        "source_pages": sorted(
            {
                page
                for page in [gold_chunk.get("page_start"), gold_chunk.get("page_end")]
                if page is not None
            }
        ),
        "source_chunks": [compact_chunk(gold_chunk, args.output_chunk_chars)],
        "source_mode": "full_corpus_llm_gold_chunk_annotation",
        "generation_note": "Reference answer generated from DT evidence and one LLM-selected gold SKF chunk.",
        "annotation_model": args.annotation_model,
    }
    candidate_row = {
        "id": item.get("id"),
        "question": item.get("question"),
        "dt_evidence": dt_evidence,
        "enhanced_query": enhanced_query,
        "annotations": annotations,
        "selected_candidates": selected_candidates,
        "gold_chunk_id": gold_chunk_id,
        "secondary_chunk_ids": secondary_chunk_ids,
        "dataset_row": dataset_row,
        "reference_row": reference_row,
        "status": "complete",
    }
    return candidate_row, dataset_row, reference_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DT+RAG gold-chunk reference dataset from full SKF chunk annotation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_OUTPUT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--references-output", type=Path, default=DEFAULT_REFERENCES_OUTPUT)
    parser.add_argument("--annotation-model", default="gpt-4o-mini")
    parser.add_argument("--annotation-api-base", default=None)
    parser.add_argument("--annotation-api-key", default=None)
    parser.add_argument("--dt-planner-model", default="gpt-4o-mini")
    parser.add_argument("--dt-planner-api-base", default=None)
    parser.add_argument("--dt-planner-api-key", default=None)
    parser.add_argument("--dt-planner-timeout", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-gold-candidates", type=int, default=12)
    parser.add_argument("--annotation-chunk-chars", type=int, default=1200)
    parser.add_argument("--selection-chunk-chars", type=int, default=1600)
    parser.add_argument("--reference-chunk-chars", type=int, default=2200)
    parser.add_argument("--output-chunk-chars", type=int, default=2200)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between annotation batches.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows from candidates-output.")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.question_id:
        questions = [item for item in questions if item.get("id") == args.question_id]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("No DT+RAG questions selected.")

    chunks = load_chunks(args.chunks)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if not chunks:
        raise RuntimeError("No SKF chunks loaded.")

    api_base = resolve_api_base(args.annotation_api_base, args.annotation_model)
    api_key = resolve_api_key(args.annotation_api_key, args.annotation_model)
    if not api_key:
        raise RuntimeError("Annotation API key is missing. Set OPENAI_API_KEY or pass --annotation-api-key.")
    client = make_openai_client(api_base, api_key, args.request_timeout)

    if not args.dt_planner_api_base:
        args.dt_planner_api_base = resolve_api_base(None, args.dt_planner_model)
    if not args.dt_planner_api_key:
        args.dt_planner_api_key = resolve_api_key(None, args.dt_planner_model)

    existing_map = build_existing_candidate_map(args.candidates_output) if args.resume else {}
    candidate_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    print(f"[INFO] DT+RAG questions: {len(questions)}", flush=True)
    print(f"[INFO] SKF chunks: {len(chunks)}", flush=True)
    print(f"[INFO] Pair annotations: {len(questions) * len(chunks)}", flush=True)
    print(f"[INFO] Annotation model: {args.annotation_model}", flush=True)
    print(f"[INFO] Annotation API base: {api_base}", flush=True)
    print(f"[INFO] Batch size: {args.batch_size}", flush=True)

    for item in questions:
        existing = existing_map.get(str(item.get("id")))
        candidate_row, dataset_row, reference_row = build_one_item(
            item=item,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
            client=client,
            args=args,
            existing=existing,
        )
        candidate_rows.append(candidate_row)
        dataset_rows.append(dataset_row)
        reference_rows.append(reference_row)
        write_json(args.candidates_output, candidate_rows)
        write_json(args.dataset_output, dataset_rows)
        write_json(args.references_output, reference_rows)
        print(f"[OK] Saved progress after {item['id']}", flush=True)

    print(f"[DONE] Candidates -> {args.candidates_output}", flush=True)
    print(f"[DONE] Dataset -> {args.dataset_output}", flush=True)
    print(f"[DONE] References -> {args.references_output}", flush=True)


if __name__ == "__main__":
    main()
