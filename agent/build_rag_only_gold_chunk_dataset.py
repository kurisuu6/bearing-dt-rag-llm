#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[0]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from build_dt_rag_gold_chunk_dataset import (  # noqa: E402
    LABEL_BY_SCORE,
    call_json_llm,
    chunk_batches,
    compact_chunk,
    load_chunks,
    load_json,
    make_openai_client,
    normalize_annotation_items,
    resolve_api_base,
    resolve_api_key,
    write_json,
)

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "SKF-RAG" / "chapters_01_11" / "skf_chapters_01_11_rag_chunks_llamaindex.json"
DEFAULT_CANDIDATES_OUTPUT = AGENT_DIR / "eval" / "rag_only_gold_chunk_candidates_v1.json"
DEFAULT_DATASET_OUTPUT = AGENT_DIR / "eval" / "rag_only_gold_chunk_dataset_v1.json"
DEFAULT_REFERENCES_OUTPUT = AGENT_DIR / "eval" / "rag_only_reference_answers_gold_v1.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Question file must be a JSON list.")
    return [row for row in rows if row.get("expected_route") == "rag_only"]


def build_annotation_prompt(
    question: str,
    batch: list[dict[str, Any]],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You are annotating gold evidence chunks for an academic RAG-only SKF manual QA benchmark. "
        "For each SKF manual chunk, decide whether it can serve as the single best evidence for answering "
        "the question. Return strict JSON only. Judge only the supplied chunk text; do not favor chunks "
        "because they may be retrieved by any system."
    )
    user_payload = {
        "scoring_rubric": {
            "3": "highly_relevant: the chunk can independently serve as the best single SKF evidence for the question",
            "2": "relevant: useful and on-topic SKF evidence, but less direct or incomplete",
            "1": "weakly_relevant: background only",
            "0": "irrelevant: does not support answering the question",
        },
        "selection_preferences": [
            "Prefer chunks that directly answer the wording of the question.",
            "Prefer specific procedure, definition, troubleshooting, inspection, lubrication, mounting, or damage evidence over broad introductions.",
            "Prefer table chunks when the question asks for tabular symptoms, causes, actions, intervals, values, or comparisons.",
            "Prefer the most focused chunk even if another broader chunk is semantically related.",
        ],
        "question": question,
        "chunks": [compact_chunk(chunk, max_chunk_chars) for chunk in batch],
        "required_output_schema": {
            "items": [
                {
                    "chunk_id": "same id as input",
                    "score": "0, 1, 2, or 3",
                    "label": "irrelevant | weakly_relevant | relevant | highly_relevant",
                    "supports": ["background"],
                    "reason": "brief reason",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def select_candidate_annotations(
    annotations: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    relevant = [item for item in annotations if int(item.get("score", 0)) > 0]
    relevant.sort(key=lambda item: (int(item.get("score", 0)), str(item.get("chunk_id"))), reverse=True)
    return relevant[:max_candidates]


def build_gold_selection_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    candidate_payload = []
    for item in candidates:
        chunk = chunks_by_id[item["chunk_id"]]
        candidate_payload.append({"annotation": item, "chunk": compact_chunk(chunk, max_chunk_chars)})
    system = (
        "You are selecting the single gold SKF manual evidence chunk for a RAG-only benchmark item. "
        "Choose exactly one gold_chunk_id from the candidate chunks. The gold chunk should be the best "
        "single evidence for answering the question. Return strict JSON only."
    )
    user_payload = {
        "question": question,
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
    gold_chunk: dict[str, Any],
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You are writing a ground-truth reference answer for a RAG-only SKF manual QA benchmark. "
        "Use only the supplied single gold SKF chunk. Do not invent unsupported procedures, values, "
        "failure causes, or maintenance actions. Return strict JSON only."
    )
    user_payload = {
        "question": question,
        "gold_skf_chunk": compact_chunk(gold_chunk, max_chunk_chars),
        "answer_requirements": [
            "Answer the user's question directly.",
            "Use only information supported by the gold SKF chunk.",
            "Keep the answer concise and technical.",
            "State a brief evidence boundary if the chunk does not support a more specific claim.",
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
        return existing, existing["dataset_row"], existing["reference_row"]

    print(f"[ITEM] {item['id']} {item['question']}", flush=True)
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
    selection = call_json_llm(
        client,
        args.annotation_model,
        build_gold_selection_prompt(
            question=item["question"],
            candidates=selected_candidates,
            chunks_by_id=chunks_by_id,
            max_chunk_chars=args.selection_chunk_chars,
        ),
        args.max_retries,
        args.retry_sleep,
    )
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
    reference_payload = call_json_llm(
        client,
        args.annotation_model,
        build_reference_prompt(
            question=item["question"],
            gold_chunk=gold_chunk,
            max_chunk_chars=args.reference_chunk_chars,
        ),
        args.max_retries,
        args.retry_sleep,
    )
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
        "chapter": item.get("chapter"),
        "chapter_title": item.get("chapter_title"),
        "question": item.get("question"),
        "expected_route": item.get("expected_route"),
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
        "chapter": item.get("chapter"),
        "chapter_title": item.get("chapter_title"),
        "question": item.get("question"),
        "ground_truth": reference_answer,
        "reference_answer": reference_answer,
        "evidence_boundary": str(reference_payload.get("evidence_boundary") or "").strip(),
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
        "generation_note": "Reference answer generated from one LLM-selected gold SKF chunk.",
        "annotation_model": args.annotation_model,
    }
    candidate_row = {
        "id": item.get("id"),
        "question": item.get("question"),
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
    parser = argparse.ArgumentParser(description="Build RAG-only single-gold-chunk reference dataset from full SKF chunk annotation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_OUTPUT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--references-output", type=Path, default=DEFAULT_REFERENCES_OUTPUT)
    parser.add_argument("--annotation-model", default="gpt-4o-mini")
    parser.add_argument("--annotation-api-base", default=None)
    parser.add_argument("--annotation-api-key", default=None)
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
        raise RuntimeError("No RAG-only questions selected.")

    chunks = load_chunks(args.chunks)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if not chunks:
        raise RuntimeError("No SKF chunks loaded.")

    api_base = resolve_api_base(args.annotation_api_base, args.annotation_model)
    api_key = resolve_api_key(args.annotation_api_key, args.annotation_model)
    if not api_key:
        raise RuntimeError("Annotation API key is missing. Set OPENAI_API_KEY or pass --annotation-api-key.")
    client = make_openai_client(api_base, api_key, args.request_timeout)

    existing_map = build_existing_candidate_map(args.candidates_output) if args.resume else {}
    candidate_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    print(f"[INFO] RAG-only questions: {len(questions)}", flush=True)
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
