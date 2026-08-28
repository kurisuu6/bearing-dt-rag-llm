#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from bearing_agent import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_MANIFEST,
    answer_question,
    resolve_ollama_openai_base,
)

DEFAULT_QUESTIONS = AGENT_DIR / "eval" / "agent_english_questions_v2.json"
DEFAULT_OUTPUT = AGENT_DIR / "eval" / "dt_rag_frozen_inputs_v1.jsonl"
REQUIRED_DT_TABLES = {"feature_records", "bearing_snapshots", "experiments"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dt_rag_questions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Question file must be a JSON list: {path}")
    return [row for row in data if row.get("expected_route") == "dt_rag"]


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


def resolve_llama_api_base(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or resolve_ollama_openai_base()


def resolve_llama_api_key(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or os.environ.get("OLLAMA_API_KEY")


def validate_dt_database(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"DT database does not exist: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"DT database is empty: {path}")
    conn = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in conn.execute("select name from sqlite_master where type='table'")
        }
    finally:
        conn.close()
    missing = sorted(REQUIRED_DT_TABLES - tables)
    if missing:
        raise RuntimeError(
            f"DT database is missing required tables {missing}: {path}. "
            "Use DT/outputs/ims_dt.db, not an empty .sqlite file."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DT+RAG routing and DT planning once, then freeze DT evidence and DT-enhanced RAG queries."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--dt-planner-mode", choices=["legacy", "llm"], default="llm")
    parser.add_argument("--dt-planner-model", default="llama3.3:70b")
    parser.add_argument("--dt-planner-api-base", default=None)
    parser.add_argument("--dt-planner-api-key", default=None)
    parser.add_argument("--dt-planner-timeout", type=float, default=120.0)
    parser.add_argument("--force-route", choices=["dt_rag"], default="dt_rag")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_dt_rag_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    api_base = resolve_api_base(args.openai_api_base, args.manifest)
    api_key = get_openai_api_key(args.openai_api_key)
    dt_planner_api_base = resolve_llama_api_base(args.dt_planner_api_base)
    dt_planner_api_key = resolve_llama_api_key(args.dt_planner_api_key)
    validate_dt_database(args.db)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[FREEZE] {index}/{len(questions)} {item.get('id')}: {question}", flush=True)
        raw_result = answer_question(
            question=question,
            db_path=args.db,
            index_dir=args.index_dir,
            manifest_path=args.manifest,
            embedding_model=args.embedding_model,
            api_base=api_base,
            api_key=api_key,
            skip_rag=True,
            generate_answer=False,
            llm_model=DEFAULT_LLM_MODEL,
            router_mode="rule",
            dt_planner_mode=args.dt_planner_mode,
            dt_planner_model=args.dt_planner_model,
            dt_planner_api_base=dt_planner_api_base,
            dt_planner_api_key=dt_planner_api_key,
            dt_planner_timeout=args.dt_planner_timeout,
            force_route=args.force_route,
        )
        plan = raw_result.get("dt_rag_retrieval_plan") or {}
        rows.append(
            {
                "id": item.get("id"),
                "question": question,
                "expected_route": item.get("expected_route"),
                "forced_route": args.force_route,
                "dt_planner_mode": args.dt_planner_mode,
                "dt_planner_model": args.dt_planner_model if args.dt_planner_mode == "llm" else None,
                "dt_result": raw_result.get("dt_result"),
                "dt_rag_retrieval_plan": plan,
                "rag_query": plan.get("rag_query"),
                "retrieval_strategy": plan.get("retrieval_strategy"),
                "raw_result": raw_result,
                "errors": raw_result.get("errors") or [],
            }
        )

    save_jsonl(rows, args.output)
    print(f"[OK] Frozen DT+RAG inputs: {len(rows)} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
