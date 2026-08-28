#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_skf_rag_chunks_llamaindex import build_chunks  # noqa: E402
from build_llamaindex_vector_index import build_index  # noqa: E402

CHAPTERS_1_12 = [
    {
        "number": 1,
        "chapter": "Basics",
        "dir": "chapter_01_basics",
        "input": "skf_ch1_basics_llama_cleaned2.json",
        "chunks": "skf_ch1_basics_rag_chunks_llamaindex_v2.json",
    },
    {
        "number": 2,
        "chapter": "Mounting rolling bearings",
        "dir": "chapter_02_mounting_rolling_bearings",
        "input": "skf_ch2_mounting_rolling_bearings_llama_cleaned.json",
        "chunks": "skf_ch2_mounting_rolling_bearings_rag_chunks_llamaindex.json",
    },
    {
        "number": 3,
        "chapter": "Mounting bearing units",
        "dir": "chapter_03_mounting_bearing_units",
        "input": "skf_ch3_mounting_bearing_units_llama_cleaned.json",
        "chunks": "skf_ch3_mounting_bearing_units_rag_chunks_llamaindex.json",
    },
    {
        "number": 4,
        "chapter": "Mounting bearing housings",
        "dir": "chapter_04_mounting_bearing_housings",
        "input": "skf_ch4_mounting_bearing_housings_llama_cleaned.json",
        "chunks": "skf_ch4_mounting_bearing_housings_rag_chunks_llamaindex.json",
    },
    {
        "number": 5,
        "chapter": "Installing seals",
        "dir": "chapter_05_installing_seals",
        "input": "skf_ch5_installing_seals_llama_cleaned.json",
        "chunks": "skf_ch5_installing_seals_rag_chunks_llamaindex.json",
    },
    {
        "number": 6,
        "chapter": "Alignment",
        "dir": "chapter_06_alignment",
        "input": "skf_ch6_alignment_llama_cleaned.json",
        "chunks": "skf_ch6_alignment_rag_chunks_llamaindex.json",
    },
    {
        "number": 7,
        "chapter": "Lubrication",
        "dir": "chapter_07_lubrication",
        "input": "skf_ch7_lubrication_llama_cleaned.json",
        "chunks": "skf_ch7_lubrication_rag_chunks_llamaindex.json",
    },
    {
        "number": 8,
        "chapter": "Inspection",
        "dir": "chapter_08_inspection",
        "input": "skf_ch8_inspection_llama_cleaned.json",
        "chunks": "skf_ch8_inspection_rag_chunks_llamaindex.json",
    },
    {
        "number": 9,
        "chapter": "Troubleshooting",
        "dir": "chapter_09_troubleshooting",
        "input": "skf_ch9_troubleshooting_llama_cleaned.json",
        "chunks": "skf_ch9_troubleshooting_rag_chunks_llamaindex.json",
    },
    {
        "number": 10,
        "chapter": "Dismounting",
        "dir": "chapter_10_dismounting",
        "input": "skf_ch10_dismounting_llama_cleaned.json",
        "chunks": "skf_ch10_dismounting_rag_chunks_llamaindex.json",
    },
    {
        "number": 11,
        "chapter": "Bearing damage and their causes",
        "dir": "chapter_11_bearing_damage_and_their_causes",
        "input": "skf_ch11_bearing_damage_and_their_causes_llama_cleaned.json",
        "chunks": "skf_ch11_bearing_damage_and_their_causes_rag_chunks_llamaindex.json",
    },
    {
        "number": 12,
        "chapter": "Maintenance support",
        "dir": "chapter_12_maintenance_support",
        "input": "skf_ch12_maintenance_support_llama_cleaned.json",
        "chunks": "skf_ch12_maintenance_support_rag_chunks_llamaindex.json",
    },
]


def chapter_path(chapter: dict[str, Any], filename_key: str) -> Path:
    return SCRIPT_DIR / chapter["dir"] / chapter[filename_key]


def build_chapter_chunks(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    for chapter in CHAPTERS_1_12:
        input_path = chapter_path(chapter, "input")
        output_path = chapter_path(chapter, "chunks")
        if chapter["number"] == 1 and output_path.exists() and not args.rebuild_ch1:
            print(f"[SKIP] Chapter 01 chunks already exist: {output_path}", flush=True)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "chapter_number": chapter["number"],
                    "chapter": chapter["chapter"],
                    "input": str(input_path),
                    "output": str(output_path),
                    "chunk_count": data.get("chunk_count", len(data.get("chunks", []))),
                    "skipped": True,
                }
            )
            continue
        if not input_path.exists():
            raise FileNotFoundError(f"Missing chapter input: {input_path}")
        print(f"[BUILD] Chapter {chapter['number']:02d}: {input_path.name} -> {output_path.name}", flush=True)
        result = build_chunks(
            input_path=input_path,
            output_path=output_path,
            chapter=chapter["chapter"],
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            merge_target_chars=args.merge_target_chars,
            merge_max_chars=args.merge_max_chars,
            max_table_chars=args.max_table_chars,
            chunk_id_prefix=f"ch{chapter['number']}",
        )
        summaries.append(
            {
                "chapter_number": chapter["number"],
                "chapter": chapter["chapter"],
                "input": str(input_path),
                "output": str(output_path),
                "semantic_text_unit_count": result.get("semantic_text_unit_count"),
                "merged_text_unit_count": result.get("merged_text_unit_count"),
                "chunk_count": result.get("chunk_count"),
                "text_chunk_count": result.get("text_chunk_count"),
                "table_chunk_count": result.get("table_chunk_count"),
                "skipped": False,
            }
        )
    return summaries


def combine_chunks(combined_output: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = []
    seen_ids = set()
    chapter_sources = []
    for chapter in CHAPTERS_1_12:
        chunks_path = chapter_path(chapter, "chunks")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks for chapter {chapter['number']}: {chunks_path}")
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chapter_chunks = data.get("chunks", [])
        for chunk in chapter_chunks:
            chunk_id = chunk.get("id")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk id: {chunk_id}")
            seen_ids.add(chunk_id)
            metadata = chunk.setdefault("metadata", {})
            metadata.setdefault("chapter_number", chapter["number"])
            chunks.append(chunk)
        chapter_sources.append(
            {
                "chapter_number": chapter["number"],
                "chapter": chapter["chapter"],
                "chunks_file": str(chunks_path),
                "chunk_count": len(chapter_chunks),
            }
        )

    chunks.sort(
        key=lambda chunk: (
            chunk.get("metadata", {}).get("chapter_number") or 0,
            chunk.get("metadata", {}).get("page_start") or 0,
            0 if chunk.get("type") == "text" else 1,
            chunk.get("id") or "",
        )
    )
    result = {
        "name": "SKF chapters 1-12 combined RAG chunks",
        "chapter_range": "1-12",
        "source_file": "SKF.pdf",
        "chapter_count": len(CHAPTERS_1_12),
        "chunk_count": len(chunks),
        "text_chunk_count": sum(1 for chunk in chunks if chunk.get("type") == "text"),
        "table_chunk_count": sum(1 for chunk in chunks if chunk.get("type") == "table"),
        "chapter_sources": chapter_sources,
        "build_summaries": summaries,
        "chunks": chunks,
    }
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    combined_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Combined {len(chunks)} chunks -> {combined_output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKF chapters 1-12 chunks and optionally vector index.")
    parser.add_argument("--combined-output", type=Path, default=SCRIPT_DIR / "chapters_01_12" / "skf_chapters_01_12_rag_chunks_llamaindex.json")
    parser.add_argument("--persist-dir", type=Path, default=SCRIPT_DIR / "chapters_01_12" / "llamaindex_vector_index_openai")
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / "chapters_01_12" / "llamaindex_vector_index_openai_manifest.json")
    parser.add_argument("--chunk-size", type=int, default=650)
    parser.add_argument("--chunk-overlap", type=int, default=90)
    parser.add_argument("--merge-target-chars", type=int, default=1200)
    parser.add_argument("--merge-max-chars", type=int, default=2600)
    parser.add_argument("--max-table-chars", type=int, default=3500)
    parser.add_argument("--rebuild-ch1", action="store_true", help="Rebuild chapter 1 chunks from cleaned2 instead of reusing existing v2 file.")
    parser.add_argument("--build-index", action="store_true", help="Build vector index from the combined chunks. Calls embedding API unless provider=mock.")
    parser.add_argument("--embedding-provider", choices=["mock", "openai"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    summaries = build_chapter_chunks(args)
    combined = combine_chunks(args.combined_output, summaries)
    print(
        f"[SUMMARY] chapters={combined['chapter_count']} chunks={combined['chunk_count']} "
        f"text={combined['text_chunk_count']} table={combined['table_chunk_count']}",
        flush=True,
    )

    if args.build_index:
        manifest = build_index(
            input_path=args.combined_output,
            persist_dir=args.persist_dir,
            manifest_path=args.manifest,
            provider=args.embedding_provider,
            model=args.embedding_model,
            mock_dim=1536,
            dimensions=args.embedding_dimensions,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
            embed_batch_size=args.embed_batch_size,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        print(f"[OK] Vector index -> {args.persist_dir}", flush=True)
        print(f"[OK] Manifest -> {args.manifest}", flush=True)
        print(json.dumps({"node_count": manifest.get("node_count"), "embedding": manifest.get("embedding")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
