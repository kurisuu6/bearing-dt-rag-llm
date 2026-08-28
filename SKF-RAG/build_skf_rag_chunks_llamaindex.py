import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


DEFAULT_INPUT = "SKF-RAG/chapter_01_basics/skf_ch1_basics_llama_cleaned.json"
DEFAULT_OUTPUT = "SKF-RAG/chapter_01_basics/skf_ch1_basics_rag_chunks_llamaindex.json"
DEFAULT_CHAPTER = "Basics"


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def slugify(text: str) -> str:
    text = normalize_spaces(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "section"


def is_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line.strip()))


def heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6})\s+", line.strip())
    return len(match.group(1)) if match else 0


def heading_text(line: str) -> str:
    return normalize_spaces(re.sub(r"^#{1,6}\s+", "", line.strip()))


def is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    if re.fullmatch(r"[:\-\s|]+", stripped):
        return True
    return stripped.startswith("|") or stripped.endswith("|")


def is_noise_line(line: str) -> bool:
    text = normalize_spaces(line)
    if not text:
        return False
    return bool(re.fullmatch(r"\\?\d+\\?", text))


def table_to_text(table: dict[str, Any]) -> str:
    markdown = table.get("markdown", "")
    if markdown:
        return markdown.strip()

    table_data = table.get("table_data")
    if isinstance(table_data, list):
        return "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in table_data
            if isinstance(row, list)
        ).strip()

    return ""


def compact_paragraph_lines(lines: list[str]) -> str:
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(stripped)
            continue
        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def has_non_heading_content(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or is_heading(stripped):
            continue
        if re.fullmatch(r"Table\s+\d+\s+cont\.?", stripped, flags=re.IGNORECASE):
            continue
        return True
    return False


def section_from_stack(stack: list[tuple[int, str]], chapter: str) -> str:
    candidates = [title for _, title in stack if title.lower() != chapter.lower()]
    return candidates[-1] if candidates else chapter


def heading_path_from_stack(stack: list[tuple[int, str]], chapter: str) -> list[str]:
    path = []
    for _, title in stack:
        if path and path[-1] == title:
            continue
        if title.lower() == chapter.lower() and path:
            continue
        path.append(title)
    if not path or path[0].lower() != chapter.lower():
        path.insert(0, chapter)
    return path


def dedupe_heading_path(path: list[str], chapter: str) -> list[str]:
    cleaned = []
    for item in path:
        title = normalize_spaces(item)
        if not title:
            continue
        if cleaned and cleaned[-1].lower() == title.lower():
            continue
        if title.lower() == chapter.lower() and cleaned:
            continue
        cleaned.append(title)
    if not cleaned or cleaned[0].lower() != chapter.lower():
        cleaned.insert(0, chapter)
    return cleaned


def heading_metadata(path: list[str], chapter: str) -> dict[str, Any]:
    clean_path = dedupe_heading_path(path, chapter)
    return {
        "chapter": chapter,
        "section": clean_path[1] if len(clean_path) > 1 else chapter,
        "subsection": clean_path[2] if len(clean_path) > 2 else None,
        "subsubsection": clean_path[3] if len(clean_path) > 3 else None,
        "heading_path": clean_path,
        "heading_path_text": " > ".join(clean_path),
        "heading_depth": len(clean_path),
    }


def split_page_into_text_units(
    markdown: str,
    page_number: int,
    chapter: str,
    initial_stack: list[tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    units = []
    stack = list(initial_stack)
    current_lines: list[str] = []
    current_start_heading: Optional[str] = None
    current_path = heading_path_from_stack(stack, chapter)

    def flush():
        nonlocal current_lines, current_start_heading, current_path
        text = compact_paragraph_lines(current_lines)
        current_lines = []
        if not text or not has_non_heading_content(text):
            return
        metadata = heading_metadata(current_path, chapter)
        units.append(
            {
                "content_type": "text",
                "text": text,
                "page_start": page_number,
                "page_end": page_number,
                **metadata,
            }
        )

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if is_markdown_table_line(stripped) or is_noise_line(stripped):
            continue

        if is_heading(stripped):
            flush()
            level = heading_level(stripped)
            title = heading_text(stripped)
            stack = [(lvl, name) for lvl, name in stack if lvl < level]
            stack.append((level, title))
            current_start_heading = title
            current_path = heading_path_from_stack(stack, chapter)
            current_lines.append(stripped)
            continue

        current_lines.append(line)

    flush()
    return units, stack

def merge_adjacent_text_units(
    units: list[dict[str, Any]],
    target_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for unit in units:
        if unit["content_type"] != "text":
            merged.append(unit)
            continue

        if (
            merged
            and merged[-1]["content_type"] == "text"
            and merged[-1]["section"] == unit["section"]
            and merged[-1]["heading_path"] == unit["heading_path"]
            and len(merged[-1]["text"]) < target_chars
            and len(merged[-1]["text"]) + len(unit["text"]) + 2 <= max_chars
        ):
            merged[-1]["text"] = merged[-1]["text"].rstrip() + "\n\n" + unit["text"]
            merged[-1]["page_end"] = unit["page_end"]
        else:
            merged.append(dict(unit))

    return merged


def make_context_text(unit: dict[str, Any], chapter: str) -> str:
    prefix = [
        f"Chapter: {chapter}",
        f"Section: {unit.get('section') or chapter}",
    ]
    if unit.get("subsection"):
        prefix.append(f"Subsection: {unit['subsection']}")
    if unit.get("subsubsection"):
        prefix.append(f"Subsubsection: {unit['subsubsection']}")
    prefix.extend(
        [
            f"Heading path: {unit.get('heading_path_text') or ' > '.join(unit.get('heading_path') or [chapter])}",
            f"Pages: {unit.get('page_start')}-{unit.get('page_end')}",
            "",
        ]
    )
    return "\n".join(prefix) + unit["text"].strip()

def build_text_documents(
    units: list[dict[str, Any]],
    source_json: Path,
    source_file: str,
    parser: str,
    chapter: str,
) -> list[Document]:
    documents = []
    for index, unit in enumerate(units, start=1):
        if unit["content_type"] != "text":
            continue
        documents.append(
            Document(
                text=make_context_text(unit, chapter),
                metadata={
                    "source_json": str(source_json),
                    "source_file": source_file,
                    "parser": parser,
                    "chapter": chapter,
                    "section": unit.get("section"),
                    "subsection": unit.get("subsection"),
                    "subsubsection": unit.get("subsubsection"),
                    "heading_path": unit.get("heading_path"),
                    "heading_path_text": unit.get("heading_path_text"),
                    "heading_depth": unit.get("heading_depth"),
                    "page_start": unit["page_start"],
                    "page_end": unit["page_end"],
                    "content_type": "text",
                    "semantic_unit_index": index,
                },
            )
        )
    return documents

def split_table_rows(markdown_table: str, max_chars: int) -> list[str]:
    lines = [line for line in markdown_table.splitlines() if line.strip()]
    if len(markdown_table) <= max_chars or len(lines) <= 3:
        return [markdown_table.strip()]

    header = lines[:2]
    rows = lines[2:]
    parts = []
    current = header[:]

    for row in rows:
        candidate = "\n".join(current + [row])
        if len(candidate) > max_chars and len(current) > len(header):
            parts.append("\n".join(current).strip())
            current = header[:] + [row]
        else:
            current.append(row)

    if len(current) > len(header):
        parts.append("\n".join(current).strip())

    return parts or [markdown_table.strip()]


def page_heading_metadata(markdown: str, chapter: str, initial_stack: list[tuple[int, str]]) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    stack = list(initial_stack)
    latest_path = heading_path_from_stack(stack, chapter)
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not is_heading(stripped):
            continue
        level = heading_level(stripped)
        title = heading_text(stripped)
        stack = [(lvl, name) for lvl, name in stack if lvl < level]
        stack.append((level, title))
        latest_path = heading_path_from_stack(stack, chapter)
    return heading_metadata(latest_path, chapter), stack


def build_table_chunks(
    data: dict[str, Any],
    input_path: Path,
    chapter: str,
    max_table_chars: int,
    chunk_id_prefix: str,
) -> list[dict[str, Any]]:
    chunks = []
    table_counter = 0
    heading_stack: list[tuple[int, str]] = [(1, chapter)]

    for page in data.get("pages", []):
        page_number = page.get("page")
        meta, heading_stack = page_heading_metadata(page.get("markdown", ""), chapter, heading_stack)

        for table in page.get("tables", []):
            table_text = table_to_text(table)
            if not table_text:
                continue
            table_counter += 1
            for part_index, part in enumerate(split_table_rows(table_text, max_table_chars), start=1):
                prefix = [
                    f"Chapter: {chapter}",
                    f"Section: {meta.get('section') or chapter}",
                ]
                if meta.get("subsection"):
                    prefix.append(f"Subsection: {meta['subsection']}")
                if meta.get("subsubsection"):
                    prefix.append(f"Subsubsection: {meta['subsubsection']}")
                prefix.extend(
                    [
                        f"Heading path: {meta.get('heading_path_text')}",
                        f"Page: {page_number}",
                        f"Table ID: {table.get('table_id') or f'table_{table_counter:03d}'}",
                        "",
                        part,
                    ]
                )
                text = "\n".join(prefix)
                chunks.append(
                    {
                        "id": f"{chunk_id_prefix}_table_{table_counter:03d}_part_{part_index:02d}",
                        "type": "table",
                        "text": text,
                        "metadata": {
                            "source_json": str(input_path),
                            "source_file": data.get("source_file"),
                            "parser": data.get("parser"),
                            "chapter": chapter,
                            "section": meta.get("section"),
                            "subsection": meta.get("subsection"),
                            "subsubsection": meta.get("subsubsection"),
                            "heading_path": meta.get("heading_path"),
                            "heading_path_text": meta.get("heading_path_text"),
                            "heading_depth": meta.get("heading_depth"),
                            "page_start": page_number,
                            "page_end": page_number,
                            "content_type": "table",
                            "table_id": table.get("table_id") or f"table_{table_counter:03d}",
                            "table_part": part_index,
                        },
                        "char_count": len(text),
                    }
                )

    return chunks


def infer_chunk_id_prefix(output_path: Path, default: str = "ch") -> str:
    for part in output_path.parts:
        match = re.search(r"chapter_(\d{2})", part)
        if match:
            return f"ch{int(match.group(1))}"
    match = re.search(r"ch(?:apter)?[_-]?(\d{1,2})", output_path.name, flags=re.IGNORECASE)
    if match:
        return f"ch{int(match.group(1))}"
    return default


def node_to_chunk(node, index: int, chunk_id_prefix: str) -> dict[str, Any]:
    metadata = dict(getattr(node, "metadata", {}) or {})
    text = node.get_content(metadata_mode="none").strip()
    section_slug = slugify(metadata.get("section", "section"))
    return {
        "id": f"{chunk_id_prefix}_text_{index:03d}_{section_slug}",
        "type": "text",
        "text": text,
        "metadata": metadata,
        "char_count": len(text),
    }


def build_chunks(
    input_path: Path,
    output_path: Path,
    chapter: str,
    chunk_size: int,
    chunk_overlap: int,
    merge_target_chars: int,
    merge_max_chars: int,
    max_table_chars: int,
    chunk_id_prefix: Optional[str] = None,
) -> dict[str, Any]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    chunk_id_prefix = chunk_id_prefix or infer_chunk_id_prefix(output_path, default=slugify(chapter)[:12] or "ch")

    text_units = []
    heading_stack: list[tuple[int, str]] = [(1, chapter)]
    for page in data.get("pages", []):
        page_units, heading_stack = split_page_into_text_units(
            page.get("markdown", ""),
            int(page.get("page")),
            chapter,
            heading_stack,
        )
        text_units.extend(page_units)

    merged_text_units = merge_adjacent_text_units(
        text_units,
        target_chars=merge_target_chars,
        max_chars=merge_max_chars,
    )

    documents = build_text_documents(
        merged_text_units,
        source_json=input_path,
        source_file=data.get("source_file"),
        parser=data.get("parser"),
        chapter=chapter,
    )
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )
    text_nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
    text_chunks = [node_to_chunk(node, index, chunk_id_prefix) for index, node in enumerate(text_nodes, start=1)]

    table_chunks = build_table_chunks(
        data,
        input_path=input_path,
        chapter=chapter,
        max_table_chars=max_table_chars,
        chunk_id_prefix=chunk_id_prefix,
    )

    chunks = text_chunks + table_chunks
    chunks.sort(
        key=lambda chunk: (
            chunk["metadata"].get("page_start") or 0,
            0 if chunk["type"] == "text" else 1,
            chunk["id"],
        )
    )

    result = {
        "source_json": str(input_path),
        "source_file": data.get("source_file"),
        "chapter": chapter,
        "chunking_strategy": {
            "goal": "SKF bearing handbook RAG question answering",
            "steps": [
                "Parse LlamaParse page markdown into heading-aware semantic text units.",
                "Skip markdown table lines from text units to avoid duplicated text/table chunks.",
                "Merge adjacent short text units only when section and heading path match.",
                "Use LlamaIndex SentenceSplitter for long text units while preserving metadata.",
                "Keep tables as independent chunks; split very long markdown tables by rows with header repeated.",
                "Add chapter, section, subsection, subsubsection, full heading path, page range, content type, and source metadata to every chunk.",
            ],
            "llamaindex_splitter": "llama_index.core.node_parser.SentenceSplitter",
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
            "merge_target_chars": merge_target_chars,
            "merge_max_chars": merge_max_chars,
            "max_table_chars": max_table_chars,
        },
        "semantic_text_unit_count": len(text_units),
        "merged_text_unit_count": len(merged_text_units),
        "document_count": len(documents),
        "chunk_id_prefix": chunk_id_prefix,
        "chunk_count": len(chunks),
        "text_chunk_count": len(text_chunks),
        "table_chunk_count": len(table_chunks),
        "chunks": chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build SKF RAG chunks from cleaned LlamaParse JSON using LlamaIndex."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input LlamaParse JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output chunks JSON.")
    parser.add_argument("--chapter", default=DEFAULT_CHAPTER, help="Chapter name.")
    parser.add_argument("--chunk-size", type=int, default=650, help="LlamaIndex token chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=90, help="LlamaIndex token overlap.")
    parser.add_argument("--merge-target-chars", type=int, default=1200)
    parser.add_argument("--merge-max-chars", type=int, default=2600)
    parser.add_argument("--max-table-chars", type=int, default=3500)
    parser.add_argument("--chunk-id-prefix", default=None, help="Optional unique chunk id prefix, e.g. ch2.")
    args = parser.parse_args()

    result = build_chunks(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chapter=args.chapter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        merge_target_chars=args.merge_target_chars,
        merge_max_chars=args.merge_max_chars,
        max_table_chars=args.max_table_chars,
        chunk_id_prefix=args.chunk_id_prefix,
    )

    print(
        f"[OK] semantic units {result['semantic_text_unit_count']} -> "
        f"chunks {result['chunk_count']} "
        f"(text {result['text_chunk_count']}, table {result['table_chunk_count']}) "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
