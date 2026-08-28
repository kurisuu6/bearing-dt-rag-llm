import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fitz
from llama_cloud import LlamaCloud


DEFAULT_PDF = "SKF_bearing.pdf"
DEFAULT_OUTPUT = "skf_ch1_basics_llamaparse.json"
DEFAULT_PAGE_RANGE = "14-47"


def to_plain(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return to_plain(value.dict())
    return str(value)


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def remove_page_number_tags(text: str) -> str:
    return re.sub(
        r"\\?<page_number>\s*.*?\s*\\?</page_number>",
        "",
        text,
        flags=re.DOTALL,
    )


def clean_page_number_tags(value: Any):
    if isinstance(value, str):
        return remove_page_number_tags(value)
    if isinstance(value, list):
        return [clean_page_number_tags(item) for item in value]
    if isinstance(value, dict):
        return {
            key: clean_page_number_tags(item)
            for key, item in value.items()
        }
    return value


def is_figure_line(line: str) -> bool:
    lowered = normalize_spaces(line).lower()
    return bool(
        re.match(r"^(fig\.?|figure)\s*\d+", lowered)
        or re.match(r"^(diagram|digram)\s*\d*", lowered)
    )


def clean_markdown(markdown: str) -> str:
    markdown = remove_page_number_tags(markdown)
    lines = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not normalize_spaces(line):
            lines.append("")
            continue
        if is_figure_line(line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_markdown_tables(markdown: str):
    tables = []
    current = []

    for line in markdown.splitlines():
        if "|" in line and re.search(r"\S", line):
            current.append(line)
            continue
        if current:
            if len(current) >= 2:
                tables.append("\n".join(current))
            current = []

    if current and len(current) >= 2:
        tables.append("\n".join(current))

    return tables


def parse_page_numbers(page_range: str) -> list[int]:
    page_numbers = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range segment: {part}")
            page_numbers.extend(range(start, end + 1))
        else:
            page_number = int(part)
            if page_number < 1:
                raise ValueError(f"Invalid page number: {part}")
            page_numbers.append(page_number)
    if not page_numbers:
        raise ValueError(f"Invalid empty page range: {page_range}")
    return page_numbers


def create_page_subset_pdf(pdf_path: Path, page_range: str) -> tuple[Path, list[int]]:
    page_numbers = parse_page_numbers(page_range)
    source = fitz.open(pdf_path)
    subset = fitz.open()
    try:
        for page_number in page_numbers:
            if page_number > source.page_count:
                raise ValueError(
                    f"Page {page_number} is outside PDF page count {source.page_count}."
                )
            page_index = page_number - 1
            subset.insert_pdf(source, from_page=page_index, to_page=page_index)

        temp = tempfile.NamedTemporaryFile(
            prefix=f"{pdf_path.stem}_pages_",
            suffix=".pdf",
            delete=False,
        )
        temp_path = Path(temp.name)
        temp.close()
        subset.save(temp_path)
        return temp_path, page_numbers
    finally:
        subset.close()
        source.close()


def page_number_from_result(page_obj, fallback_index: int):
    page = to_plain(page_obj)
    for key in ("page", "page_number", "pageNumber"):
        value = page.get(key) if isinstance(page, dict) else None
        if isinstance(value, int):
            return value
    return fallback_index


def markdown_from_page(page_obj) -> str:
    page = to_plain(page_obj)
    if isinstance(page, dict):
        for key in ("markdown", "md", "text"):
            value = page.get(key)
            if isinstance(value, str):
                return value
    return str(page_obj)


def normalize_result(
    result,
    source_file: str,
    page_range: str,
    page_number_map: Optional[list[int]] = None,
):
    raw_result = clean_page_number_tags(to_plain(result))
    markdown_pages = []

    markdown_container = getattr(result, "markdown", None)
    if markdown_container is not None and hasattr(markdown_container, "pages"):
        markdown_pages = list(markdown_container.pages)
    elif isinstance(raw_result, dict):
        markdown = raw_result.get("markdown", {})
        if isinstance(markdown, dict):
            markdown_pages = markdown.get("pages", []) or []

    pages = []
    all_tables = []
    for index, page_obj in enumerate(markdown_pages, start=1):
        if page_number_map and index <= len(page_number_map):
            page_number = page_number_map[index - 1]
        else:
            page_number = page_number_from_result(page_obj, index)
        raw_markdown = markdown_from_page(page_obj)
        markdown = clean_markdown(raw_markdown)
        tables = extract_markdown_tables(markdown)

        page_tables = []
        for table_index, table_markdown in enumerate(tables, start=1):
            table = {
                "table_id": f"p{page_number}_t{table_index}",
                "page": page_number,
                "format": "markdown",
                "markdown": table_markdown,
            }
            page_tables.append(table)
            all_tables.append(table)

        pages.append(
            {
                "page": page_number,
                "markdown": markdown,
                "tables": page_tables,
            }
        )

    return {
        "source_file": source_file,
        "parser": "llamaparse",
        "page_range": page_range,
        "page_count": len(pages),
        "table_count": len(all_tables),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "tables": all_tables,
        "raw_result": raw_result,
    }


def parse_pdf(pdf_path: Path, page_range: str, tier: str, version: str):
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        raise RuntimeError("Missing LLAMA_CLOUD_API_KEY environment variable.")

    client = LlamaCloud()
    uploaded_file = client.files.create(file=str(pdf_path), purpose="parse")
    return client.parsing.parse(
        file_id=uploaded_file.id,
        tier=tier,
        version=version,
        page_ranges={"target_pages": page_range},
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,
                    "merge_continued_tables": True,
                },
                "inline_images": False,
            }
        },
        expand=["markdown"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parse SKF_bearing.pdf with LlamaParse/LlamaCloud."
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Input PDF path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument(
        "--page-range",
        default=DEFAULT_PAGE_RANGE,
        help='1-based pages/ranges for LlamaParse, e.g. "14-47".',
    )
    parser.add_argument(
        "--tier",
        default="agentic",
        choices=["fast", "cost_effective", "agentic", "agentic_plus"],
        help="LlamaParse tier.",
    )
    parser.add_argument("--version", default="latest", help="LlamaParse version.")
    parser.add_argument(
        "--upload-full-pdf",
        action="store_true",
        help="Upload the full PDF and let LlamaParse apply --page-range. By default, upload only a local page-range subset.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_number_map = None
    temp_upload_pdf = None
    try:
        if args.upload_full_pdf:
            upload_pdf_path = pdf_path
            upload_page_range = args.page_range
        else:
            temp_upload_pdf, page_number_map = create_page_subset_pdf(
                pdf_path,
                args.page_range,
            )
            upload_pdf_path = temp_upload_pdf
            upload_page_range = f"1-{len(page_number_map)}"
            print(
                f"[INFO] Created local page-range PDF {temp_upload_pdf} "
                f"from source pages {args.page_range}.",
                flush=True,
            )

        result = parse_pdf(upload_pdf_path, upload_page_range, args.tier, args.version)
    finally:
        if temp_upload_pdf and temp_upload_pdf.exists():
            temp_upload_pdf.unlink()

    normalized = normalize_result(
        result,
        str(pdf_path),
        args.page_range,
        page_number_map=page_number_map,
    )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] LlamaParse extracted pages {args.page_range}, "
        f"tables {normalized['table_count']} -> {output_path}"
    )


if __name__ == "__main__":
    main()
