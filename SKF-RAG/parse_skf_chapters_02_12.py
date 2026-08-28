import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PDF = "SKF-RAG/SKF.pdf"
DEFAULT_PARSER = "SKF-RAG/parse_skf_with_llamaparse.py"


CHAPTERS = [
    {
        "chapter": 2,
        "name": "mounting_rolling_bearings",
        "folder": "SKF-RAG/chapter_02_mounting_rolling_bearings",
        "page_range": "50-95",
        "output": "skf_ch2_mounting_rolling_bearings_llama_cleaned.json",
    },
    {
        "chapter": 3,
        "name": "mounting_bearing_units",
        "folder": "SKF-RAG/chapter_03_mounting_bearing_units",
        "page_range": "98-125",
        "output": "skf_ch3_mounting_bearing_units_llama_cleaned.json",
    },
    {
        "chapter": 4,
        "name": "mounting_bearing_housings",
        "folder": "SKF-RAG/chapter_04_mounting_bearing_housings",
        "page_range": "128-143",
        "output": "skf_ch4_mounting_bearing_housings_llama_cleaned.json",
    },
    {
        "chapter": 5,
        "name": "installing_seals",
        "folder": "SKF-RAG/chapter_05_installing_seals",
        "page_range": "146-161",
        "output": "skf_ch5_installing_seals_llama_cleaned.json",
    },
    {
        "chapter": 6,
        "name": "alignment",
        "folder": "SKF-RAG/chapter_06_alignment",
        "page_range": "164-181",
        "output": "skf_ch6_alignment_llama_cleaned.json",
    },
    {
        "chapter": 7,
        "name": "lubrication",
        "folder": "SKF-RAG/chapter_07_lubrication",
        "page_range": "184-219",
        "output": "skf_ch7_lubrication_llama_cleaned.json",
    },
    {
        "chapter": 8,
        "name": "inspection",
        "folder": "SKF-RAG/chapter_08_inspection",
        "page_range": "222-231",
        "output": "skf_ch8_inspection_llama_cleaned.json",
    },
    {
        "chapter": 9,
        "name": "troubleshooting",
        "folder": "SKF-RAG/chapter_09_troubleshooting",
        "page_range": "234-255",
        "output": "skf_ch9_troubleshooting_llama_cleaned.json",
    },
    {
        "chapter": 10,
        "name": "dismounting",
        "folder": "SKF-RAG/chapter_10_dismounting",
        "page_range": "258-291",
        "output": "skf_ch10_dismounting_llama_cleaned.json",
    },
    {
        "chapter": 11,
        "name": "bearing_damage_and_their_causes",
        "folder": "SKF-RAG/chapter_11_bearing_damage_and_their_causes",
        "page_range": "294-327",
        "output": "skf_ch11_bearing_damage_and_their_causes_llama_cleaned.json",
    },
    {
        "chapter": 12,
        "name": "maintenance_support",
        "folder": "SKF-RAG/chapter_12_maintenance_support",
        "page_range": "330-335",
        "output": "skf_ch12_maintenance_support_llama_cleaned.json",
    },
]


def parse_page_range(page_range: str) -> list[int]:
    pages = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return pages


def page_ranges_from_pages(pages: list[int], batch_size: int) -> list[str]:
    ranges = []
    for index in range(0, len(pages), batch_size):
        batch = pages[index : index + batch_size]
        if batch[0] == batch[-1]:
            ranges.append(str(batch[0]))
        else:
            ranges.append(f"{batch[0]}-{batch[-1]}")
    return ranges


def run_parser(
    pdf: Path,
    parser_script: Path,
    output_path: Path,
    page_range: str,
    tier: str,
    version: str,
):
    cmd = [
        sys.executable,
        str(parser_script),
        "--pdf",
        str(pdf),
        "--output",
        str(output_path),
        "--page-range",
        page_range,
        "--tier",
        tier,
        "--version",
        version,
    ]
    subprocess.run(cmd, check=True)


def merge_chapter_outputs(
    partial_paths: list[Path],
    final_output_path: Path,
    source_file: str,
    chapter_page_range: str,
):
    pages = []
    tables = []
    raw_batches = []

    for batch_index, path in enumerate(partial_paths, start=1):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages.extend(data.get("pages", []))
        tables.extend(data.get("tables", []))
        raw_batches.append(
            {
                "batch": batch_index,
                "page_range": data.get("page_range"),
                "raw_result": data.get("raw_result"),
            }
        )

    pages.sort(key=lambda item: item.get("page", 0))
    tables.sort(key=lambda item: (item.get("page", 0), item.get("table_id", "")))

    merged = {
        "source_file": source_file,
        "parser": "llamaparse",
        "page_range": chapter_page_range,
        "page_count": len(pages),
        "table_count": len(tables),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "tables": tables,
        "raw_result_batches": raw_batches,
    }
    final_output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_chapter(
    chapter: dict,
    pdf: Path,
    parser_script: Path,
    tier: str,
    version: str,
    fallback_batch_size: int,
):
    folder = Path(chapter["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / chapter["output"]

    print(
        f"[RUN] Chapter {chapter['chapter']:02d} "
        f"pages {chapter['page_range']} -> {output_path}",
        flush=True,
    )
    try:
        run_parser(pdf, parser_script, output_path, chapter["page_range"], tier, version)
        return
    except subprocess.CalledProcessError:
        if fallback_batch_size < 1:
            raise
        print(
            f"[WARN] Whole-chapter upload failed for chapter {chapter['chapter']:02d}. "
            f"Retrying in {fallback_batch_size}-page batches and merging output.",
            flush=True,
        )

    pages = parse_page_range(chapter["page_range"])
    ranges = page_ranges_from_pages(pages, fallback_batch_size)
    partial_paths = []
    with tempfile.TemporaryDirectory(prefix=f"skf_ch{chapter['chapter']:02d}_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for index, page_range in enumerate(ranges, start=1):
            partial_path = temp_dir_path / f"part_{index:02d}.json"
            partial_paths.append(partial_path)
            print(
                f"[RUN] Chapter {chapter['chapter']:02d} part {index}/{len(ranges)} "
                f"pages {page_range}",
                flush=True,
            )
            run_parser(pdf, parser_script, partial_path, page_range, tier, version)

        merge_chapter_outputs(
            partial_paths,
            output_path,
            source_file=str(pdf),
            chapter_page_range=chapter["page_range"],
        )
    print(f"[OK] Merged chapter {chapter['chapter']:02d} -> {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Parse SKF.pdf chapters 2-12 into chapter folders with LlamaParse."
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--parser", default=DEFAULT_PARSER)
    parser.add_argument(
        "--tier",
        default="agentic",
        choices=["fast", "cost_effective", "agentic", "agentic_plus"],
    )
    parser.add_argument("--version", default="latest")
    parser.add_argument(
        "--only",
        default="",
        help='Optional comma list of chapter numbers, e.g. "2,3,7".',
    )
    parser.add_argument(
        "--fallback-batch-size",
        type=int,
        default=10,
        help="If whole-chapter upload fails, retry with this many source PDF pages per batch and merge the JSON. Set 0 to disable fallback.",
    )
    args = parser.parse_args()

    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        raise RuntimeError(
            "Missing LLAMA_CLOUD_API_KEY. Run: export LLAMA_CLOUD_API_KEY='your_key'"
        )

    pdf = Path(args.pdf)
    parser_script = Path(args.parser)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not parser_script.exists():
        raise FileNotFoundError(f"Parser script not found: {parser_script}")

    selected = None
    if args.only.strip():
        selected = {int(item.strip()) for item in args.only.split(",") if item.strip()}

    for chapter in CHAPTERS:
        if selected and chapter["chapter"] not in selected:
            continue
        run_chapter(
            chapter,
            pdf,
            parser_script,
            args.tier,
            args.version,
            args.fallback_batch_size,
        )

    print("[OK] Finished parsing selected chapters.")


if __name__ == "__main__":
    main()
