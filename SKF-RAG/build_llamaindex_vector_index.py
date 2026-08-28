import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


DEFAULT_INPUT = "SKF-RAG/chapter_01_basics/skf_ch1_basics_rag_chunks_llamaindex.json"
DEFAULT_PERSIST_DIR = "SKF-RAG/chapter_01_basics/llamaindex_vector_index"
DEFAULT_MANIFEST = "SKF-RAG/chapter_01_basics/llamaindex_vector_index_manifest.json"


def metadata_to_plain(metadata: dict[str, Any]) -> dict[str, Any]:
    plain = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            plain[key] = value
        elif isinstance(value, list):
            plain[key] = " > ".join(str(item) for item in value)
        else:
            plain[key] = json.dumps(value, ensure_ascii=False)
    return plain


def strip_existing_context_prefix(text: str) -> str:
    lines = text.strip().splitlines()
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if any(
            stripped.startswith(prefix)
            for prefix in (
                "Chapter:",
                "Section:",
                "Subsection:",
                "Subsubsection:",
                "Heading path:",
                "Pages:",
                "Page:",
                "Table ID:",
            )
        ):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


def build_embedding_text(chunk: dict[str, Any]) -> str:
    if chunk.get("embedding_text"):
        return chunk["embedding_text"].strip()

    text = strip_existing_context_prefix(chunk.get("text", ""))
    metadata = chunk.get("metadata", {})
    chapter = metadata.get("chapter", "")
    section = metadata.get("section", "")
    heading_path = metadata.get("heading_path", "")

    if isinstance(heading_path, list):
        heading_path = " > ".join(str(item) for item in heading_path)

    context_lines = []
    if chapter:
        context_lines.append(f"Chapter: {chapter}")
    if section:
        context_lines.append(f"Section: {section}")
    if heading_path:
        context_lines.append(f"Heading path: {heading_path}")

    if context_lines:
        return "\n".join(context_lines) + "\n\n" + text
    return text


def chunks_to_nodes(chunks: list[dict[str, Any]]) -> list[TextNode]:
    nodes = []
    for chunk in chunks:
        text = build_embedding_text(chunk)
        if not text:
            continue

        metadata = metadata_to_plain(chunk.get("metadata", {}))
        metadata.update(
            {
                "chunk_id": chunk.get("id"),
                "chunk_type": chunk.get("type"),
                "char_count": chunk.get("char_count"),
            }
        )

        node = TextNode(
            id_=chunk.get("id"),
            text=text,
            metadata=metadata,
        )
        nodes.append(node)

    return nodes


def configure_embedding(
    provider: str,
    model: str,
    mock_dim: int,
    dimensions: Optional[int],
    openai_api_base: Optional[str],
    openai_api_key: Optional[str],
    embedding_api_url: Optional[str],
    embedding_api_key: Optional[str],
    embed_batch_size: int,
    request_timeout: float,
    max_retries: int,
):
    if provider == "mock":
        Settings.embed_model = MockEmbedding(embed_dim=mock_dim)
        return {
            "provider": "mock",
            "model": f"MockEmbedding(dim={mock_dim})",
            "note": "For pipeline testing only; not suitable for real semantic retrieval.",
        }

    if provider == "openai":
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        api_base = (
            openai_api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_api_key'"
            )

        from llama_index.embeddings.openai import OpenAIEmbedding

        kwargs = {
            "model": model,
            "api_key": api_key,
            "embed_batch_size": embed_batch_size,
            "timeout": request_timeout,
            "max_retries": max_retries,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if dimensions:
            kwargs["dimensions"] = dimensions
        Settings.embed_model = OpenAIEmbedding(**kwargs)
        return {
            "provider": "openai",
            "model": model,
            "dimensions": dimensions,
            "api_base": api_base,
            "embed_batch_size": embed_batch_size,
            "timeout": request_timeout,
            "max_retries": max_retries,
        }

    if provider == "http":
        from http_embedding import make_http_embedding

        Settings.embed_model = make_http_embedding(
            api_url=embedding_api_url,
            model=model,
            embed_batch_size=embed_batch_size,
            timeout=request_timeout,
            api_key=embedding_api_key,
        )
        return {
            "provider": "http",
            "model": model,
            "api_url": embedding_api_url or os.environ.get("EMBEDDING_URL") or os.environ.get("BGE_EMBEDDING_URL"),
            "dimensions": 1024,
            "embed_batch_size": embed_batch_size,
            "timeout": request_timeout,
        }

    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_index(
    input_path: Path,
    persist_dir: Path,
    manifest_path: Path,
    provider: str,
    model: str,
    mock_dim: int,
    dimensions: Optional[int],
    openai_api_base: Optional[str],
    openai_api_key: Optional[str],
    embedding_api_url: Optional[str],
    embedding_api_key: Optional[str],
    embed_batch_size: int,
    request_timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    print(f"[INFO] Loading chunks: {input_path}", flush=True)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    nodes = chunks_to_nodes(chunks)
    print(f"[INFO] Prepared {len(nodes)} TextNode objects from {len(chunks)} chunks.", flush=True)
    print(f"[INFO] Configuring embedding provider: {provider}, model: {model}", flush=True)
    embedding_info = configure_embedding(
        provider,
        model,
        mock_dim,
        dimensions,
        openai_api_base,
        openai_api_key,
        embedding_api_url,
        embedding_api_key,
        embed_batch_size,
        request_timeout,
        max_retries,
    )

    print("[INFO] Building VectorStoreIndex. This step calls the embedding API.", flush=True)
    index = VectorStoreIndex(nodes)
    print(f"[INFO] Persisting index: {persist_dir}", flush=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

    manifest = {
        "source_chunks": str(input_path),
        "persist_dir": str(persist_dir),
        "embedding": embedding_info,
        "node_count": len(nodes),
        "chunk_count": len(chunks),
        "chapter": data.get("chapter"),
        "source_file": data.get("source_file"),
        "index_type": "llama_index.core.VectorStoreIndex",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def test_retrieve(persist_dir: Path, query: str, top_k: int):
    print(f"[INFO] Loading persisted index for test query: {persist_dir}", flush=True)
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=top_k)
    print("[INFO] Running test retrieval. This step embeds the query.", flush=True)
    return retriever.retrieve(query)


def main():
    parser = argparse.ArgumentParser(
        description="Build a LlamaIndex vector index from SKF RAG chunks."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input chunks JSON.")
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="Directory for persisted LlamaIndex vector index.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest JSON path.")
    parser.add_argument(
        "--embedding-provider",
        default="mock",
        choices=["mock", "openai", "http"],
        help="Embedding provider. Use mock for local pipeline testing.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model name for non-mock providers.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Optional embedding dimensions for OpenAI text-embedding-3 models.",
    )
    parser.add_argument(
        "--openai-api-base",
        default=None,
        help="Optional OpenAI-compatible base URL, e.g. https://your-proxy.example.com/v1. Defaults to OPENAI_BASE_URL or OPENAI_API_BASE.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional API key. Prefer OPENAI_API_KEY env var instead of passing secrets on the command line.",
    )
    parser.add_argument(
        "--embedding-api-url",
        default=None,
        help="HTTP embedding service URL, e.g. http://your-embedding-service:8001. Defaults to EMBEDDING_URL.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="Optional bearer token for HTTP embedding service. Defaults to EMBEDDING_API_KEY.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Embedding batch size. Use a smaller value for slow or unstable proxy services.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="OpenAI-compatible embedding request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="OpenAI-compatible embedding request retry count.",
    )
    parser.add_argument("--mock-dim", type=int, default=384, help="Mock embedding dimension.")
    parser.add_argument(
        "--test-query",
        default="What are radial bearings?",
        help="Optional query used to test retrieval after index creation.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k for test query.")
    args = parser.parse_args()

    manifest = build_index(
        input_path=Path(args.input),
        persist_dir=Path(args.persist_dir),
        manifest_path=Path(args.manifest),
        provider=args.embedding_provider,
        model=args.embedding_model,
        mock_dim=args.mock_dim,
        dimensions=args.embedding_dimensions,
        openai_api_base=args.openai_api_base,
        openai_api_key=args.openai_api_key,
        embedding_api_url=args.embedding_api_url,
        embedding_api_key=args.embedding_api_key,
        embed_batch_size=args.embed_batch_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    print(
        f"[OK] Built {manifest['index_type']} with {manifest['node_count']} nodes "
        f"using {manifest['embedding']['provider']} embeddings -> {args.persist_dir}"
    )
    print(f"[OK] Manifest -> {args.manifest}")

    if args.test_query:
        results = test_retrieve(Path(args.persist_dir), args.test_query, args.top_k)
        print(f"[TEST] Query: {args.test_query}")
        for rank, result in enumerate(results, start=1):
            node = result.node
            metadata = node.metadata or {}
            text = node.get_content(metadata_mode="none").replace("\n", " ")
            print(
                f"[TEST] #{rank} score={result.score} "
                f"chunk_id={metadata.get('chunk_id')} "
                f"section={metadata.get('section')} "
                f"pages={metadata.get('page_start')}-{metadata.get('page_end')}"
            )
            print(f"[TEST]    {text[:220]}")


if __name__ == "__main__":
    main()
