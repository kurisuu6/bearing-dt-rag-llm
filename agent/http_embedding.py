from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import Field


DEFAULT_EMBEDDING_URL = os.environ.get("EMBEDDING_URL") or os.environ.get("BGE_EMBEDDING_URL")


class HTTPEmbedding(BaseEmbedding):
    """LlamaIndex embedding wrapper for the local HTTP /embed service."""

    api_url: str = Field(description="Base URL of the embedding service, e.g. http://host:8001")
    api_key: Optional[str] = Field(default=None, description="Optional bearer token for the embedding service.")
    timeout: float = Field(default=60.0, description="HTTP request timeout in seconds.")

    def _post_embed(self, texts: list[str]) -> list[Embedding]:
        try:
            import requests
        except ModuleNotFoundError as exc:
            raise RuntimeError("The requests package is required for HTTP embedding support. Install it with: pip install requests") from exc

        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = self.api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("BGE_EMBEDDING_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload = {"texts": texts, "batch_size": self.embed_batch_size}
        response = requests.post(f"{self.api_url.rstrip('/')}/embed", json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Embedding service returned invalid payload: {data}")
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding service returned {len(embeddings)} embeddings for {len(texts)} texts.")
        return embeddings

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._post_embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._post_embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._post_embed(texts)


def make_http_embedding(
    api_url: Optional[str],
    model: str = "bge-m3",
    embed_batch_size: int = 16,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> HTTPEmbedding:
    url = api_url or DEFAULT_EMBEDDING_URL
    if not url:
        raise RuntimeError("EMBEDDING_URL is not set. Example: export EMBEDDING_URL=http://your-embedding-service:8001")
    return HTTPEmbedding(
        model_name=model,
        api_url=url,
        api_key=api_key,
        embed_batch_size=embed_batch_size,
        timeout=timeout,
    )


class HTTPBGEEmbeddings(Embeddings):
    """LangChain-compatible embedding wrapper for RAGAS."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        self.api_url = api_url or DEFAULT_EMBEDDING_URL
        if not self.api_url:
            raise RuntimeError("EMBEDDING_URL is not set. Example: export EMBEDDING_URL=http://your-embedding-service:8001")
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import requests
        except ModuleNotFoundError as exc:
            raise RuntimeError("The requests package is required for HTTP embedding support. Install it with: pip install requests") from exc

        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = self.api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("BGE_EMBEDDING_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"texts": batch, "batch_size": self.batch_size}
            response = requests.post(f"{self.api_url.rstrip('/')}/embed", json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list):
                raise RuntimeError(f"Embedding service returned invalid payload: {data}")
            if len(embeddings) != len(batch):
                raise RuntimeError(f"Embedding service returned {len(embeddings)} embeddings for {len(batch)} texts.")
            output.extend(embeddings)
        return output

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._post_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._post_embed([text])[0]


def make_langchain_http_embeddings(
    api_url: Optional[str],
    api_key: Optional[str] = None,
    batch_size: int = 16,
    timeout: float = 60.0,
) -> HTTPBGEEmbeddings:
    return HTTPBGEEmbeddings(api_url=api_url, api_key=api_key, batch_size=batch_size, timeout=timeout)
