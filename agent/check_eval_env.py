#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from urllib.parse import urlparse


REQUIRED_VARS = [
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "EMBEDDING_URL",
    "RERANKER_URL",
]


URL_VARS = {
    "OLLAMA_BASE_URL",
    "OPENAI_API_BASE",
    "EMBEDDING_URL",
    "RERANKER_URL",
}


SECRET_VARS = {
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
}


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def validate_url(name: str, value: str) -> list[str]:
    warnings = []
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        warnings.append("not a valid http(s) URL")
    if name == "OLLAMA_BASE_URL" and value.rstrip("/").endswith("/v1"):
        warnings.append("OLLAMA_BASE_URL should usually omit /v1; scripts append /v1 automatically")
    return warnings


def main() -> int:
    missing = []
    print("Environment check for DT + RAG evaluation\n")
    for name in REQUIRED_VARS:
        value = os.environ.get(name, "")
        if not value:
            missing.append(name)
            print(f"[MISSING] {name}")
            continue

        notes = []
        display = "<set>"
        if name in SECRET_VARS:
            display = mask_secret(value)
        elif name in URL_VARS:
            display = value
            notes.extend(validate_url(name, value))

        suffix = f" ({'; '.join(notes)})" if notes else ""
        print(f"[OK]      {name}={display}{suffix}")

    if missing:
        print("\nMissing variables:")
        for name in missing:
            print(f"  export {name}=...")
        return 1

    print("\nAll required variables are set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
