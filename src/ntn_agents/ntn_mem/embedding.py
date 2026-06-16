"""Embedding provider adapters for MEM search.

This module keeps provider concerns out of the WSGI app. The default `hash`
provider is deterministic and local, suitable for smoke/CI. Operators can use it
as a fallback while wiring a real model/vector service behind the same boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


@dataclass
class HashEmbeddingProvider:
    name: str = "hash"
    dimensions: int = 64

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[idx] += sign
        return vector


@dataclass
class FailingEmbeddingProvider:
    name: str = "failing"

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("configured failing embedding provider")


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible `/embeddings` provider.

    SiliconFlow exposes an OpenAI-compatible API, so the same adapter is used for
    `openai`, `openai-compatible`, and `siliconflow` provider names. Secrets are
    read only from environment variables and are never logged or persisted here.
    """

    name: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-ada-002"
    api_key: str | None = None
    timeout: float = 15.0

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("NTN_MEM_EMBEDDING_API_KEY is required for openai-compatible embedding provider")
        endpoint = f"{self.base_url.rstrip('/')}/embeddings"
        payload = json.dumps({"model": self.model, "input": text or ""}).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        embedding = _extract_openai_embedding(body)
        return [float(value) for value in embedding]


def _extract_openai_embedding(body: dict[str, Any]) -> list[int | float]:
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("openai-compatible embedding response missing data[0].embedding")
    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError("openai-compatible embedding response data[0] must be an object")
    embedding = first.get("embedding")
    if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
        raise RuntimeError("openai-compatible embedding response missing numeric embedding vector")
    return embedding


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _tokens(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text or "")
    return [token for token in normalized.split() if token]


def get_embedding_provider() -> EmbeddingProvider:
    provider = os.environ.get("NTN_MEM_EMBEDDING_PROVIDER", "hash").lower()
    if provider in {"hash", "local", "local-hash"}:
        return HashEmbeddingProvider(name="hash")
    if provider in {"openai", "openai-compatible", "siliconflow"}:
        return OpenAICompatibleEmbeddingProvider(
            name="openai",
            base_url=os.environ.get("NTN_MEM_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("NTN_MEM_EMBEDDING_MODEL", "text-embedding-ada-002"),
            api_key=os.environ.get("NTN_MEM_EMBEDDING_API_KEY"),
            timeout=_float_env("NTN_MEM_EMBEDDING_TIMEOUT", 15.0),
        )
    if provider == "failing":
        return FailingEmbeddingProvider()
    # Unknown providers degrade through the same failure path as a broken remote provider.
    return FailingEmbeddingProvider(name=provider)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
