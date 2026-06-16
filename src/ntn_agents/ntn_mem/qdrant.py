"""Qdrant vector-store adapter for NTN MEM.

The adapter is intentionally stdlib-only so MEM can keep running in lean
containers. SQLite remains the authoritative store; Qdrant is a best-effort
secondary vector index used for recall acceleration/ranking.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class QdrantSearchResult:
    memory_id: str
    score: float


@dataclass
class QdrantClient:
    url: str
    collection: str = "ntn_memories"
    timeout: float = 5.0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            f"{self.url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"qdrant request failed: {exc}") from exc
        return json.loads(raw or "{}")

    def ensure_collection(self, dimensions: int) -> None:
        try:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": dimensions, "distance": "Cosine"}},
            )
            return
        except RuntimeError as exc:
            # Qdrant returns HTTP 409 when the collection already exists.  That
            # is OK only if the existing vector size matches the embedding size.
            if "HTTP Error 409" not in str(exc):
                raise
        body = self._request("GET", f"/collections/{self.collection}")
        vectors = (((body.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors") or {}
        size = vectors.get("size") if isinstance(vectors, dict) else None
        if int(size or 0) != dimensions:
            raise RuntimeError(f"qdrant collection dimension mismatch: existing={size} expected={dimensions}")

    def upsert(self, memory_id: str, vector: list[float] | list[int], payload: dict[str, Any]) -> None:
        vector_f = [float(v) for v in vector]
        self.ensure_collection(len(vector_f))
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {
                "points": [
                    {
                        "id": _point_id(memory_id),
                        "vector": vector_f,
                        "payload": {**payload, "memory_id": memory_id},
                    }
                ]
            },
        )

    def search(self, vector: list[float] | list[int], limit: int, filters: dict[str, Any] | None = None) -> list[QdrantSearchResult]:
        vector_f = [float(v) for v in vector]
        must = []
        for key, value in (filters or {}).items():
            if value is None:
                continue
            must.append({"key": key, "match": {"value": value}})
        payload: dict[str, Any] = {"vector": vector_f, "limit": limit, "with_payload": True}
        if must:
            payload["filter"] = {"must": must}
        body = self._request("POST", f"/collections/{self.collection}/points/search", payload)
        results: list[QdrantSearchResult] = []
        for item in body.get("result") or []:
            point_payload = item.get("payload") or {}
            memory_id = point_payload.get("memory_id")
            if memory_id:
                results.append(QdrantSearchResult(memory_id=str(memory_id), score=float(item.get("score") or 0.0)))
        return results


def _point_id(memory_id: str) -> str:
    # Qdrant point IDs support UUID strings. MEM IDs are usually "mem-<uuid>".
    if memory_id.startswith("mem-"):
        candidate = memory_id[4:]
        if len(candidate) == 36:
            return candidate
    return memory_id


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_qdrant_client() -> QdrantClient | None:
    url = os.environ.get("NTN_QDRANT_URL")
    if not url:
        return None
    return QdrantClient(
        url=url,
        collection=os.environ.get("NTN_QDRANT_COLLECTION", "ntn_memories"),
        timeout=_float_env("NTN_QDRANT_TIMEOUT", 5.0),
    )
