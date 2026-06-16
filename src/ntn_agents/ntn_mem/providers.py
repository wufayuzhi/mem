"""Provider adapters for MEM writes.

All external memory-provider calls stay inside MEM. Other containers should call
MEM APIs only; they must not talk to Mem0/vector stores directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen


class ProviderClient(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]: ...


class UrllibProviderClient:
    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is operator-configured provider endpoint.
            return json.loads(response.read().decode("utf-8") or "{}")


def configured_provider() -> str:
    provider = os.environ.get("NTN_MEM_PROVIDER", "local").lower()
    return provider if provider in {"local", "mem0"} else "local"


def mem0_user_id(project_id: str | None, role: str | None, scope: str | None) -> str:
    return f"ntn:{project_id or 'global'}:{role or 'unknown'}:{scope or 'default'}"


@dataclass
class ProviderWriteResult:
    status: str
    provider: str
    provider_memory_id: str | None
    provider_event_id: str | None
    error: dict[str, Any] | None = None


def write_provider(data: dict[str, Any], *, memory_id: str, client: ProviderClient | None = None) -> ProviderWriteResult:
    provider = configured_provider()
    if provider == "local":
        provider_event_id = data.get("provider_event_id") or memory_id
        provider_memory_id = data.get("provider_memory_id") or memory_id
        return ProviderWriteResult("SUCCEEDED", "local", provider_memory_id, provider_event_id)
    if provider == "mem0":
        endpoint = os.environ.get("NTN_MEM0_URL")
        if not endpoint:
            return ProviderWriteResult("PENDING_PROVIDER", "mem0", None, None, {"code": "PROVIDER_CONFIG_MISSING", "message": "NTN_MEM0_URL is required"})
        provider_client = client or data.get("provider_client") or UrllibProviderClient()
        token = os.environ.get("NTN_MEM0_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        payload = {
            "memory_id": memory_id,
            "text": data["text"],
            "user_id": mem0_user_id(data.get("project_id"), data.get("owner_role") or data.get("role"), data.get("scope")),
            "metadata": data.get("metadata") or {},
        }
        try:
            response = provider_client.post_json(endpoint.rstrip("/") + "/v1/memories", payload, headers=headers, timeout=float(os.environ.get("NTN_MEM0_TIMEOUT", "5")))
        except Exception as exc:  # provider failures must not lose local memory
            return ProviderWriteResult("PENDING_PROVIDER", "mem0", None, None, {"code": "PROVIDER_WRITE_FAILED", "message": str(exc)})
        return ProviderWriteResult("SUCCEEDED", "mem0", response.get("id"), response.get("event_id") or response.get("id"))
    return ProviderWriteResult("SUCCEEDED", "local", memory_id, memory_id)
