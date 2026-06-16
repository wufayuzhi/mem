"""SQL-http /events polling bridge for MEM.

The poller is intentionally HTTP-based: MEM consumes SQL-http events through the
public `/events` API and never reaches into SQL-http's SQLite state directly.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .app import consume_event

DEFAULT_CURSOR_PATH = "/data/mem-event-cursor.json"


class HTTPClient(Protocol):
    def get_json(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]: ...


class UrllibHTTPClient:
    def get_json(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        request = Request(url, headers=headers or {}, method="GET")
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is operator-configured service endpoint.
            return json.loads(response.read().decode("utf-8"))


def _read_cursor(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("after_id") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _write_cursor(path: str, after_id: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"after_id": int(after_id)}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _event_text(event: dict[str, Any]) -> str:
    payload = event.get("payload") or {}
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        return result.get("summary") or result.get("message") or ""
    return payload.get("summary") or payload.get("message") or "" if isinstance(payload, dict) else ""


def _event_project_id(event: dict[str, Any], default_project_id: str | None) -> str | None:
    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        return payload.get("project_id") or payload.get("project") or default_project_id
    return default_project_id


def _normalize_event(event: dict[str, Any], default_project_id: str | None) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "task_id": event.get("task_id"),
        "project_id": _event_project_id(event, default_project_id),
        "role": event.get("target_role") or event.get("role"),
        "target_role": event.get("target_role") or event.get("role"),
        "text": _event_text(event),
        "payload": payload,
    }


@dataclass
class EventPoller:
    sql_http_url: str
    cursor_path: str = DEFAULT_CURSOR_PATH
    token: str | None = None
    http_client: HTTPClient | None = None
    default_project_id: str | None = None
    limit: int = 100
    timeout: float = 5.0

    def poll_once(self) -> dict[str, int]:
        client = self.http_client or UrllibHTTPClient()
        after_id = _read_cursor(self.cursor_path)
        base = self.sql_http_url.rstrip("/")
        url = f"{base}/events?{urlencode({'after_id': after_id, 'limit': self.limit})}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        payload = client.get_json(url, headers=headers, timeout=self.timeout)
        events = payload.get("events") or []
        accepted = 0
        failed = 0
        max_seen = after_id
        for event in events:
            event_id = int(event.get("event_id") or max_seen)
            max_seen = max(max_seen, event_id)
            consumed = consume_event(_normalize_event(event, self.default_project_id))
            if consumed.get("accepted"):
                accepted += 1
            else:
                failed += 1
        next_after_id = int(payload.get("next_after_id") or max_seen)
        _write_cursor(self.cursor_path, next_after_id)
        return {"fetched": len(events), "accepted": accepted, "failed": failed, "after_id": next_after_id}

    def run_forever(self, interval_seconds: float = 2.0) -> None:
        while True:
            self.poll_once()
            time.sleep(interval_seconds)


def poll_once(
    *,
    sql_http_url: str,
    cursor_path: str = DEFAULT_CURSOR_PATH,
    token: str | None = None,
    http_client: HTTPClient | None = None,
    default_project_id: str | None = None,
    limit: int = 100,
    timeout: float = 5.0,
) -> dict[str, int]:
    return EventPoller(
        sql_http_url=sql_http_url,
        cursor_path=cursor_path,
        token=token,
        http_client=http_client,
        default_project_id=default_project_id,
        limit=limit,
        timeout=timeout,
    ).poll_once()


def main() -> None:
    poller = EventPoller(
        sql_http_url=os.environ["NTN_SQL_HTTP_URL"],
        cursor_path=os.environ.get("NTN_MEM_EVENT_CURSOR", DEFAULT_CURSOR_PATH),
        token=os.environ.get("NTN_SQL_HTTP_TOKEN"),
        default_project_id=os.environ.get("NTN_PROJECT_ID"),
        limit=int(os.environ.get("NTN_MEM_EVENT_LIMIT", "100")),
        timeout=float(os.environ.get("NTN_MEM_EVENT_TIMEOUT", "5")),
    )
    interval = float(os.environ.get("NTN_MEM_EVENT_INTERVAL", "2"))
    poller.run_forever(interval_seconds=interval)


if __name__ == "__main__":
    main()
