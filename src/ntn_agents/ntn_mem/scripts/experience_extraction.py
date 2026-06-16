"""Nightly experience extraction for NTN MEM.

Runs as a systemd oneshot triggered by a daily timer (02:30).  Scans the
previous day's per-agent private memories, feeds them to the LLM for structured
experience extraction, and writes the result to the 'experience_reserve' project.

Architecture (no changes to the existing three-layer memory system):
  1. SQL query yesterday's private-* project memories
  2. LLM (SiliconFlow DeepSeek-R1-0528) → structured summary + insights
  3. Write result to MEM as shared knowledge in 'experience_reserve' project

The extraction is shared-intel — any agent that pulls with
shared_knowledge_projects=["experience_reserve"] can see it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.request import Request, urlopen


# ── Config (overridable via env vars) ──
MEM_URL = os.environ.get("NTN_MEM_URL", "http://localhost:8081").rstrip("/")
QDRANT_URL = os.environ.get("NTN_QDRANT_URL", "http://10.69.68.15:6333")
MEM_DB = os.environ.get("NTN_MEM_DB", "/data/mem.db")
LLM_MODEL = os.environ.get("EXTRACTION_MODEL", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true")

# How many days back to scan
LOOKBACK_DAYS = int(os.environ.get("EXTRACTION_LOOKBACK_DAYS", "1"))
# Max memories to feed to the LLM per extraction
MAX_MEMORIES = int(os.environ.get("EXTRACTION_MAX_MEMORIES", "200"))
# Max input text chars for LLM
MAX_INPUT_CHARS = int(os.environ.get("EXTRACTION_MAX_CHARS", "12000"))


# ── Helpers ──

def _get_llm_api_key() -> str | None:
    """Read SiliconFlow API key from env or secrets file."""
    key = os.environ.get("NTN_MEM_EMBEDDING_API_KEY")
    if key and "***" not in key:
        return key
    try:
        with open("/etc/ntn-agents/secrets.env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("NTN_MEM_EMBEDDING_API_KEY="):
                    val = line.split("=", 1)[1].strip("\"' \t\n\r")
                    if val and "***" not in val:
                        return val
    except OSError:
        pass
    return None


def _llm_chat(messages: list[dict], system: str | None = None, max_tokens: int = 4096) -> str:
    """Call SiliconFlow chat/completions."""
    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError("SiliconFlow API key not found")

    base_url = os.environ.get("NTN_MEM_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    full_messages = list(messages)
    if system:
        full_messages.insert(0, {"role": "system", "content": system})

    payload = {
        "model": LLM_MODEL,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty LLM response: {json.dumps(body, ensure_ascii=False)[:300]}")

    return (choices[0].get("message", {}).get("content") or "").strip()


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output (handles ``` fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _mem_request(method: str, path: str, body: dict | None = None, timeout: float = 15) -> dict:
    url = f"{MEM_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_memories_for_date(target_date: date) -> list[dict]:
    """Fetch all memories from private-* projects created on target_date."""
    import sqlite3

    date_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    date_end = date_start + timedelta(hours=23, minutes=59, seconds=59)

    conn = sqlite3.connect(MEM_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT memory_id, text, project_id, scope, memory_type,
                      source, created_at, temperature
               FROM memories
               WHERE project_id LIKE 'private-%'
                 AND COALESCE(deleted, 0) = 0
                 AND temperature != 'decayed'
                 AND created_at >= ? AND created_at <= ?
               ORDER BY created_at ASC""",
            (date_start.isoformat(), date_end.isoformat()),
        ).fetchall()

        return [
            {
                "memory_id": r["memory_id"],
                "text": r["text"],
                "project_id": r["project_id"],
                "scope": r["scope"],
                "memory_type": r["memory_type"],
                "source": r["source"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def _format_for_llm(memories: list[dict]) -> str:
    """Format memories into an LLM-friendly text block."""
    lines = []
    for i, mem in enumerate(memories, 1):
        text = (mem.get("text") or "").strip()[:500]
        when = (mem.get("created_at") or "")[:19]
        src = mem.get("source", "unknown")[:20]
        proj = (mem.get("project_id") or "?").replace("private-", "", 1)[:20]
        lines.append(f"[{i}] ({when}) [{proj}/{src}] {text}")
    return "\n".join(lines)


# ── Main pipeline ──

def run_extraction() -> dict:
    """Run one extraction cycle and return results."""
    target_date = date.today() - timedelta(days=LOOKBACK_DAYS)
    extraction_date = target_date.isoformat()

    print(f"=== Experience Extraction — {extraction_date} ===")

    # 1. Fetch memories
    print(f"  Fetching private memories from {extraction_date}...")
    memories = _get_memories_for_date(target_date)
    print(f"  Found {len(memories)} memories")

    if not memories:
        return {"date": extraction_date, "source_count": 0, "status": "no_data"}

    # 2. Truncate if too many
    if len(memories) > MAX_MEMORIES:
        memories = memories[-MAX_MEMORIES:]
        print(f"  Truncated to {MAX_MEMORIES} most recent memories")

    memories_text = _format_for_llm(memories)
    if len(memories_text) > MAX_INPUT_CHARS:
        memories_text = "\n".join(memories_text.splitlines()[-200:])
        if len(memories_text) > MAX_INPUT_CHARS:
            memories_text = memories_text[-MAX_INPUT_CHARS:]
        print(f"  Truncated text to ~{len(memories_text)} chars")

    # 3. LLM extraction
    print(f"  Calling LLM ({LLM_MODEL})...")

    system_prompt = """你是 NTN MEM 体验提炼助手。从用户的每日对话记录中提取结构化知识。

输出必须是纯 JSON（不要 markdown 代码块）：

{
  "summary": "2-4句中文总结这一天的主要活动和知识点",
  "key_insights": [
    {"topic": "话题", "detail": "具体知识", "importance": "high|medium|low"}
  ],
  "decisions": ["做出的决策"],
  "action_items": [{"action": "待办", "owner": "负责人", "priority": "high|medium|low"}],
  "tags": ["#标签1", "#标签2"]
}

原则：只从提供内容中提炼，不发明信息。如果某种数据不存在返回空数组。"""

    try:
        raw_response = _llm_chat(
            [{"role": "user", "content": f"今日记录：\n{memories_text}"}],
            system=system_prompt,
        )
        result = _extract_json(raw_response)
    except Exception as e:
        print(f"  LLM error: {e}")
        result = {
            "summary": f"LLM extraction failed: {e}",
            "key_insights": [],
            "decisions": [],
            "action_items": [],
            "tags": [],
        }

    # 4. Auto-register experience_reserve in kb_registry if not already
    try:
        import sqlite3 as _sqlite3
        reg_path = os.environ.get("NTN_MEM_REGISTRY_DB", "/data/registry.db")
        reg_conn = _sqlite3.connect(reg_path, timeout=5.0)
        reg_conn.executescript(
            "CREATE TABLE IF NOT EXISTS kb_registry ("
            "  kb_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,"
            "  project_id TEXT NOT NULL, tags_json TEXT DEFAULT '[]', owner TEXT,"
            "  created_at TEXT NOT NULL, updated_at TEXT, status TEXT DEFAULT 'active'"
            ")"
        )
        existing = reg_conn.execute(
            "SELECT kb_id FROM kb_registry WHERE project_id='experience_reserve' AND status='active'"
        ).fetchone()
        if existing is None:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            tag = json.dumps(["auto_extraction", "daily"], ensure_ascii=False)
            reg_conn.execute(
                "INSERT OR REPLACE INTO kb_registry "
                "(kb_id, name, description, project_id, tags_json, owner, created_at, updated_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')",
                ("experience-reserve", "经验储备库",
                 "LLM 每天 02:30 自动提炼的经验总结，所有 agent 共享读取",
                 "experience_reserve", tag, "system", now, now),
            )
            print("  Registered: experience-reserve in kb_registry")
        else:
            # Update timestamp
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            reg_conn.execute(
                "UPDATE kb_registry SET updated_at=? WHERE project_id='experience_reserve'",
                (now,),
            )
        reg_conn.close()
    except Exception as e:
        print(f"  (kb_registry update skipped: {e})")

    # 5. Write to MEM as shared knowledge
    extraction_payload = {
        "text": json.dumps({
            "type": "daily_reserve",
            "date": extraction_date,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "model": LLM_MODEL,
            "source_count": len(memories),
            **result,
        }, ensure_ascii=False),
        "project_id": "experience_reserve",
        "scope": "shared",
        "memory_type": "knowledge_doc",
        "source": "experience_extraction",
        "metadata": {
            "extraction_type": "daily",
            "extraction_date": extraction_date,
            "source_count": len(memories),
            "insight_count": len(result.get("key_insights", [])),
            "decision_count": len(result.get("decisions", [])),
            "tag_count": len(result.get("tags", [])),
        },
    }

    if DRY_RUN:
        print(f"\n  === DRY RUN: would write ===")
        print(f"  Summary: {result.get('summary', '')[:200]}")
        print(f"  Insights: {len(result.get('key_insights', []))}")
        print(f"  Decisions: {len(result.get('decisions', []))}")
        mem_id = "(dry-run)"
    else:
        try:
            write_result = _mem_request("POST", "/v1/memory/add", extraction_payload, timeout=15)
            mem_id = write_result.get("memory_id") or write_result.get("provider_event_id") or write_result.get("id", "(unknown)")
            print(f"  Written to MEM: {mem_id}")
        except Exception as e:
            print(f"  MEM write error: {e}")
            mem_id = f"(failed: {e})"

    return {
        "date": extraction_date,
        "source_count": len(memories),
        "mem_id": mem_id,
        "summary": result.get("summary", "")[:300],
        "insight_count": len(result.get("key_insights", [])),
        "decision_count": len(result.get("decisions", [])),
        "action_count": len(result.get("action_items", [])),
        "status": "completed",
    }


def main() -> int:
    start_ts = time.time()
    try:
        result = run_extraction()
        elapsed = time.time() - start_ts
        print(f"\n=== Done ({elapsed:.1f}s) ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("completed", "no_data") else 1
    except Exception as e:
        print(f"\nFATAL: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
