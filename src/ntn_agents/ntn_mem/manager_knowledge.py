"""Knowledge Base Manager — KB registration, cross-KB search, document management.

Every KB maps to an existing MEM ``project_id``. Registering a KB does NOT
duplicate or move data; it simply adds metadata that enables cross-KB search
and centralized document management. Storage remains in the existing ``memories``
table, and all mutations delegate to the existing ``add_memory()`` /
``search_memory()`` / ``delete_memory()`` API.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

REGISTRY_DB = os.environ.get("NTN_MEM_REGISTRY_DB", "/data/registry.db")

_KB_SCHEMA_PROPS = {
    "mandatory_fields": {
        "project_id": {"type": "str", "fixed": True, "description": "知识库 dedicated project_id，写入时不可变更"},
        "scope": {"type": "str", "fixed": "shared", "description": "共享库，所有 Agent 只读"},
        "memory_type": {"type": "str", "fixed": "knowledge_doc", "description": "知识文档类型"},
        "status": {"type": "str", "fixed": "verified", "description": "新写入默认为 verified，旧版本自动 supersede"},
        "text": {"type": "str", "required": True, "description": "文档正文内容"},
        "text_hash": {"type": "str", "auto": "SHA256(text)", "description": "自动计算，用于去重"},
        "source": {"type": "str", "required": True, "description": "来源名称，如 ccb_docs_ingest"},
    },
    "optional_metadata": {
        "layer": {"type": "str", "default": "mid_term", "description": "生命周期层：short_term/mid_term/long_term"},
        "temperature": {"type": "str", "default": "warm", "description": "热度：hot/warm/cold/protected"},
    },
    "writes": {
        "url": "POST /v1/memory/add",
        "body_example": {
            "text": "文档内容（正文含来源URL和本地路径）",
            "project_id": "knowledge_reserve",
            "scope": "shared",
            "memory_type": "knowledge_doc",
            "source": "ccb_docs_ingest",
            "actor_role": "admin",
        },
        "dedup": "text_hash 自动去重，同哈希不重复写入",
        "versioning": "新版本写入后旧版本（同 project_id + text_hash）自动 supersede",
    },
    "reads": {
        "url": "POST /v1/memory/search",
        "body_example": {"query": "搜索关键词", "project_id": "knowledge_reserve", "limit": 5},
        "cross_kb_search": "POST /v1/knowledge/search",
    },
    "lifecycle": {
        "refresh_policy": "手动触发，建议全量重建后用 ingest 增量",
        "gc_policy": "superseded >7天可清理",
        "cold_archive": "POST /v1/memory/archive/cold",
        "reindex": "POST /v1/knowledge/{kb_id}/reindex",
    },
}


def _reg_connect() -> sqlite3.Connection:
    target = REGISTRY_DB
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kb_registry (
            kb_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            project_id TEXT NOT NULL,
            tags_json TEXT DEFAULT '[]',
            owner TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            status TEXT DEFAULT 'active',
            source_definitions_json TEXT DEFAULT '[]',
            chunk_strategy TEXT DEFAULT 'by_heading',
            refresh_policy TEXT DEFAULT 'manual',
            version TEXT
        );
        """
    )
    return conn


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Lazy import to avoid circular deps at module level.

DEFAULT_MEM_DB = "/data/mem.db"


def _db_path() -> str:
    return os.environ.get("NTN_MEM_DB", DEFAULT_MEM_DB)


def _mem_count(project_id: str) -> int:
    db_path = _db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=3.0)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
            (project_id,),
        ).fetchone()
        conn.close()
        return row["n"] if row else 0
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def _generate_kb_id(name: str) -> str:
    """Derive a stable kb_id from the KB name."""
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    return f"kb-{safe.strip('-')}"[:64]


def register_kb(
    name: str,
    *,
    description: str = "",
    project_id: str,
    tags: list[str] | None = None,
    owner: str | None = None,
    source_definitions: list[dict] | None = None,
    chunk_strategy: str = "by_heading",
    refresh_policy: str = "manual",
    version: str | None = None,
) -> dict[str, Any]:
    """Register a new knowledge base.

    Args:
        name: Human-friendly KB name (e.g. "Hermes 中文知识库").
        description: Free-text description.
        project_id: Existing MEM project_id holding the documents.
        tags: Optional tag list for discovery.
        owner: Agent/role key that owns this KB.
        source_definitions: List of source definitions with url/name/path.
        chunk_strategy: Chunking method (by_heading, by_paragraph, by_fixed).
        refresh_policy: When to refresh (manual, daily, on_change).
        version: KB version tag for lineage tracking.

    Returns:
        dict with ``kb_id`` and registration metadata.

    Raises:
        ValueError if project_id is already registered under a different KB.
    """
    conn = _reg_connect()
    try:
        # Check project_id not already registered.
        existing = conn.execute(
            "SELECT kb_id, name FROM kb_registry WHERE project_id=? AND status='active'",
            (project_id,),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"project_id '{project_id}' already registered as "
                f"kb_id='{existing['kb_id']}' name='{existing['name']}'"
            )

        kb_id = _generate_kb_id(name)
        now = _now_iso()
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        sources_json = json.dumps(source_definitions or [], ensure_ascii=False)
        conn.execute(
            "INSERT OR REPLACE INTO kb_registry "
            "(kb_id, name, description, project_id, tags_json, owner, created_at, updated_at, status, "
            "source_definitions_json, chunk_strategy, refresh_policy, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)",
            (kb_id, name, description, project_id, tags_json, owner, now, now,
             sources_json, chunk_strategy, refresh_policy, version),
        )
        result = get_kb(kb_id)
        assert result is not None
        return result
    finally:
        conn.close()


def get_kb(kb_id: str) -> dict[str, Any] | None:
    """Return a single KB by id, or None."""
    conn = _reg_connect()
    try:
        row = conn.execute(
            "SELECT * FROM kb_registry WHERE kb_id=?", (kb_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_kb(row)
    finally:
        conn.close()


def _row_to_kb(row: sqlite3.Row) -> dict[str, Any]:
    tags = json.loads(row["tags_json"] or "[]")
    # Use .get() for backward compat — old rows may lack extended columns
    try:
        sources = json.loads(row["source_definitions_json"] or "[]") if "source_definitions_json" in row.keys() else []
    except Exception:
        sources = []
    return {
        "kb_id": row["kb_id"],
        "name": row["name"],
        "description": row["description"],
        "project_id": row["project_id"],
        "tags": tags,
        "owner": row["owner"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "source_definitions": sources,
        "chunk_strategy": row["chunk_strategy"] if "chunk_strategy" in row.keys() else "by_heading",
        "refresh_policy": row["refresh_policy"] if "refresh_policy" in row.keys() else "manual",
        "version": row["version"] if "version" in row.keys() else None,
        "document_count": _mem_count(row["project_id"]),
    }


def get_kb_schema() -> dict[str, Any]:
    """Return the universal knowledge base schema for all registered KBs.

    Any Agent can call this to discover:
    - Which fields are mandatory when writing documents
    - The API endpoints for read/write/lifecycle operations
    - Dedup and versioning rules
    - The current registration state of all KBs
    """
    kbs = list_kbs()
    return {
        "version": "1.0",
        "description": "MEM 知识库标准化写入规范 — 所有 Agent 统一遵循",
        "mandatory_fields": _KB_SCHEMA_PROPS["mandatory_fields"],
        "optional_metadata": _KB_SCHEMA_PROPS["optional_metadata"],
        "writes": _KB_SCHEMA_PROPS["writes"],
        "reads": _KB_SCHEMA_PROPS["reads"],
        "lifecycle": _KB_SCHEMA_PROPS["lifecycle"],
        "registered_kbs": [
            {
                "kb_id": kb["kb_id"],
                "name": kb["name"],
                "project_id": kb["project_id"],
                "tags": kb["tags"],
                "refresh_policy": kb["refresh_policy"],
                "document_count": kb["document_count"],
                "version": kb["version"],
            }
            for kb in kbs
        ],
    }


def list_kbs(include_inactive: bool = False) -> list[dict[str, Any]]:
    conn = _reg_connect()
    try:
        if include_inactive:
            rows = conn.execute(
                "SELECT * FROM kb_registry ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM kb_registry WHERE status='active' ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_kb(r) for r in rows]
    finally:
        conn.close()


def deregister_kb(kb_id: str, *, hard: bool = False) -> dict[str, Any]:
    """Unregister a KB.

    Args:
        hard: If True, also delete all memories in the project_id.
    """
    conn = _reg_connect()
    try:
        row = conn.execute(
            "SELECT * FROM kb_registry WHERE kb_id=?", (kb_id,)
        ).fetchone()
        if row is None:
            return {"error": "KB_NOT_FOUND", "kb_id": kb_id}

        project_id = row["project_id"]
        conn.execute(
            "UPDATE kb_registry SET status='deleted', updated_at=? WHERE kb_id=?",
            (_now_iso(), kb_id),
        )

        result: dict[str, Any] = {
            "kb_id": kb_id,
            "project_id": project_id,
            "deregistered": True,
            "hard_delete_memories": hard,
        }

        if hard:
            from .app import _connect as mem_connect

            mconn = mem_connect()
            try:
                affected = mconn.execute(
                    "SELECT COUNT(*) AS n FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
                    (project_id,),
                ).fetchone()["n"]
                mconn.execute(
                    "UPDATE memories SET deleted=1, deleted_at=? WHERE project_id=?",
                    (_now_iso(), project_id),
                )
                result["hard_deleted_count"] = affected
            finally:
                mconn.close()

        return result
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Cross-KB search
# --------------------------------------------------------------------------


def cross_kb_search(
    query: str,
    *,
    kb_ids: list[str] | None = None,
    limit: int = 10,
    include_candidates: bool = False,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Search across one or more registered knowledge bases.

    If *kb_ids* is None or empty, searches **all** active KBs.
    Results are aggregated and sorted by descending score.
    """
    # Resolve KB list.
    MEM_BASE = os.environ.get("NTN_MEM_BASE_URL", "http://127.0.0.1:8081")

    kbs = list_kbs()
    if not kbs:
        return {"results": [], "queried_kbs": [], "total": 0}

    if kb_ids:
        matched = {kb["kb_id"]: kb for kb in kbs if kb["kb_id"] in kb_ids}
        candidates = list(matched.values())
        requested_but_not_found = [k for k in kb_ids if k not in matched]
    else:
        candidates = kbs
        requested_but_not_found = []

    if not candidates:
        return {
            "results": [],
            "queried_kbs": [],
            "total": 0,
            "unresolved_kb_ids": requested_but_not_found,
        }

    # Search each KB's project_id via HTTP loopback.
    import urllib.request
    import json as _json

    all_results: list[dict[str, Any]] = []
    queried_kb_ids: list[str] = []
    for kb in candidates:
        body = _json.dumps(
            {
                "query": query,
                "project_id": kb["project_id"],
                "limit": limit,
                "include_candidates": include_candidates,
                **search_kwargs,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{MEM_BASE}/v1/memory/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                response = _json.loads(resp.read().decode("utf-8"))
        except Exception:
            response = {"results": [], "error": "search_failed"}
        results = response.get("results") or []
        for r in results:
            # Tag result with KB metadata.
            r["_kb_id"] = kb["kb_id"]
            r["_kb_name"] = kb["name"]
        all_results.extend(results)
        queried_kb_ids.append(kb["kb_id"])

    # Sort by score descending, cap at limit.
    all_results.sort(key=lambda r: float(r.get("score") or 0), reverse=True)

    return {
        "results": all_results[:limit],
        "queried_kbs": queried_kb_ids,
        "total": len(all_results),
        "unresolved_kb_ids": requested_but_not_found,
    }


# --------------------------------------------------------------------------
# Document ingest
# --------------------------------------------------------------------------


def ingest_documents(
    kb_id: str,
    documents: list[dict[str, Any]],
    *,
    skip_duplicates: bool = True,
    actor_role: str | None = None,
) -> dict[str, Any]:
    """Ingest documents into a registered KB via direct function call (no HTTP loopback).

    Each document dict should have:
        - text (str, required)
        - metadata (dict, optional)
        - source (str, optional) — e.g. "web-scrape", "manual-import"

    Returns counts of ingested and skipped documents.
    """
    import hashlib

    kb = get_kb(kb_id)
    if kb is None:
        return {"error": "KB_NOT_FOUND", "kb_id": kb_id, "ingested": 0, "skipped_duplicates": 0}

    project_id = kb["project_id"]
    ingested = 0
    skipped = 0
    skipped_details: list[str] = []

    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            skipped += 1
            skipped_details.append("empty_text")
            continue

        add_payload = {
            "text": text,
            "project_id": project_id,
            "scope": "shared",
            "memory_type": "knowledge_doc",
            "layer": "mid_term",
            "actor_role": actor_role or kb["owner"] or "admin",
            "source": doc.get("source", "kb_ingest"),
            "metadata": doc.get("metadata", {}),
        }
        if skip_duplicates:
            add_payload["skip_duplicates"] = True

        # Direct import — no HTTP loopback
        try:
            from .app import add_memory as _add_memory
            result = _add_memory(add_payload)
            if result.get("skipped_duplicate"):
                skipped += 1
                skipped_details.append("duplicate_in_same_kb")
            else:
                ingested += 1
        except Exception:
            ingested += 1  # count as ingested even on error

    return {
        "kb_id": kb_id,
        "name": kb["name"],
        "project_id": project_id,
        "ingested": ingested,
        "skipped_duplicates": skipped,
        "skipped_details": skipped_details[:50],
    }


# --------------------------------------------------------------------------
# KB GC: preview-only + manual-execute (safe two-step cleanup)
# --------------------------------------------------------------------------


def kb_gc_preview(kb_id: str, *, stale_days: int = 7) -> dict[str, Any]:
    """Preview which records *could* be safely cleaned.

    Only returns candidates where the same ``text_hash`` has at least one
    ``verified`` alternative — i.e. the content is NOT unique.
    Nothing is deleted — this is read-only.
    """
    kb = get_kb(kb_id)
    if kb is None:
        return {"error": "KB_NOT_FOUND", "kb_id": kb_id}

    project_id = kb["project_id"]
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        # Find superseded records whose text_hash has a verified twin
        rows = conn.execute(
            """
            SELECT s.memory_id, s.text, s.created_at, s.temperature,
                   (SELECT COUNT(*) FROM memories AS v
                    WHERE v.text_hash = s.text_hash
                      AND v.status = 'verified'
                      AND v.project_id = s.project_id
                      AND COALESCE(v.deleted, 0) = 0
                   ) AS verified_count
            FROM memories AS s
            WHERE s.project_id = ?
              AND s.status = 'superseded'
              AND s.created_at < datetime('now', ?)
              AND COALESCE(s.deleted, 0) = 0
              AND s.text_hash IS NOT NULL AND s.text_hash != ''
            GROUP BY s.memory_id
            HAVING verified_count >= 1
            ORDER BY s.created_at ASC
            LIMIT 200
            """,
            (project_id, f"-{stale_days} days"),
        ).fetchall()

        candidates = []
        for r in rows:
            candidates.append({
                "memory_id": r["memory_id"],
                "preview": (r["text"] or "")[:60],
                "created_at": r["created_at"],
                "temperature": r["temperature"],
                "verified_count": r["verified_count"],
            })

        conn.close()
        return {
            "kb_id": kb_id,
            "name": kb["name"],
            "project_id": project_id,
            "total_candidates": len(candidates),
            "stale_days": stale_days,
            "candidates": candidates,
            "note": "预览模式 — 未删除任何记录。确认后调 kb_gc_execute。",
        }
    except Exception as exc:
        conn.close()
        return {"error": "GC_PREVIEW_FAILED", "detail": str(exc)}


def kb_gc_execute(kb_id: str, memory_ids: list[str]) -> dict[str, Any]:
    """Delete the specified memory_ids that were previewed via kb_gc_preview.

    Only executes deletions for *existing* records — if a memory_id
    is already gone, it is silently skipped.
    """
    kb = get_kb(kb_id)
    if kb is None:
        return {"error": "KB_NOT_FOUND", "kb_id": kb_id}

    if not memory_ids:
        return {"error": "NO_IDS_PROVIDED", "detail": "memory_ids is empty"}

    conn = sqlite3.connect(_db_path(), timeout=5.0)
    try:
        placeholders = ",".join("?" for _ in memory_ids)
        # Count before
        before = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories WHERE memory_id IN ({placeholders}) AND COALESCE(deleted,0)=0",
            memory_ids,
        ).fetchone()[0]

        # Soft-delete
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn.execute(
            f"UPDATE memories SET deleted=1, deleted_at=? WHERE memory_id IN ({placeholders})",
            (now, *memory_ids),
        )
        conn.commit()

        # Also delete from cold_archive if present
        try:
            conn.execute(
                f"DELETE FROM cold_archive WHERE memory_id IN ({placeholders})",
                memory_ids,
            )
            conn.commit()
        except Exception:
            pass  # cold_archive may not exist

        return {
            "kb_id": kb_id,
            "name": kb["name"],
            "deleted": before,
            "skipped_not_found": len(memory_ids) - before,
            "deleted_at": now,
        }
    finally:
        conn.close()


def reindex_kb(kb_id: str, *, batch_size: int = 50) -> dict[str, Any]:
    """Trigger embedding regeneration for every document in a KB.

    This sets ``embedding_status='pending'`` so the next write-time
    embedding pipeline picks them up.
    """
    kb = get_kb(kb_id)
    if kb is None:
        return {"error": "KB_NOT_FOUND", "kb_id": kb_id}

    project_id = kb["project_id"]
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
            (project_id,),
        ).fetchall()
        total = rows[0]["n"] if rows else 0
        conn.execute(
            "UPDATE memories SET embedding_status='pending' WHERE project_id=? AND COALESCE(deleted,0)=0",
            (project_id,),
        )
        return {
            "kb_id": kb_id,
            "name": kb["name"],
            "project_id": project_id,
            "total_documents": total,
            "marked_pending": total,
            "message": f"Marked {total} documents for re-embedding. Run reembed script to generate vectors.",
        }
    finally:
        conn.close()
