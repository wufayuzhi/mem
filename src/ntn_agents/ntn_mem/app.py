"""Minimal WSGI application for NTN MEM v2.

The implementation intentionally keeps the runtime lightweight (stdlib WSGI +
SQLite) while enforcing the v2 memory boundary rules: role/private isolation,
project scoping, shared-intel ACLs, soft-delete/history, and provider/job state
tracking. Embedding/vector/Mem0 integrations are represented by local fallback
adapters so the API contract is usable before external providers are wired in.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time as _time
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from ._timeutil import now_iso
from .agent_standard import apply_standard_pull_defaults, apply_standard_push_defaults, standard_agent_profile
from .embedding import cosine, get_embedding_provider
from .logger import log_error, log_info, log_operation, log_warn, query_error_stats, query_errors
from .providers import write_provider
from .qdrant import get_qdrant_client
from .recollect import recount_gist as _recollect
from .recollect import recount_gist as _recollect_detail
from .recollect import _detect_decision_direction
from .tiering import classify_memory, is_expired, retrieval_weight

_SYSLOG = logging.getLogger("ntn-mem")

# ── Runtime embedding provider ────────────────────────────────────────────
# Initialised once at import time so search, write, and worker all use the
# same provider without re-reading env vars every call.
_EMBEDDING_PROVIDER = None  # lazy singleton

def _embedding_provider():
    global _EMBEDDING_PROVIDER
    if _EMBEDDING_PROVIDER is None:
        from .embedding import get_embedding_provider
        _EMBEDDING_PROVIDER = get_embedding_provider()
        import os
        _SYSLOG.info("embedding provider init: name=%s model=%s base_url set=%s api_key set=%s",
                     getattr(_EMBEDDING_PROVIDER, 'name', '?'),
                     os.environ.get('NTN_MEM_EMBEDDING_MODEL', 'not-set'),
                     'yes' if os.environ.get('NTN_MEM_EMBEDDING_BASE_URL') else 'no',
                     'yes' if os.environ.get('NTN_MEM_EMBEDDING_API_KEY') else 'no')
    return _EMBEDDING_PROVIDER

StartResponse = Callable[[str, list[tuple[str, str]]], Any]
_JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
DEFAULT_MEM_DB = "/data/mem.db"
ROLE_KEYS = {"laohao", "xiaok", "xiaozhou", "laoli", "laochen", "laosun", "laowang", "laozhang"}
ADMIN_ROLES = {"admin", "laohao"}

# ── 会话热缓存 ──
_SEARCH_CACHE: dict[str, tuple[list[dict], float]] = {}  # cache_key → (results, timestamp)
_SEARCH_CACHE_TTL = 300  # 5分钟

def _cache_key(agent_key: str, query: str, project_id: str | None) -> str:
    return hashlib.md5(f"{agent_key}|{query}|{project_id}".encode()).hexdigest()

def _cache_get(key: str) -> list[dict] | None:
    entry = _SEARCH_CACHE.get(key)
    if entry and (datetime.now(timezone.utc).timestamp() - entry[1]) < _SEARCH_CACHE_TTL:
        return entry[0]
    return None

def _cache_set(key: str, results: list[dict]) -> None:
    _SEARCH_CACHE[key] = (results, datetime.now(timezone.utc).timestamp())

def _detect_lang(text: str) -> str:
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if cjk > 0 else "en"

def _lang_match(text: str, lang: str) -> bool:
    if not text:
        return True
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if lang == "zh":
        return has_cjk
    return not has_cjk


def _json_response(start_response: StartResponse, status: str, payload: dict[str, Any]) -> Iterable[bytes]:
    # wsgiref mutates the header list to append Content-Length for single-item
    # responses.  Never pass the module-level _JSON_HEADERS list directly or a
    # short response such as /health will poison later, longer JSON responses.
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [*_JSON_HEADERS, ("Content-Length", str(len(body)))]
    start_response(status, headers)
    return [body]


def _authorized(environ: dict[str, Any], token_env: str) -> bool:
    token = os.environ.get(token_env)
    if not token:
        return True
    return environ.get("HTTP_AUTHORIZATION") == f"Bearer {token}"


def _unauthorized(start_response: StartResponse) -> Iterable[bytes]:
    return _json_response(start_response, "401 Unauthorized", {"error": {"code": "UNAUTHORIZED"}})


def _read_json(environ: dict[str, Any]) -> dict[str, Any]:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    stream = environ.get("wsgi.input")
    raw = stream.read(length) if stream is not None and length > 0 else b"{}"
    try:
        value = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("BAD_JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("BAD_JSON")
    return value


def _db_path() -> str:
    return os.environ.get("NTN_MEM_DB", DEFAULT_MEM_DB)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _connect() -> sqlite3.Connection:
    target = _db_path()
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            role TEXT,
            project_id TEXT,
            layer TEXT,
            scope TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_jobs (
            memory_job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            memory_ids_json TEXT NOT NULL,
            provider TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_history (
            history_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_role TEXT,
            before_json TEXT NOT NULL,
            after_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cold_archive (
            memory_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            role TEXT,
            project_id TEXT,
            layer TEXT,
            scope TEXT,
            temperature TEXT,
            importance INTEGER,
            protected INTEGER,
            supersedes TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            archived_at TEXT NOT NULL,
            archive_reason TEXT,
            original_json TEXT NOT NULL
        );
        """
    )
    _ensure_columns(
        conn,
        "memories",
        {
            "provider": "TEXT DEFAULT 'local'",
            "provider_memory_id": "TEXT",
            "summary": "TEXT",
            "owner_role": "TEXT",
            "actor_role": "TEXT",
            "source_role": "TEXT",
            "task_id": "TEXT",
            "event_id": "TEXT",
            "visibility": "TEXT",
            "acl_json": "TEXT DEFAULT '[]'",
            "allowed_roles_json": "TEXT DEFAULT '[]'",
            "memory_type": "TEXT",
            "source": "TEXT",
            "status": "TEXT DEFAULT 'verified'",
            "confidence": "TEXT DEFAULT 'verified'",
            "vitality": "INTEGER DEFAULT 0",
            "recall_count": "INTEGER DEFAULT 0",
            "updated_at": "TEXT",
            "last_recalled_at": "TEXT",
            "deleted": "INTEGER DEFAULT 0",
            "deleted_at": "TEXT",
            "deleted_by": "TEXT",
            "delete_reason": "TEXT",
            "archived": "INTEGER DEFAULT 0",
            "archive_id": "TEXT",
            "expires_at": "TEXT",
            "text_hash": "TEXT",
            "embedding_status": "TEXT DEFAULT 'pending'",
            "temperature": "TEXT DEFAULT 'warm'",
            "importance": "INTEGER DEFAULT 50",
            "protected": "INTEGER DEFAULT 0",
            "supersedes": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        "memory_jobs",
        {
            "provider_event_id": "TEXT",
            "error_json": "TEXT",
            "attempts": "INTEGER DEFAULT 0",
            "next_retry_at": "TEXT",
            "updated_at": "TEXT",
        },
    )
    return conn


def route_memory(data: dict[str, Any]) -> dict[str, Any]:
    text = (data.get("text") or "").lower()
    keywords = ("继续", "历史", "经验", "之前", "记忆", "remember", "memory")
    should = any(k in text for k in keywords)
    budget = data.get("budget") or {}
    scopes = ["role_private", "project", "shared"]
    layers = ["long_term", "mid_term"]
    if budget.get("allow_cold"):
        layers.append("cold")
    return {
        "should_recall": should,
        "degraded": False,
        "reason": "dispatch_keyword_hit" if should else "no_recall_keyword",
        "layers": layers if should else [],
        "scopes": scopes if should else [],
        "filters": {
            "role": [data.get("target_role")] if data.get("target_role") else [],
            "project_id": data.get("project_id"),
        },
        "budget": {
            "max_results": int(budget.get("max_results") or budget.get("top_k") or 10),
            "max_tokens": int(budget.get("max_tokens") or 1200),
        },
    }


def _require_agent_key(data: dict[str, Any]) -> str:
    return standard_agent_profile(data).agent_key


def agent_push_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Standard Agent -> MEM push path, independent from SQL-HTTP events.

    New standardized Agents only need a stable agent_key. MEM then normalizes
    private/shared namespaces and always runs memory-tiering during add_memory().

    On first push with ``scope=role_private`` the agent is automatically
    registered in the Private Memory Manager's agent registry.
    """
    normalized_data = apply_standard_push_defaults(data)
    agent_key = normalized_data["agent_key"]
    text = str(normalized_data.get("text") or "").strip()
    if not text:
        raise ValueError("TEXT_REQUIRED")
    scope = normalized_data["scope"]
    # Auto-register private agents.
    if scope == "role_private" and agent_key:
        from .manager_private import ensure_agent_registered

        reg_project_id = ensure_agent_registered(agent_key)
        normalized_data["project_id"] = reg_project_id
    project_id = normalized_data["project_id"]
    result = add_memory(
        {
            **normalized_data,
            "text": text,
            "memory_type": normalized_data.get("memory_type") or normalized_data.get("kind"),
        }
    )
    return {"agent_key": agent_key, "project_id": project_id, "scope": scope, "source": "agent_push", **result}


def agent_pull_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Standard Agent <- MEM pull path across private + shared projects."""
    normalized_data = apply_standard_pull_defaults(data)
    agent_key = normalized_data["agent_key"]
    private_project = normalized_data["private_memory_project"]
    shared_projects = [p for p in _listify(normalized_data.get("shared_knowledge_projects")) if p]
    queried_projects: list[str] = []
    results: list[dict[str, Any]] = []
    limit = int(data.get("limit") or data.get("top_k") or 10)
    for project_id in [private_project, *shared_projects]:
        if project_id in queried_projects:
            continue
        queried_projects.append(project_id)
        response = search_memory(
            {
                **normalized_data,
                "caller_role": normalized_data.get("caller_role") or agent_key,
                "target_role": normalized_data.get("target_role") if project_id != private_project else normalized_data.get("target_role") or agent_key,
                "project_id": project_id,
                "limit": limit,
            }
        )
        results.extend(response.get("results") or [])
    results.sort(key=lambda item: (item.get("score") or 0, item.get("created_at") or ""), reverse=True)
    return {"agent_key": agent_key, "project_id": private_project, "queried_projects": queried_projects, "results": results[:limit]}


def _tokenize(text: str) -> set[str]:
    lowered = (text or "").lower()
    tokens: set[str] = set()
    current: list[str] = []
    for ch in lowered:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                tokens.add(token)
                if any("\u4e00" <= c <= "\u9fff" for c in token):
                    tokens.update(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
                current.clear()
    if current:
        token = "".join(current)
        tokens.add(token)
        if any("\u4e00" <= c <= "\u9fff" for c in token):
            tokens.update(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
    return {t for t in tokens if t}


def _local_embedding(text: str) -> list[int]:
    """Deterministic tiny embedding fallback used until a real model is wired."""
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    return [b for b in digest[:16]]


def _ensure_embedding_cache(conn: sqlite3.Connection, text: str, provider_name: str | None = None, embedding: list[float] | list[int] | None = None) -> bool:
    provider = provider_name or "local"
    h = _text_hash(f"{provider}:{text}")
    row = conn.execute("SELECT text_hash FROM embedding_cache WHERE text_hash=?", (h,)).fetchone()
    if row:
        return True
    vector = embedding if embedding is not None else _local_embedding(text)
    conn.execute(
        "INSERT INTO embedding_cache (text_hash, provider, embedding_json, created_at) VALUES (?, ?, ?, ?)",
        (h, provider, json.dumps(vector), now_iso()),
    )
    return False


def _row_to_memory(row: sqlite3.Row) -> dict[str, Any]:
    allowed_roles = _json_loads(row["allowed_roles_json"] if "allowed_roles_json" in row.keys() else None, [])
    return {
        "memory_id": row["memory_id"],
        "text": row["text"],
        "summary": row["summary"] if "summary" in row.keys() else None,
        "role": row["role"],
        "owner_role": row["owner_role"] if "owner_role" in row.keys() else row["role"],
        "actor_role": row["actor_role"] if "actor_role" in row.keys() else None,
        "source_role": row["source_role"] if "source_role" in row.keys() else None,
        "project_id": row["project_id"],
        "task_id": row["task_id"] if "task_id" in row.keys() else None,
        "event_id": row["event_id"] if "event_id" in row.keys() else None,
        "layer": row["layer"],
        "scope": row["scope"],
        "visibility": row["visibility"] if "visibility" in row.keys() else None,
        "allowed_roles": allowed_roles,
        "acl": _json_loads(row["acl_json"] if "acl_json" in row.keys() else None, []),
        "memory_type": row["memory_type"] if "memory_type" in row.keys() else None,
        "source": row["source"] if "source" in row.keys() else None,
        "status": row["status"] if "status" in row.keys() else "verified",
        "confidence": row["confidence"] if "confidence" in row.keys() else "verified",
        "provider": row["provider"] if "provider" in row.keys() else "local",
        "provider_memory_id": row["provider_memory_id"] if "provider_memory_id" in row.keys() else None,
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"] if "updated_at" in row.keys() else None,
        "deleted": bool(row["deleted"]) if "deleted" in row.keys() else False,
        "archived": bool(row["archived"]) if "archived" in row.keys() else False,
        "expires_at": row["expires_at"] if "expires_at" in row.keys() else None,
        "temperature": row["temperature"] if "temperature" in row.keys() else "warm",
        "importance": row["importance"] if "importance" in row.keys() else row["vitality"] if "vitality" in row.keys() else 50,
        "protected": bool(row["protected"]) if "protected" in row.keys() else False,
        "supersedes": row["supersedes"] if "supersedes" in row.keys() else None,
        "embedding_status": row["embedding_status"] if "embedding_status" in row.keys() else "pending",
    }


def _is_admin(role: str | None) -> bool:
    return bool(role and role in ADMIN_ROLES)


def _is_candidate_status(status: Any) -> bool:
    return str(status or "").lower() == "candidate"


def _visible(memory: dict[str, Any], caller_role: str | None, project_id: str | None, include_candidates: bool) -> bool:
    if memory.get("deleted"):
        return False
    if _is_candidate_status(memory.get("status")) and not include_candidates:
        # Candidate shared intel is never visible to peer/ACL roles unless
        # callers explicitly ask for candidate review. Owner/source/admin may
        # inspect candidates by default for moderation and promotion.
        return bool(caller_role and (caller_role == memory.get("owner_role") or caller_role == memory.get("source_role") or _is_admin(caller_role)))
    scope = memory.get("scope") or "role_private"
    visibility = memory.get("visibility") or ("private" if scope == "role_private" else "project_roles")
    owner = memory.get("owner_role") or memory.get("role")
    allowed = set(_listify(memory.get("allowed_roles"))) | set(_listify(memory.get("acl")))

    if _is_admin(caller_role):
        return True
    if visibility == "admin_only":
        return False
    if scope == "task_context":
        return bool(project_id and memory.get("project_id") == project_id)
    if scope == "role_private" or visibility == "private":
        return bool(caller_role and caller_role == owner)
    if scope == "shared" or visibility == "shared_acl":
        # Knowledge base documents (scope=shared) are readable by any caller.
        # If no caller_role is specified, shared docs are still visible.
        if not caller_role and (scope == "shared" or memory.get("memory_type") == "knowledge_doc"):
            return True
        return bool(caller_role and (caller_role in allowed or caller_role == owner or caller_role == memory.get("source_role")))
    if scope == "project" or visibility == "project_roles":
        return bool(project_id and memory.get("project_id") == project_id)
    if scope == "company":
        return bool(caller_role)
    return False


def search_memory(data: dict[str, Any]) -> dict[str, Any]:
    query_tokens = _tokenize(data.get("query") or data.get("text") or "")
    limit = int(data.get("limit") or data.get("top_k") or 10)
    caller_role = data.get("caller_role") or data.get("actor_role") or data.get("role") or data.get("agent_key")
    target_role = data.get("target_role") or data.get("role")
    project_id = data.get("project_id")
    scopes = set(_listify(data.get("scopes")))
    layers = set(_listify(data.get("layers")))
    agent_key = data.get("agent_key") or caller_role or ""

    # ── 语言检测 ↴ 过滤不同语言的记忆 ──
    lang = _detect_lang(data.get("query") or data.get("text") or "")

    # ── 会话热缓存 ↴ 同一话题不重复搜索 ──
    cache_key = _cache_key(agent_key, data.get("query", ""), project_id)
    cached = _cache_get(cache_key)
    if cached is not None:
        return {
            "results": cached[:limit],
            "usage": {
                "embedding_cache_hit": True,
                "vector_ms": 0,
                "total_ms": 0,
                "provider": "cache",
                "degraded": False,
            },
        }

    # ── boost_terms 话题偏置 ↴ 给相关记忆更高权重 ──
    boost_terms = data.get("boost_terms", [])
    if isinstance(boost_terms, str):
        boost_terms = [t.strip() for t in boost_terms.split(",") if t.strip()]
    query_text = data.get("query") or data.get("text") or ""
    # ── 决策方向偏置 ↴ 最高优先级——检测 query 中的决策信号并加权 ──
    if not boost_terms and query_text:
        query_lower = query_text.lower()
        _decision_match = _detect_decision_direction(query_lower)
        if _decision_match:
            boost_terms = _decision_match["boost"]
    # ── 场景感知加权 ↴ 次优先级——未匹配决策方向时自动检测场景 ──
    if not boost_terms and query_text:
        try:
            from .recollect import _detect_scene, _add_scene_boost
            scene = _detect_scene(query_text)
            scene_boost = _add_scene_boost(scene)
            if scene_boost:
                boost_terms = scene_boost
        except Exception:
            log_warn(module="app", operation="search_memory",
                     summary="Scene detection failed, skipping scene boost")
            pass
    # `include_candidates` is an explicit review mode for owner/source/admin only;
    # peer ACL roles must wait until the intel is promoted to verified.
    include_candidates = bool(data.get("include_candidates")) and bool(
        caller_role and (target_role is None or caller_role == target_role or _is_admin(caller_role))
    )

    # ── 自动解析 project_id ↴ 如果 caller_role 但没传 project_id ──
    if caller_role and not project_id:
        try:
            from .manager_private import ensure_agent_registered
            project_id = ensure_agent_registered(caller_role)
        except Exception:
            log_warn(module="app", operation="search_memory",
                     summary="Auto-register agent failed for project_id resolution")
            pass

    if caller_role and not project_id and not _is_admin(caller_role):
        return {
            "results": [],
            "usage": {
                "embedding_cache_hit": False,
                "vector_ms": 0,
                "total_ms": 0,
                "provider": "local",
                "degraded": True,
                "reason": "project_id_required",
            },
        }

    conn = _connect()
    cache_hit = False
    provider_name = "local-token"
    degraded = False
    degrade_reason = None
    query_embedding: list[float] | None = None
    qdrant_rank: dict[str, tuple[int, float]] = {}
    try:
        provider = _embedding_provider()
        provider_name = provider.name
        query_embedding = provider.embed(data.get("query") or data.get("text") or "")
        cache_hit = _ensure_embedding_cache(conn, data.get("query") or data.get("text") or "", provider_name, query_embedding)
        qdrant_client = get_qdrant_client()
        if qdrant_client is not None:
            try:
                qdrant_results = qdrant_client.search(
                    query_embedding,
                    limit=max(limit * 4, limit, 10),
                    filters={"project_id": project_id} if project_id else None,
                )
                qdrant_rank = {item.memory_id: (idx, item.score) for idx, item in enumerate(qdrant_results)}
                provider_name = f"{provider_name}+qdrant"
            except Exception:
                degraded = True
                degrade_reason = "qdrant_search_failed"
                log_warn(module="app", operation="search_memory",
                         summary="Qdrant search failed, falling back to local search",
                         exc=Exception(degrade_reason))
    except Exception:
        provider_name = "local-token"
        degraded = True
        degrade_reason = "embedding_provider_failed"
        query_embedding = None
        cache_hit = _ensure_embedding_cache(conn, data.get("query") or data.get("text") or "", "local-token")
        log_error(module="app", operation="search_memory",
                  summary="Embedding provider failed, falling back to token search",
                  exc=Exception(degrade_reason))
    try:
        sql = "SELECT * FROM memories WHERE COALESCE(deleted, 0)=0"
        params: list[Any] = []
        if target_role:
            # Backward-compatible target narrowing; ACL still runs after this.
            sql += " AND (role=? OR owner_role=? OR scope IN ('project', 'shared', 'task_context'))"
            params.extend([target_role, target_role])
        if project_id:
            sql += " AND (project_id=? OR project_id IS NULL)"
            params.append(project_id)
        if scopes:
            sql += f" AND scope IN ({','.join('?' for _ in scopes)})"
            params.extend(sorted(scopes))
        if layers:
            sql += f" AND layer IN ({','.join('?' for _ in layers)})"
            params.extend(sorted(layers))
        rows = conn.execute(sql + " ORDER BY created_at DESC", params).fetchall()

        results: list[dict[str, Any]] = []
        recalled_ids: list[str] = []
        for row in rows:
            memory = _row_to_memory(row)
            if _is_candidate_status(memory.get("status")) and not include_candidates:
                if not (caller_role and (caller_role == memory.get("owner_role") or caller_role == memory.get("source_role") or _is_admin(caller_role))):
                    continue
            if not _visible(memory, caller_role, project_id, include_candidates):
                continue
            if is_expired(memory):
                continue
            if memory.get("temperature") == "decayed" and not data.get("include_decayed"):
                continue
            if qdrant_rank and memory["memory_id"] not in qdrant_rank:
                continue
            text = memory["text"]
            text_tokens = _tokenize(text)
            overlap = query_tokens & text_tokens
            has_lexical_match = bool(overlap) or any(token in text.lower() for token in query_tokens)
            if query_tokens and not has_lexical_match and (degraded or provider_name in {"local-token"}):
                # Avoid surfacing unrelated rows from weak/token-only fallback.
                # In particular, task_context rows should not match queries such
                # as "upstream timeout" solely because a fallback score exists.
                continue
            if qdrant_rank:
                score = qdrant_rank[memory["memory_id"]][1]
            else:
                score = len(overlap) / max(len(query_tokens), 1)
                if query_embedding is not None:
                    try:
                        vector_score = cosine(query_embedding, _embedding_provider().embed(text))
                        if has_lexical_match or vector_score > 0:
                            score = max(score, vector_score)
                        elif memory.get("scope") == "task_context":
                            continue
                    except Exception:
                        degraded = True
                        degrade_reason = "embedding_provider_failed"
                        log_warn(module="app", operation="search_memory",
                                 summary="Token fallback embedding failed for score calculation")
                if score == 0 and query_tokens:
                    score = 0.1
            memory["score"] = score if qdrant_rank else score + retrieval_weight(memory)
            results.append(memory)
            recalled_ids.append(memory["memory_id"])
        if recalled_ids:
            conn.executemany(
                "UPDATE memories SET recall_count=COALESCE(recall_count,0)+1, last_recalled_at=? WHERE memory_id=?",
                [(now_iso(), mid) for mid in recalled_ids],
            )
    finally:
        conn.close()

    if qdrant_rank:
        results.sort(key=lambda item: qdrant_rank[item["memory_id"]][0])
    else:
        results.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)

    # ── 冷归档层 fallback ↴ 热层不够时查 cold_archive ──
    if not results and query_tokens:
        try:
            conn2 = _connect()
            archive_rows = conn2.execute(
                "SELECT memory_id, text, project_id, role, created_at, metadata_json "
                "FROM cold_archive WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{list(query_tokens)[0] if query_tokens else ''}%", limit),
            ).fetchall()
            for row in archive_rows:
                memory = _row_to_memory(dict(row))
                memory["score"] = 0.05
                memory["_source"] = "cold_archive"
                results.append(memory)
            if archive_rows:
                _SYSLOG.info("cold archive fallback: %d results for query", len(archive_rows))
            conn2.close()
        except Exception:
            log_warn(module="app", operation="search_memory",
                     summary="Cold archive fallback query failed")
            pass

    # ── 话题偏置 ↴ boost_terms 匹配的记忆 ×1.2 ──
    if boost_terms:
        boost_lower = [t.lower() for t in boost_terms]
        for r in results:
            text = r.get("text", "").lower()
            if any(t in text for t in boost_lower):
                r["score"] = r.get("score", 0) * 1.2

    # ── 语言过滤 ↴ 同语言记忆排前面 ──
    if lang:
        lang_results = [r for r in results if _lang_match(r.get("text", ""), lang)]
        other_results = [r for r in results if not _lang_match(r.get("text", ""), lang)]
        results = lang_results + other_results

    usage = {"embedding_cache_hit": cache_hit, "vector_ms": 0, "total_ms": 0, "provider": provider_name, "degraded": degraded}
    if degrade_reason:
        usage["reason"] = degrade_reason

    # ── 热缓存 ↴ 只缓存非退化结果 ──
    if not degraded:
        _cache_set(cache_key, results)

    return {
        "results": results[:limit],
        "usage": usage,
    }


def _normalize_memory_input(data: dict[str, Any]) -> dict[str, Any]:
    tier = classify_memory(data)
    scope = data.get("scope") or "role_private"
    role = data.get("role") or data.get("target_role")
    owner_role = data.get("owner_role") or role
    visibility = data.get("visibility")
    if not visibility:
        visibility = "private" if scope == "role_private" else "shared_acl" if scope == "shared" else "project_roles"
    allowed_roles = _listify(data.get("allowed_roles") or data.get("acl"))
    metadata = data.get("metadata") or {}
    if data.get("evidence") is not None:
        metadata.setdefault("evidence", data.get("evidence"))
    # ── source_path 自动标记：根据 source 字段映射写入路径 ──
    _SOURCE_PATH_MAP = {
        "hermesagent_cn_kb_ingest": "kb_sync/hermes_docs",
        "ccb_docs_ingest": "kb_sync/ccb_docs",
        "agent_push": "agent_push",
        "event_ingest": "event_ingest",
        "mem_profile_update": "profile_update",
        "kb_ingest": "kb_sync/manual_kb",
        "experience_extraction": "experience_extraction",
    }
    source_path = _SOURCE_PATH_MAP.get(data.get("source"))
    if source_path:
        metadata.setdefault("source_path", source_path)
    return {
        "text": data["text"],
        "summary": data.get("summary"),
        "role": role,
        "owner_role": owner_role,
        "actor_role": data.get("actor_role"),
        "source_role": data.get("source_role") or data.get("actor_role") or role,
        "project_id": data.get("project_id"),
        "task_id": data.get("task_id"),
        "event_id": str(data.get("event_id")) if data.get("event_id") is not None else None,
        "layer": tier.layer,
        "scope": scope,
        "visibility": visibility,
        "acl_json": _json_dumps(allowed_roles),
        "allowed_roles_json": _json_dumps(allowed_roles),
        "memory_type": tier.memory_type,
        "source": data.get("source"),
        "status": data.get("status") or ("candidate" if data.get("candidate") else "verified"),
        "confidence": data.get("confidence") or data.get("status") or "verified",
        "metadata_json": _json_dumps(metadata),
        "provider": data.get("provider") or "local",
        "provider_memory_id": data.get("provider_memory_id"),
        "expires_at": tier.expires_at,
        "archived": 1 if tier.archived else 0,
        "archive_id": data.get("archive_id"),
        "temperature": tier.temperature,
        "importance": tier.importance,
        "protected": 1 if tier.protected else 0,
        "supersedes": data.get("supersedes"),
        "text_hash": _text_hash(data["text"]),
        "embedding_status": data.get("embedding_status") or "pending",
    }


def add_memory(data: dict[str, Any]) -> dict[str, Any]:
    memory_id = data.get("memory_id") or f"mem-{uuid.uuid4()}"
    memory_job_id = f"job-{uuid.uuid4()}"
    created_at = now_iso()
    
    # ── 全局 text_hash 去重检查: 相同 project_id + text_hash + 非deleted 就跳过 ──
    text = (data.get("text") or "").strip()
    project_id = data.get("project_id", "")
    if text and project_id:
        text_hash_val = hashlib.sha256(text.encode("utf-8")).hexdigest()
        dup_conn = _connect()
        try:
            existing = dup_conn.execute(
                "SELECT memory_id FROM memories WHERE project_id=? AND text_hash=? AND COALESCE(deleted,0)=0 LIMIT 1",
                (project_id, text_hash_val),
            ).fetchone()
            if existing is not None:
                dup_conn.close()
                return {
                    "memory_id": existing["memory_id"],
                    "skipped_duplicate": True,
                    "message": "Duplicate detected — same project_id + text_hash already exists",
                }
        except Exception:
            log_warn(module="app", operation="add_memory",
                     summary=f"Duplicate check failed for project={project_id}")
            pass
        finally:
            dup_conn.close()
    
    # Normalize after dedup check
    normalized = _normalize_memory_input(data)
    provider_result = write_provider({**data, **normalized}, memory_id=memory_id, client=data.get("provider_client"))
    normalized["provider"] = provider_result.provider
    normalized["provider_memory_id"] = provider_result.provider_memory_id
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_embedding_cache(conn, normalized["text"])
        conn.execute(
            """
            INSERT INTO memories (
                memory_id, text, role, project_id, layer, scope, metadata_json, created_at,
                provider, provider_memory_id, summary, owner_role, actor_role, source_role,
                task_id, event_id, visibility, acl_json, allowed_roles_json, memory_type,
                source, status, confidence, updated_at, archived, archive_id, expires_at,
                temperature, importance, protected, supersedes, text_hash, embedding_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                normalized["text"],
                normalized["role"],
                normalized["project_id"],
                normalized["layer"],
                normalized["scope"],
                normalized["metadata_json"],
                created_at,
                normalized["provider"],
                normalized["provider_memory_id"],
                normalized["summary"],
                normalized["owner_role"],
                normalized["actor_role"],
                normalized["source_role"],
                normalized["task_id"],
                normalized["event_id"],
                normalized["visibility"],
                normalized["acl_json"],
                normalized["allowed_roles_json"],
                normalized["memory_type"],
                normalized["source"],
                normalized["status"],
                normalized["confidence"],
                created_at,
                normalized["archived"],
                normalized["archive_id"],
                normalized["expires_at"],
                normalized["temperature"],
                normalized["importance"],
                normalized["protected"],
                normalized["supersedes"],
                normalized["text_hash"],
                normalized["embedding_status"],
            ),
        )
        conn.execute(
            """
            INSERT INTO memory_jobs (
                memory_job_id, status, memory_ids_json, provider, provider_event_id,
                error_json, attempts, next_retry_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_job_id,
                provider_result.status,
                json.dumps([memory_id]),
                normalized["provider"],
                provider_result.provider_event_id,
                json.dumps(provider_result.error, ensure_ascii=False),
                1 if provider_result.error else 0,
                _future_iso(_retry_delay_seconds(1)) if provider_result.error else None,
                created_at,
                created_at,
            ),
        )
        conn.execute(
            "INSERT INTO memory_history (history_id, memory_id, action, actor_role, before_json, after_json, created_at) VALUES (?, ?, 'create', ?, '{}', ?, ?)",
            (f"hist-{uuid.uuid4()}", memory_id, normalized["actor_role"] or normalized["source_role"], json.dumps({**normalized, "memory_id": memory_id}, ensure_ascii=False), created_at),
        )
        conn.execute("COMMIT")
        vector_provider = "hash"
        vector_degraded = False
        vector_reason = None
        qdrant_client = get_qdrant_client()
        if qdrant_client is not None:
            try:
                embedding_provider = _embedding_provider()
                vector = embedding_provider.embed(normalized["text"])
                qdrant_client.upsert(
                    memory_id,
                    vector,
                    {
                        "memory_id": memory_id,
                        "project_id": normalized["project_id"],
                        "role": normalized["role"],
                        "owner_role": normalized["owner_role"],
                        "source_role": normalized["source_role"],
                        "scope": normalized["scope"],
                        "visibility": normalized["visibility"],
                        "status": normalized["status"],
                        "layer": normalized["layer"],
                        "memory_type": normalized["memory_type"],
                        "source_path": json.loads(normalized["metadata_json"]).get("source_path", ""),
                        "created_at": created_at,
                    },
                )
                # Also save the real vector to embedding_cache for search-time reuse.
                try:
                    _ensure_embedding_cache(
                        _connect(),
                        normalized["text"],
                        provider_name=embedding_provider.name,
                        embedding=vector,
                    )
                except Exception:
                    log_warn(module="app", operation="add_memory",
                             summary="Embedding cache write failed (non-blocking)")
                    pass
                vector_provider = f"{embedding_provider.name}+qdrant"
            except Exception:
                vector_degraded = True
                vector_reason = "qdrant_upsert_failed"
                log_warn(module="app", operation="add_memory",
                         summary="Qdrant upsert failed (non-blocking)", exc=Exception(vector_reason))
        response = {
            "accepted": True,
            "memory_job_id": memory_job_id,
            "provider": normalized["provider"],
            "provider_event_id": provider_result.provider_event_id or memory_id,
            "vector_provider": vector_provider,
            "vector_degraded": vector_degraded,
        }
        if vector_reason:
            response["vector_reason"] = vector_reason
        # ---- Lightweight: incremental graph index on write ----
        try:
            from .graph_index import ensure_graph_tables, add_entity_relations
            ensure_graph_tables(_connect())
            add_entity_relations(memory_id, normalized["text"], dict(normalized.get("metadata") or {}))
        except Exception:
            log_warn(module="app", operation="add_memory",
                     summary="Graph index incremental update failed (non-blocking)")
            pass  # graph index failure must NEVER block memory write
        return response
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    finally:
        conn.close()


def get_memory(memory_id: str, caller_role: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            return None
        memory = _row_to_memory(row)
        if not _visible(memory, caller_role or memory.get("owner_role"), memory.get("project_id"), include_candidates=True):
            return None
        return memory
    finally:
        conn.close()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def run_lifecycle(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run MEM-side lifecycle governance as a backend job.

    Protected memories are never decayed. Everything else is governed in-place:
    expired -> decayed/cold/archived, old hot -> warm, old warm -> cold.

    Supports offset-based pagination: pass ``offset`` to resume from where a
    previous call left off.  Returns ``next_offset`` = 0 when all rows scanned.
    """
    data = data or {}
    hot_days = int(data.get("hot_days") or 7)
    warm_days = int(data.get("warm_days") or 30)
    limit = int(data.get("limit") or 1000)
    offset = int(data.get("offset") or 0)
    dry_run = bool(data.get("dry_run"))
    now = datetime.now(timezone.utc)
    conn = _connect()
    scanned = expired = hot_to_warm = warm_to_cold = decayed = 0
    next_offset = 0
    changes: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT * FROM memories WHERE COALESCE(deleted,0)=0 ORDER BY created_at LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        for row in rows:
            scanned += 1
            memory = _row_to_memory(row)
            if memory.get("protected"):
                continue
            created_at = _parse_iso(memory.get("created_at")) or now
            age_days = max(0, (now - created_at).days)
            updates: dict[str, Any] = {}
            reason = None
            if is_expired(memory, now):
                updates = {"temperature": "decayed", "layer": "cold", "archived": 1}
                reason = "expired_to_decayed"
                expired += 1
                decayed += 1
            elif memory.get("temperature") == "hot" and age_days >= hot_days:
                updates = {"temperature": "warm", "layer": "mid_term" if memory.get("memory_type") == "episodic" else "long_term"}
                reason = "hot_to_warm"
                hot_to_warm += 1
            elif memory.get("temperature") == "warm" and age_days >= warm_days:
                updates = {"temperature": "cold", "layer": "cold", "archived": 1}
                reason = "warm_to_cold"
                warm_to_cold += 1
            if not updates:
                continue
            after = {**memory, **updates, "updated_at": now_iso()}
            changes.append({"memory_id": memory["memory_id"], "reason": reason, "before": {"temperature": memory.get("temperature"), "layer": memory.get("layer"), "archived": memory.get("archived")}, "after": {"temperature": after.get("temperature"), "layer": after.get("layer"), "archived": after.get("archived")}})
            if dry_run:
                continue
            set_clause = ", ".join(f"{key}=?" for key in updates)
            conn.execute(f"UPDATE memories SET {set_clause}, updated_at=? WHERE memory_id=?", [*updates.values(), now_iso(), memory["memory_id"]])
            conn.execute(
                "INSERT INTO memory_history (history_id, memory_id, action, actor_role, before_json, after_json, created_at) VALUES (?, ?, 'lifecycle', ?, ?, ?, ?)",
                (f"hist-{uuid.uuid4()}", memory["memory_id"], data.get("actor_role") or "mem-lifecycle", json.dumps(memory, ensure_ascii=False), json.dumps(after, ensure_ascii=False), now_iso()),
            )
        next_offset = offset + limit if scanned == limit else 0
        # ── 自动冷归档 ↴ lifecycle 治理后 cold+superseded>7d → cold_archive ──
        if not dry_run and next_offset == 0:
            try:
                cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
                archive_rows = conn.execute(
                    "SELECT m.* FROM memories m LEFT JOIN cold_archive ca ON m.memory_id=ca.memory_id "
                    "WHERE m.temperature='cold' AND m.status='superseded' AND m.created_at < ? AND ca.memory_id IS NULL",
                    (cutoff,),
                ).fetchall()
                archived = 0
                for row in archive_rows:
                    d = dict(row)
                    import json as _json
                    conn.execute(
                        "INSERT OR IGNORE INTO cold_archive (memory_id, text, role, project_id, layer, scope, temperature, importance, protected, supersedes, metadata_json, created_at, archived_at, archive_reason, original_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (d["memory_id"], d["text"], d.get("role"), d.get("project_id"), d.get("layer"), d.get("scope"), d.get("temperature"), d.get("importance"), d.get("protected"), d.get("supersedes"), d.get("metadata_json", "{}"), d["created_at"], now_iso(), "lifecycle_cold_archive", _json.dumps(d)),
                    )
                    conn.execute("DELETE FROM memories WHERE memory_id=?", (d["memory_id"],))
                    archived += 1
                if archived:
                    conn.commit()
            except Exception:
                pass
        return {"accepted": True, "dry_run": dry_run, "scanned": scanned, "expired": expired, "hot_to_warm": hot_to_warm, "warm_to_cold": warm_to_cold, "decayed": decayed, "changes": changes, "next_offset": next_offset, "has_more": scanned == limit}
    finally:
        conn.close()


CONTRADICTION_PATTERNS = {
    "是", "不是", "有", "没有", "启用", "禁用", "开", "关", "是", "否",
    "能", "不能", "需要", "不需要", "支持", "不支持", "使用", "不使用",
    "端口是", "端口不是", "开启", "关闭", "已启用", "已禁用",
}


def _is_text_conflict(text_a: str, text_b: str) -> bool:
    """Detect if two texts semantically contradict each other.

    Uses a lightweight rule-based approach: check for negation flip
    on known predicate keywords.
    """
    norm = lambda t: t.strip().lower().rstrip("。！？.!?")
    a, b = norm(text_a), norm(text_b)
    if a == b:
        return False
    for pat in CONTRADICTION_PATTERNS:
        pos = pat
        neg = "不" + pat if not pat.startswith("不") else pat[1:]
        if pos in a and neg in b:
            return True
        if neg in a and pos in b:
            return True
    return False


def _text_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity (0-1) for conflict detection."""
    norm = lambda t: set(t.strip().lower().replace("。！？!?", "").replace("，", " ").replace(" ", "").split("："))
    tokens_a, tokens_b = norm(a), norm(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def run_supersede(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply supersede and contradiction governance.

    Handles:
      - explicit ``supersedes`` (manual assertion)
      - duplicate fingerprint (same text_hash)
      - semantic contradiction (same scope/owner, conflicting text)
      - rank decay for superseded/contradicted memories

    ``limit`` and ``offset`` support pagination. Returns ``next_offset=0`` when done.
    """
    data = data or {}
    dry_run = bool(data.get("dry_run"))
    actor = data.get("actor_role") or "mem-supersede"
    limit = int(data.get("limit") or 5000)
    offset = int(data.get("offset") or 0)
    conflict_threshold = float(data.get("conflict_threshold") or 0.3)
    conn = _connect()
    changes: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT * FROM memories WHERE COALESCE(deleted,0)=0 ORDER BY created_at ASC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        by_id = {m["memory_id"]: m for m in memories}
        latest_by_fingerprint: dict[tuple[Any, ...], dict[str, Any]] = {}
        # Phase 1: explicit supersedes + fingerprint dedup
        for memory in memories:
            if memory.get("protected"):
                continue
            explicit = memory.get("supersedes")
            explicit_ids = [item for item in _listify(explicit) if item in by_id]
            for old_id in explicit_ids:
                old = by_id[old_id]
                if old.get("protected"):
                    continue
                changes.append({"memory_id": old_id, "superseded_by": memory["memory_id"], "reason": "explicit_supersedes"})
            fingerprint = (memory.get("owner_role"), memory.get("project_id"), memory.get("scope"), memory.get("text_hash"))
            previous = latest_by_fingerprint.get(fingerprint)
            if previous and previous.get("memory_id") != memory.get("memory_id") and not previous.get("protected"):
                changes.append({"memory_id": previous["memory_id"], "superseded_by": memory["memory_id"], "reason": "duplicate_fingerprint"})
            latest_by_fingerprint[fingerprint] = memory
        # Phase 2: semantic contradiction detection (same scope+role, different text_hash)
        grouped_by_scope: dict[str, list[dict[str, Any]]] = {}
        for memory in memories:
            if memory.get("protected") or memory.get("temperature") in ("decayed", "cold"):
                continue
            scope_key = f"{memory.get('owner_role')}:{memory.get('project_id')}:{memory.get('scope', 'default')}"
            grouped_by_scope.setdefault(scope_key, []).append(memory)
        for scope_key, scope_memories in grouped_by_scope.items():
            for i in range(len(scope_memories)):
                for j in range(i + 1, len(scope_memories)):
                    a, b = scope_memories[i], scope_memories[j]
                    if a.get("text_hash") == b.get("text_hash"):
                        continue  # already handled by fingerprint
                    text_a, text_b = (a.get("text") or ""), (b.get("text") or "")
                    if not text_a or not text_b:
                        continue
                    sim = _text_similarity(text_a, text_b)
                    if sim < conflict_threshold:
                        continue
                    if _is_text_conflict(text_a, text_b):
                        # Both stay active but get contradiction marker
                        newer, older = (b, a) if (b.get("created_at") or "") > (a.get("created_at") or "") else (a, b)
                        changes.append({"memory_id": older["memory_id"], "superseded_by": newer["memory_id"], "reason": "contradiction_detected", "conflict_type": "semantic"})
        # Phase 3: rank decay — lower importance for superseded/contradicted memories
        decay_targets = []
        for memory in memories:
            if memory.get("status") in ("superseded", "contradicted", "deprecated"):
                if (memory.get("importance") or 50) > 5:
                    decay_targets.append(memory["memory_id"])
        if not dry_run:
            for mid in decay_targets:
                conn.execute("UPDATE memories SET importance=MAX(importance-5, 5) WHERE memory_id=?", (mid,))
        # De-duplicate changes while preserving first reason
        deduped: dict[str, dict[str, Any]] = {}
        for change in changes:
            deduped.setdefault(change["memory_id"], change)
        changes = list(deduped.values())
        if not dry_run:
            for change in changes:
                old = by_id.get(change["memory_id"])
                if not old:
                    continue
                status = "contradicted" if change.get("reason") == "contradiction_detected" else "superseded"
                after = {**old, "status": status, "temperature": "cold", "archived": True, "updated_at": now_iso(), "metadata": {**(old.get("metadata") or {}), "superseded_by": change["superseded_by"], "supersede_reason": change["reason"]}}
                conn.execute(
                    "UPDATE memories SET status=?, temperature='cold', archived=1, metadata_json=?, updated_at=? WHERE memory_id=?",
                    (status, json.dumps(after["metadata"], ensure_ascii=False), now_iso(), change["memory_id"]),
                )
                conn.execute(
                    "INSERT INTO memory_history (history_id, memory_id, action, actor_role, before_json, after_json, created_at) VALUES (?, ?, 'supersede', ?, ?, ?, ?)",
                    (f"hist-{uuid.uuid4()}", change["memory_id"], actor, json.dumps(old, ensure_ascii=False), json.dumps(after, ensure_ascii=False), now_iso()),
                )
        next_offset = offset + limit if len(rows) == limit else 0
        return {"accepted": True, "dry_run": dry_run, "superseded": len(changes), "decayed_importance": len(decay_targets), "changes": changes, "next_offset": next_offset, "has_more": len(rows) == limit}
    finally:
        conn.close()

def governance_dashboard(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a compact MEM governance/audit dashboard."""
    data = data or {}
    project_id = data.get("project_id")
    params: list[Any] = []
    where = "WHERE COALESCE(deleted,0)=0"
    if project_id:
        where += " AND project_id=?"
        params.append(project_id)
    conn = _connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM memories {where}", params).fetchone()["n"]
        def grouped(column: str) -> dict[str, int]:
            return {str(row[column] or "unknown"): int(row["n"]) for row in conn.execute(f"SELECT {column}, COUNT(*) AS n FROM memories {where} GROUP BY {column}", params).fetchall()}
        candidates = [
            _row_to_memory(row)
            for row in conn.execute(
                f"SELECT * FROM memories {where} AND status='candidate' ORDER BY created_at DESC LIMIT 20",
                params,
            ).fetchall()
        ]
        recent_history = [
            {"memory_id": row["memory_id"], "action": row["action"], "actor_role": row["actor_role"], "created_at": row["created_at"]}
            for row in conn.execute("SELECT memory_id, action, actor_role, created_at FROM memory_history ORDER BY created_at DESC LIMIT 20").fetchall()
        ]
        return {"total": total, "by_layer": grouped("layer"), "by_temperature": grouped("temperature"), "by_memory_type": grouped("memory_type"), "by_status": grouped("status"), "candidates": candidates, "recent_history": recent_history}
    finally:
        conn.close()


def update_memory(memory_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
    actor_role = data.get("actor_role") or data.get("caller_role")
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            return None
        before = _row_to_memory(row)
        if actor_role and not _visible(before, actor_role, before.get("project_id"), include_candidates=True):
            return None
        updates: dict[str, Any] = {}
        for key in [
            "text",
            "summary",
            "layer",
            "scope",
            "visibility",
            "memory_type",
            "status",
            "confidence",
            "expires_at",
            "temperature",
            "importance",
            "protected",
            "supersedes",
        ]:
            if key in data:
                updates[key] = data[key]
        if "allowed_roles" in data or "acl" in data:
            roles = _listify(data.get("allowed_roles") or data.get("acl"))
            updates["allowed_roles_json"] = _json_dumps(roles)
            updates["acl_json"] = _json_dumps(roles)
        if "metadata" in data:
            updates["metadata_json"] = _json_dumps(data.get("metadata") or {})
        if "text" in updates:
            updates["text_hash"] = _text_hash(updates["text"])
            updates["embedding_status"] = "pending"
        updates["updated_at"] = now_iso()
        set_clause = ", ".join(f"{key}=?" for key in updates)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"UPDATE memories SET {set_clause} WHERE memory_id=?", [*updates.values(), memory_id])
        after_row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        after = _row_to_memory(after_row)
        conn.execute(
            "INSERT INTO memory_history (history_id, memory_id, action, actor_role, before_json, after_json, created_at) VALUES (?, ?, 'update', ?, ?, ?, ?)",
            (f"hist-{uuid.uuid4()}", memory_id, actor_role, json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False), now_iso()),
        )
        conn.execute("COMMIT")
        return after
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    finally:
        conn.close()


def delete_memory(memory_id: str, data: dict[str, Any], hard: bool = False) -> bool:
    actor_role = data.get("actor_role") or data.get("caller_role")
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            return False
        before = _row_to_memory(row)
        if hard and not _is_admin(actor_role):
            return False
        if actor_role and not (_visible(before, actor_role, before.get("project_id"), include_candidates=True) or _is_admin(actor_role)):
            return False
        conn.execute("BEGIN IMMEDIATE")
        if hard:
            conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
            action = "hard_delete"
            after = {}
        else:
            conn.execute(
                "UPDATE memories SET deleted=1, deleted_at=?, deleted_by=?, delete_reason=?, updated_at=? WHERE memory_id=?",
                (now_iso(), actor_role, data.get("reason"), now_iso(), memory_id),
            )
            action = "delete"
            after = {**before, "deleted": True}
        conn.execute(
            "INSERT INTO memory_history (history_id, memory_id, action, actor_role, before_json, after_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"hist-{uuid.uuid4()}", memory_id, action, actor_role, json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False), now_iso()),
        )
        conn.execute("COMMIT")
        return True
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    finally:
        conn.close()


def memory_history(memory_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM memory_history WHERE memory_id=? ORDER BY created_at", (memory_id,)).fetchall()
        return [
            {
                "history_id": row["history_id"],
                "memory_id": row["memory_id"],
                "action": row["action"],
                "actor_role": row["actor_role"],
                "before": _json_loads(row["before_json"], {}),
                "after": _json_loads(row["after_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def _retry_delay_seconds(attempts: int) -> int:
    base = int(os.environ.get("NTN_MEM_PROVIDER_RETRY_BASE_SECONDS", "30"))
    cap = int(os.environ.get("NTN_MEM_PROVIDER_RETRY_MAX_SECONDS", "3600"))
    return min(cap, base * (2 ** max(attempts - 1, 0)))


def _future_iso(seconds: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ms = dt.microsecond // 1000
    return dt.strftime(f"%Y-%m-%dT%H:%M:%S.{ms:03d}Z")


def _provider_max_attempts() -> int:
    return int(os.environ.get("NTN_MEM_PROVIDER_MAX_ATTEMPTS", "3"))


def get_job(memory_job_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM memory_jobs WHERE memory_job_id=?", (memory_job_id,)).fetchone()
        if row is None:
            return None
        return {
            "status": row["status"],
            "memory_ids": json.loads(row["memory_ids_json"]),
            "provider": row["provider"],
            "provider_event_id": row["provider_event_id"] if "provider_event_id" in row.keys() else None,
            "error": _json_loads(row["error_json"] if "error_json" in row.keys() else None, None),
            "attempts": row["attempts"] if "attempts" in row.keys() else 0,
            "next_retry_at": row["next_retry_at"] if "next_retry_at" in row.keys() else None,
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def retry_pending_jobs(provider_client: Any | None = None, limit: int = 100) -> dict[str, int]:
    conn = _connect()
    retried = succeeded = failed = 0
    try:
        rows = conn.execute(
            "SELECT j.*, m.* FROM memory_jobs j JOIN memories m ON m.memory_id = json_extract(j.memory_ids_json, '$[0]') "
            "WHERE j.status = 'PENDING_PROVIDER' AND (j.next_retry_at IS NULL OR j.next_retry_at <= ?) "
            "ORDER BY j.created_at LIMIT ?",
            (now_iso(), limit),
        ).fetchall()
        for row in rows:
            retried += 1
            memory = _row_to_memory(row)
            provider_result = write_provider(memory, memory_id=memory["memory_id"], client=provider_client)
            attempts = int(row["attempts"] or 0) + 1
            if provider_result.status == "SUCCEEDED":
                job_status = "SUCCEEDED"
                next_retry_at = None
            elif attempts >= _provider_max_attempts():
                job_status = "FAILED_PROVIDER"
                next_retry_at = None
            else:
                job_status = "PENDING_PROVIDER"
                next_retry_at = _future_iso(_retry_delay_seconds(attempts))
            conn.execute(
                "UPDATE memory_jobs SET status=?, provider_event_id=?, error_json=?, attempts=?, next_retry_at=?, updated_at=? WHERE memory_job_id=?",
                (
                    job_status,
                    provider_result.provider_event_id,
                    json.dumps(provider_result.error, ensure_ascii=False),
                    attempts,
                    next_retry_at,
                    now_iso(),
                    row["memory_job_id"],
                ),
            )
            if job_status == "SUCCEEDED":
                succeeded += 1
                conn.execute(
                    "UPDATE memories SET provider=?, provider_memory_id=?, updated_at=? WHERE memory_id=?",
                    (provider_result.provider, provider_result.provider_memory_id, now_iso(), memory["memory_id"]),
                )
            else:
                failed += 1
    finally:
        conn.close()
    return {"retried": retried, "succeeded": succeeded, "failed": failed}


def consume_event(data: dict[str, Any]) -> dict[str, Any]:
    """Consume one SQL-http event and create task_context/candidate memories.

    This is the lightweight P2 bridge; a daemon can call it after polling
    SQL-http /events. It avoids direct state.db reads and keeps writes inside MEM.
    """
    event_type = data.get("event_type") or data.get("type")
    task_id = data.get("task_id")
    project_id = data.get("project_id")
    role = data.get("role") or data.get("target_role")
    details = data.get("details") or data.get("payload") or {}
    text = data.get("text") or details.get("summary") or details.get("message") or ""
    if not text:
        return {"accepted": False, "reason": "empty_event_text"}
    scope = "task_context" if event_type != "task_finished" else "shared"
    status = "candidate" if event_type == "task_finished" else "verified"
    result = add_memory(
        {
            "source": "sql_http_event",
            "event_id": data.get("event_id"),
            "task_id": task_id,
            "role": role,
            "owner_role": role,
            "source_role": role,
            "project_id": project_id,
            "layer": "mid_term",
            "scope": scope,
            "visibility": "project_roles" if scope == "task_context" else "shared_acl",
            "allowed_roles": data.get("allowed_roles") or ROLE_KEYS,
            "memory_type": "handoff" if scope == "task_context" else "task_result",
            "status": status,
            "confidence": "observed",
            "text": text,
            "metadata": {"event_type": event_type, "details": details},
        }
    )
    return {"accepted": True, **result}


def _log_handler(environ, start_response, operation: str, handler):
    """Wrap a route handler with timing + success/error logging."""
    t0 = _time.time()
    try:
        result = handler(environ, start_response)
        dur = int((_time.time() - t0) * 1000)
        log_info(module="app", operation=operation,
                 summary=f"{operation} OK", duration_ms=dur)
        return result
    except Exception as exc:
        dur = int((_time.time() - t0) * 1000)
        snippet = ""
        try:
            body = environ.get("wsgi.input", b"").read(500)
            snippet = body.decode("utf-8", errors="replace")[:500]
            environ["wsgi.input"] = None
        except Exception:
            pass
        log_error(module="app", operation=operation,
                  summary=f"{operation} failed after {dur}ms",
                  exc=exc, request_snippet=snippet)
        raise


def application(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    query = parse_qs(environ.get("QUERY_STRING", ""))

    t0 = _time.time()
    op = f"{method} {path}"

    if method == "GET" and path == "/health":
        return _json_response(start_response, "200 OK", {"status": "ok", "service": "ntn-mem"})

    if not _authorized(environ, "NTN_MEM_TOKEN"):
        return _unauthorized(start_response)

    # ---- Log query endpoints ----
    if method == "GET" and path == "/v1/system/logs/errors":
        since = int(query.get("since_hours", ["24"])[0])
        op_filter = query.get("operation", [None])[0]
        return _json_response(start_response, "200 OK",
                               {"errors": query_errors(since_hours=since, operation=op_filter)})
    if method == "GET" and path == "/v1/system/logs/stats":
        since = int(query.get("since_hours", ["24"])[0])
        log_info(module="app", operation=op,
                 summary=f"logs/stats?since={since}", duration_ms=int((_time.time()-t0)*1000))
        return _json_response(start_response, "200 OK",
                               query_error_stats(since_hours=since))

    if method == "POST" and path == "/v1/agents/memory/push":
        try:
            result = agent_push_memory(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"push OK", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "202 Accepted", result)
        except ValueError as exc:
            code = str(exc) if str(exc) in {"AGENT_KEY_REQUIRED", "TEXT_REQUIRED"} else "BAD_JSON"
            log_warn(module="app", operation=op,
                     summary=f"push ValueError: {code}", exc=exc)
            return _json_response(start_response, "400 Bad Request", {"error": {"code": code}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="push failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "PUSH_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/agents/memory/pull":
        try:
            result = agent_pull_memory(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"pull OK", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError as exc:
            code = str(exc) if str(exc) in {"AGENT_KEY_REQUIRED"} else "BAD_JSON"
            log_warn(module="app", operation=op,
                     summary=f"pull ValueError: {code}", exc=exc)
            return _json_response(start_response, "400 Bad Request", {"error": {"code": code}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="pull failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "PULL_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/route":
        try:
            result = route_memory(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"route OK", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="route ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
    if method == "POST" and path == "/v1/memory/search":
        try:
            result = search_memory(_read_json(environ))
            n = len(result.get("results", []))
            log_info(module="app", operation=op,
                     summary=f"search: {n} results", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="search ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary=f"search failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "SEARCH_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/add":
        try:
            result = add_memory(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"add: memory_id={result.get('memory_id','?')[:16]}...",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "202 Accepted", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="add ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary=f"add failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "ADD_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/profile/update":
        try:
            t0 = _time.time()
            from .profile_distill import route_profile_update
            result = route_profile_update(environ, start_response)
            log_info(module="app", operation=op,
                     summary=f"profile update OK",
                     duration_ms=int((_time.time() - t0) * 1000))
            return result
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Profile update failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "PROFILE_UPDATE_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/memory/profile/updates":
        try:
            from .profile_distill import route_get_updates
            return route_get_updates(environ, start_response)
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Profile updates fetch failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "PROFILE_UPDATES_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/recollect":
        try:
            data = _read_json(environ)
            result = _recollect(
                query=data.get("query", ""),
                agent_key=data.get("agent_key") or data.get("caller_role"),
                limit=int(data.get("limit", 5)),
            )
            log_info(module="app", operation=op,
                     summary=f"recollect OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="recollect ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Recollect failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "RECOLLECT_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/recollect/detail":
        try:
            data = _read_json(environ)
            result = _recollect_detail(
                query=data.get("query", ""),
                agent_key=data.get("agent_key") or data.get("caller_role"),
                detail=True,
                memory_ids=data.get("memory_ids"),
                archive_filter=data.get("archive_filter"),
                limit=int(data.get("limit", 5)),
            )
            log_info(module="app", operation=op,
                     summary=f"recollect/detail OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="recollect/detail ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Recollect detail failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "RECOLLECT_DETAIL_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/events/consume":
        try:
            result = consume_event(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"events/consume OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "202 Accepted", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="events/consume ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
    if method == "POST" and path == "/v1/memory/governance/lifecycle/run":
        try:
            result = run_lifecycle(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"lifecycle/run OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="lifecycle/run ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Lifecycle run failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "LIFECYCLE_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/governance/supersede/run":
        try:
            result = run_supersede(_read_json(environ))
            log_info(module="app", operation=op,
                     summary=f"supersede/run OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="supersede/run ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Supersede run failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "SUPERSEDE_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/governance/dashboard":
        try:
            result = governance_dashboard(_read_json(environ))
            log_info(module="app", operation=op,
                     summary="dashboard OK",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except ValueError:
            log_warn(module="app", operation=op, summary="dashboard ValueError: BAD_JSON")
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "BAD_JSON"}})
    if method == "GET" and path == "/v1/memory/governance/dashboard":
        result = governance_dashboard({"project_id": query.get("project_id", [None])[0]})
        log_info(module="app", operation=op,
                 summary="dashboard GET OK",
                 duration_ms=int((_time.time()-t0)*1000))
        return _json_response(start_response, "200 OK", result)
    if method == "GET" and path == "/v1/system/lifecycle-daemon/state":
        from .lifecycle_daemon import read_lifecycle_state
        return _json_response(start_response, "200 OK", read_lifecycle_state())
    if method == "POST" and path == "/v1/system/lifecycle-daemon/trigger":
        from .lifecycle_daemon import LifecycleDaemon
        try:
            t0 = _time.time()
            result = LifecycleDaemon().run_once()
            log_info(module="app", operation=op,
                     summary="Lifecycle daemon triggered",
                     duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            log_error(module="app", operation=op,
                      summary="Lifecycle daemon run failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "LIFECYCLE_TRIGGER_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/system/procedural-to-skill/state":
        from .procedural_to_skill import read_generation_state
        return _json_response(start_response, "200 OK", read_generation_state())
    if method == "POST" and path == "/v1/system/procedural-to-skill/trigger":
        from .procedural_to_skill import generate_skills
        try:
            t0 = _time.time()
            result = generate_skills()
            log_info(module="app", operation="skill_trigger",
                     summary="Skill generation triggered", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            log_error(module="app", operation="skill_trigger",
                      summary="Skill generation failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "SKILL_TRIGGER_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/system/shared-kb-ingestion/state":
        from .shared_kb_ingestion import read_ingest_state
        return _json_response(start_response, "200 OK", read_ingest_state())
    if method == "POST" and path == "/v1/system/shared-kb-ingestion/ingest":
        from .shared_kb_ingestion import ingest_documents
        try:
            t0 = _time.time()
            result = ingest_documents()
            log_info(module="app", operation="kb_ingest",
                     summary="KB ingestion triggered", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            log_error(module="app", operation="kb_ingest",
                      summary="KB ingestion failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "INGEST_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/system/graph/state":
        from .graph_index import read_graph_state
        return _json_response(start_response, "200 OK", read_graph_state())
    if method == "POST" and path == "/v1/system/graph/rebuild":
        from .graph_index import rebuild_graph
        data = _read_json(environ) if environ.get("CONTENT_LENGTH") else {}
        try:
            t0 = _time.time()
            result = rebuild_graph(
                limit=int(data.get("limit", 5000)),
                force_rebuild=bool(data.get("force_rebuild", False)),
            )
            log_info(module="app", operation="graph_rebuild",
                     summary=f"Graph rebuilt", duration_ms=int((_time.time()-t0)*1000))
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            log_error(module="app", operation="graph_rebuild",
                      summary="Graph rebuild failed", exc=exc)
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "GRAPH_REBUILD_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/system/graph/stats":
        from .graph_index import query_graph_stats
        return _json_response(start_response, "200 OK", query_graph_stats())
    if method == "POST" and path == "/v1/system/graph/query":
        from .graph_index import query_graph_neighborhood
        data = _read_json(environ)
        try:
            entity = data.get("entity")
            if not entity:
                return _json_response(start_response, "400 Bad Request", {"error": {"code": "ENTITY_REQUIRED"}})
            result = query_graph_neighborhood(entity, max_depth=int(data.get("max_depth", 2)))
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "GRAPH_QUERY_FAILED", "detail": str(exc)}})
    if method == "GET" and path == "/v1/system/graph/entities":
        from .graph_index import query_graph_entities
        result = query_graph_entities(
            limit=int(query.get("limit", [100])[0]),
            min_weight=int(query.get("min_weight", [1])[0]),
        )
        return _json_response(start_response, "200 OK", result)
    # ---- Governance Dashboard endpoints ----
    if method == "GET" and path == "/v1/system/dashboard/summary":
        from .governance_dashboard import dashboard_summary
        since_raw = query.get("since_days", [None])[0]
        since_days = int(since_raw) if since_raw is not None else None
        return _json_response(start_response, "200 OK", dashboard_summary(since_days=since_days))
    if method == "GET" and path.startswith("/v1/system/dashboard/agent/"):
        agent_key = path.removeprefix("/v1/system/dashboard/agent/")
        from .governance_dashboard import dashboard_agent_audit
        result = dashboard_agent_audit(
            agent_key,
            limit=int(query.get("limit", [100])[0]),
            include_deleted=query.get("include_deleted", ["false"])[0].lower() == "true",
        )
        return _json_response(start_response, "200 OK", result)
    if method == "GET" and path == "/v1/system/dashboard/matrix":
        from .governance_dashboard import dashboard_layer_temperature_matrix
        since_raw = query.get("since_days", [None])[0]
        since_days = int(since_raw) if since_raw is not None else None
        return _json_response(start_response, "200 OK", dashboard_layer_temperature_matrix(since_days=since_days))
    if method == "GET" and path == "/v1/system/dashboard/stale":
        from .governance_dashboard import dashboard_stale_memories
        result = dashboard_stale_memories(
            not_recalled_days=int(query.get("days", ["30"])[0]),
            limit=int(query.get("limit", ["50"])[0]),
        )
        return _json_response(start_response, "200 OK", result)
    if method == "GET" and path.startswith("/v1/memory/jobs/"):
        memory_job_id = path.removeprefix("/v1/memory/jobs/")
        job = get_job(memory_job_id)
        if job is None:
            return _json_response(start_response, "404 Not Found", {"error": {"code": "MEMORY_JOB_NOT_FOUND"}})
        return _json_response(start_response, "200 OK", job)
    # ---- Archives: unified catalog of all MEM repositories ----
    if path == "/v1/memory/archives" and method == "GET":
        try:
            from .manager_knowledge import list_kbs
            from .manager_private import list_registered_agents
            kbs = list_kbs()
            archives = []
            for kb in kbs:
                entry = {
                    "id": kb["kb_id"],
                    "name": kb["name"],
                    "description": kb.get("description", ""),
                    "project_id": kb["project_id"],
                    "type": "knowledge_base" if "manual" in (kb.get("tags") or []) else "experience_base" if "auto_extraction" in (kb.get("tags") or []) else "knowledge_base",
                    "scope": "shared",
                    "document_count": kb.get("document_count", 0),
                    "access": "all agents (read-only via shared_knowledge_projects)",
                    "how_to_write": f"POST /v1/memory/add {{project_id:'{kb['project_id']}', ...}}" if "manual" in (kb.get("tags") or []) else "system auto (not manually writable)",
                    "how_to_read": "automatically included in agent_pull",
                    "auto_maintained": "auto_extraction" in (kb.get("tags") or []),
                    "tags": kb.get("tags", []),
                }
                archives.append(entry)
            return _json_response(start_response, "200 OK", {"archives": archives, "total": len(archives)})
        except Exception as exc:
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "ARCHIVES_FAILED", "detail": str(exc)}})
    if method == "POST" and path == "/v1/memory/archive/cold":
        """"冷归档：把 cold + superseded >7天的记忆移到 cold_archive 表"""
        try:
            conn = _connect()
            now = now_iso()
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE temperature='cold' AND status='superseded' AND created_at < ?
                   AND memory_id NOT IN (SELECT memory_id FROM cold_archive)""",
                (cutoff,),
            ).fetchall()
            archived = 0
            for row in rows:
                d = dict(row)
                import json as _json
                conn.execute(
                    """INSERT OR IGNORE INTO cold_archive
                       (memory_id, text, role, project_id, layer, scope,
                        temperature, importance, protected, supersedes,
                        metadata_json, created_at, archived_at, archive_reason, original_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        d["memory_id"], d["text"], d.get("role"), d.get("project_id"),
                        d.get("layer"), d.get("scope"), d.get("temperature"),
                        d.get("importance"), d.get("protected"), d.get("supersedes"),
                        d.get("metadata_json", "{}"), d["created_at"], now,
                        "cold_storage_auto", _json.dumps(d),
                    ),
                )
                conn.execute("DELETE FROM memories WHERE memory_id=?", (d["memory_id"],))
                archived += 1
            conn.commit()
            conn.close()
            return _json_response(start_response, "200 OK", {
                "archived": archived,
                "archived_at": now,
                "cutoff": cutoff,
            })
        except Exception as exc:
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "COLD_ARCHIVE_FAILED", "detail": str(exc)}})
    if path.startswith("/v1/memory/"):
        suffix = path.removeprefix("/v1/memory/")
        if suffix.endswith("/history") and method == "GET":
            memory_id = suffix.removesuffix("/history")
            return _json_response(start_response, "200 OK", {"history": memory_history(memory_id)})
        memory_id = suffix
        caller_role = query.get("caller_role", [None])[0]
        if method == "GET":
            memory = get_memory(memory_id, caller_role=caller_role)
            if memory is None:
                return _json_response(start_response, "404 Not Found", {"error": {"code": "MEMORY_NOT_FOUND"}})
            return _json_response(start_response, "200 OK", memory)
        if method == "PATCH":
            memory = update_memory(memory_id, _read_json(environ))
            if memory is None:
                return _json_response(start_response, "404 Not Found", {"error": {"code": "MEMORY_NOT_FOUND"}})
            return _json_response(start_response, "200 OK", memory)
        if method == "DELETE":
            hard = query.get("hard", ["false"])[0].lower() == "true" or path.startswith("/v1/admin/")
            ok = delete_memory(memory_id, _read_json(environ), hard=hard)
            if not ok:
                return _json_response(start_response, "404 Not Found", {"error": {"code": "MEMORY_NOT_FOUND"}})
            return _json_response(start_response, "200 OK", {"deleted": True, "hard": hard})
    if path.startswith("/v1/admin/memory/") and method == "DELETE":
        memory_id = path.removeprefix("/v1/admin/memory/")
        hard = query.get("hard", ["false"])[0].lower() == "true"
        ok = delete_memory(memory_id, _read_json(environ), hard=hard)
        if not ok:
            return _json_response(start_response, "404 Not Found", {"error": {"code": "MEMORY_NOT_FOUND"}})
        return _json_response(start_response, "200 OK", {"deleted": True, "hard": hard})
    # ---- Private Memory Manager routes ----
    if path == "/v1/manage/private/agents" and method == "GET":
        from .manager_private import list_registered_agents
        return _json_response(start_response, "200 OK", {"agents": list_registered_agents()})
    if path.startswith("/v1/manage/private/") and method == "DELETE":
        suffix = path.removeprefix("/v1/manage/private/")
        from .manager_private import unregister_agent
        result = unregister_agent(suffix)
        status = "404 Not Found" if "error" in result else "200 OK"
        return _json_response(start_response, status, result)
    if path.startswith("/v1/manage/private/") and method == "GET":
        # /v1/manage/private/{agent_key}/memories?status=&layer=&temperature=&page=&per_page=
        suffix = path.removeprefix("/v1/manage/private/")
        if suffix.endswith("/memories"):
            agent_key = suffix.removesuffix("/memories")
            from .manager_private import list_agent_memories
            result = list_agent_memories(
                agent_key,
                status=query.get("status", [None])[0],
                layer=query.get("layer", [None])[0],
                temperature=query.get("temperature", [None])[0],
                memory_type=query.get("memory_type", [None])[0],
                page=int(query.get("page", ["1"])[0]),
                per_page=int(query.get("per_page", ["50"])[0]),
            )
            return _json_response(start_response, "200 OK", result)
    if path.startswith("/v1/manage/private/") and method == "POST":
        suffix = path.removeprefix("/v1/manage/private/")
        if suffix.endswith("/gc"):
            agent_key = suffix.removesuffix("/gc")
            data = _read_json(environ)
            from .manager_private import agent_gc
            result = agent_gc(
                agent_key,
                action=data.get("action", "delete_decayed"),
                dry_run=data.get("dry_run", True),
            )
            return _json_response(start_response, "200 OK", result)
    if path == "/v1/manage/private/gc-all" and method == "POST":
        data = _read_json(environ)
        from .manager_private import gc_all_agents
        result = gc_all_agents(
            action=data.get("action", "delete_decayed"),
            dry_run=data.get("dry_run", True),
        )
        return _json_response(start_response, "200 OK", result)
    # ---- Knowledge Manager routes ----
    if path == "/v1/knowledge" and method == "GET":
        from .manager_knowledge import list_kbs
        return _json_response(start_response, "200 OK", {"knowledge_bases": list_kbs()})
    if path == "/v1/knowledge/schema" and method == "GET":
        from .manager_knowledge import get_kb_schema
        return _json_response(start_response, "200 OK", get_kb_schema())
    if path == "/v1/knowledge/register" and method == "POST":
        from .manager_knowledge import register_kb
        try:
            data = _read_json(environ)
            result = register_kb(
                data["name"],
                description=data.get("description", ""),
                project_id=data["project_id"],
                tags=data.get("tags"),
                owner=data.get("owner"),
            )
            return _json_response(start_response, "201 Created", result)
        except (ValueError, KeyError) as exc:
            return _json_response(start_response, "400 Bad Request", {"error": {"code": str(exc)}})
    if path == "/v1/knowledge/search" and method == "POST":
        from .manager_knowledge import cross_kb_search
        try:
            data = _read_json(environ)
            result = cross_kb_search(
                data.get("query", ""),
                kb_ids=data.get("kb_ids"),
                limit=int(data.get("limit", 10)),
                include_candidates=bool(data.get("include_candidates", False)),
            )
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:
            return _json_response(start_response, "500 Internal Server Error", {"error": {"code": "KNOWLEDGE_SEARCH_FAILED", "detail": str(exc)}})
    if path.startswith("/v1/knowledge/") and method == "GET":
        suffix = path.removeprefix("/v1/knowledge/")
        if suffix and "/" not in suffix:
            from .manager_knowledge import get_kb
            kb = get_kb(suffix)
            if kb is None:
                return _json_response(start_response, "404 Not Found", {"error": {"code": "KB_NOT_FOUND"}})
            return _json_response(start_response, "200 OK", kb)
    if path.startswith("/v1/knowledge/") and method == "DELETE":
        suffix = path.removeprefix("/v1/knowledge/")
        if suffix and "/" not in suffix:
            from .manager_knowledge import deregister_kb
            result = deregister_kb(suffix, hard=query.get("hard", ["false"])[0].lower() == "true")
            return _json_response(start_response, "200 OK", result)
    if path.endswith("/ingest") and method == "POST":
        # /v1/knowledge/{kb_id}/ingest  — fallback to body.kb_id for Python 3.10 compat
        prefix = path.removesuffix("/ingest")
        kb_id = None
        if prefix.startswith("/v1/knowledge/"):
            kb_id = prefix.removeprefix("/v1/knowledge/")
            if not kb_id or "/" in kb_id:
                kb_id = None
        data_body = _read_json(environ)
        if kb_id is None:
            kb_id = data_body.get("kb_id") or query.get("kb_id", [None])[0]
        if not kb_id:
            return _json_response(start_response, "400 Bad Request", {"error": {"code": "KB_ID_REQUIRED", "detail": "kb_id in path (ASCII-only), query (?kb_id=), or body (JSON .kb_id)"}})
        from .manager_knowledge import ingest_documents
        result = ingest_documents(
            kb_id,
            data_body.get("documents", []),
            skip_duplicates=data_body.get("skip_duplicates", True),
            actor_role=data_body.get("actor_role"),
        )
        return _json_response(start_response, "200 OK", result)
    if path.endswith("/gc") and method == "POST":
        # POST /v1/knowledge/{kb_id}/gc — preview, fallback to ?kb_id=
        prefix = path.removesuffix("/gc")
        kb_id = None
        if prefix.startswith("/v1/knowledge/"):
            kb_id = prefix.removeprefix("/v1/knowledge/")
            if not kb_id or "/" in kb_id:
                kb_id = None
        if kb_id is None:
            kb_id = query.get("kb_id", [None])[0]
        if kb_id:
            from .manager_knowledge import kb_gc_preview
            data = _read_json(environ)
            result = kb_gc_preview(
                kb_id,
                stale_days=int(data.get("stale_days", 7)),
            )
            return _json_response(start_response, "200 OK", result)
    if path.endswith("/gc/execute") and method == "POST":
        # POST /v1/knowledge/{kb_id}/gc/execute — delete confirmed, fallback to ?kb_id=
        prefix = path.removesuffix("/gc/execute")
        kb_id = None
        if prefix.startswith("/v1/knowledge/"):
            kb_id = prefix.removeprefix("/v1/knowledge/")
            if not kb_id or "/" in kb_id:
                kb_id = None
        if kb_id is None:
            kb_id = query.get("kb_id", [None])[0]
        if kb_id:
            from .manager_knowledge import kb_gc_execute
            data = _read_json(environ)
            memory_ids = data.get("memory_ids", [])
            if not memory_ids:
                return _json_response(start_response, "400 Bad Request", {"error": {"code": "NO_IDS_PROVIDED"}})
            result = kb_gc_execute(kb_id, memory_ids)
            return _json_response(start_response, "200 OK", result)
    if path.endswith("/reindex") and method == "POST":
        prefix = path.removesuffix("/reindex")
        if prefix.startswith("/v1/knowledge/"):
            kb_id = prefix.removeprefix("/v1/knowledge/")
            from .manager_knowledge import reindex_kb
            result = reindex_kb(kb_id)
            return _json_response(start_response, "200 OK", result)
    # ---- Plugin System (self-service agent onboarding) ----
    PLUGIN_SYSTEM_DIR = "/data/plugin-system"
    if path == "/v1/plugin-system" and method == "GET":
        try:
            entries = []
            for f in sorted(os.listdir(PLUGIN_SYSTEM_DIR)):
                if os.path.isfile(os.path.join(PLUGIN_SYSTEM_DIR, f)):
                    fpath = os.path.join(PLUGIN_SYSTEM_DIR, f)
                    with open(fpath) as fh:
                        content = fh.read()
                    entries.append({"name": f, "size": len(content)})
            return _json_response(start_response, "200 OK", {
                "base_url": os.environ.get("NTN_MEM_PUBLIC_URL", "http://0.0.0.0:8081"),
                "files": entries
            })
        except Exception as exc:
            return _json_response(start_response, "200 OK", {
                "base_url": os.environ.get("NTN_MEM_PUBLIC_URL", "http://0.0.0.0:8081"),
                "files": [],
                "note": f"Plugin system dir not ready: {exc}"
            })
    if path == "/v1/plugin-system/specifications.md" and method == "GET":
        spec_path = os.path.join(PLUGIN_SYSTEM_DIR, "specifications.md")
        try:
            with open(spec_path) as fh:
                content = fh.read()
            headers = [("Content-Type", "text/markdown; charset=utf-8")]
            start_response("200 OK", headers)
            return [content.encode("utf-8")]
        except FileNotFoundError:
            return _json_response(start_response, "404 Not Found", {"error": {"code": "SPEC_NOT_FOUND"}})
    return _json_response(start_response, "404 Not Found", {"error": {"code": "NOT_FOUND"}})
