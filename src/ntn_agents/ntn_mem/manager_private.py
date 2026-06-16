"""Private Memory Manager — agent registration, namespace isolation, bulk ops.

This module adds system-level management on top of the lower-level storage.
It does NOT bypass or duplicate ``add_memory()`` / ``search_memory()`` /
``delete_memory()``; every storage mutation delegates to the existing API.

Agent Registry
--------------
Each agent gets its own ``project_id`` on first registration. Subsequent pushes
under ``scope=role_private`` are automatically scoped to that project_id.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# Registry lives in its own DB so schema changes don't couple with mem.db.
REGISTRY_DB = os.environ.get("NTN_MEM_REGISTRY_DB", "/data/registry.db")


def _reg_connect() -> sqlite3.Connection:
    target = REGISTRY_DB
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_registry (
            agent_key TEXT PRIMARY KEY,
            project_id TEXT NOT NULL UNIQUE,
            owner TEXT,
            registered_at TEXT NOT NULL,
            last_active_at TEXT,
            status TEXT DEFAULT 'active'
        );
        """
    )
    return conn


# Circular-import-safe lazy import for count queries.


def _mem_count(project_id: str) -> int:
    from .app import _db_path

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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_agent_registered(agent_key: str, owner: str | None = None) -> str:
    """Return the project_id for *agent_key*, creating a registration if missing.

    Side-effect: sets last_active_at on every call so we can detect stale agents.
    """
    conn = _reg_connect()
    try:
        row = conn.execute(
            "SELECT project_id FROM agent_registry WHERE agent_key=?",
            (agent_key,),
        ).fetchone()
        if row is not None:
            project_id = row["project_id"]
            conn.execute(
                "UPDATE agent_registry SET last_active_at=? WHERE agent_key=?",
                (_now_iso(), agent_key),
            )
            return str(project_id)

        # First registration — derive project_id from agent_key.
        project_id = f"private-{agent_key}"
        conn.execute(
            "INSERT OR IGNORE INTO agent_registry "
            "(agent_key, project_id, owner, registered_at, last_active_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'active')",
            (agent_key, project_id, owner or agent_key, _now_iso(), _now_iso()),
        )
        # If the above INSERT succeeded, project_id is ours.
        row = conn.execute(
            "SELECT project_id FROM agent_registry WHERE agent_key=?",
            (agent_key,),
        ).fetchone()
        return str(row["project_id"]) if row else project_id
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Query / list agents
# --------------------------------------------------------------------------


def list_registered_agents() -> list[dict[str, Any]]:
    conn = _reg_connect()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_registry ORDER BY registered_at DESC"
        ).fetchall()
        agents: list[dict[str, Any]] = []
        for row in rows:
            agents.append(
                {
                    "agent_key": row["agent_key"],
                    "project_id": row["project_id"],
                    "owner": row["owner"],
                    "registered_at": row["registered_at"],
                    "last_active_at": row["last_active_at"],
                    "status": row["status"],
                    "memory_count": _mem_count(row["project_id"]),
                }
            )
        return agents
    finally:
        conn.close()


def get_agent(agent_key: str) -> dict[str, Any] | None:
    conn = _reg_connect()
    try:
        row = conn.execute(
            "SELECT * FROM agent_registry WHERE agent_key=?", (agent_key,)
        ).fetchone()
        if row is None:
            return None
        return {
            "agent_key": row["agent_key"],
            "project_id": row["project_id"],
            "owner": row["owner"],
            "registered_at": row["registered_at"],
            "last_active_at": row["last_active_at"],
            "status": row["status"],
            "memory_count": _mem_count(row["project_id"]),
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Bulk operations
# --------------------------------------------------------------------------


def _mem_conn() -> sqlite3.Connection | None:
    from .app import _db_path

    db_path = _db_path()
    if not db_path:
        return None
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def agent_gc(
    agent_key: str,
    *,
    action: str = "delete_decayed",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Bulk GC for a single agent's private memories.

    Supported actions:
      - archive_expired    soft-delete expired memories
      - delete_decayed     hard-delete decayed-temperature memories
      - downgrade_unrecalled  set temperature=cold for memories not recalled in 90 days
    """
    profile = get_agent(agent_key)
    if profile is None:
        return {"error": "AGENT_NOT_FOUND", "agent_key": agent_key}

    project_id = profile["project_id"]
    conn = _mem_conn()
    if conn is None:
        return {"error": "MEM_DB_UNAVAILABLE", "agent_key": agent_key}

    try:
        now_iso = _now_iso()
        rows: list[sqlite3.Row] = []

        if action == "archive_expired":
            rows = conn.execute(
                "SELECT memory_id, text FROM memories WHERE project_id=? "
                "AND COALESCE(deleted,0)=0 AND expires_at IS NOT NULL AND expires_at < ?",
                (project_id, now_iso),
            ).fetchall()
            if not dry_run:
                for row in rows:
                    conn.execute(
                        "UPDATE memories SET deleted=1, deleted_at=?, deleted_by=? "
                        "WHERE memory_id=?",
                        (now_iso, agent_key, row["memory_id"]),
                    )

        elif action == "delete_decayed":
            rows = conn.execute(
                "SELECT memory_id, text FROM memories WHERE project_id=? "
                "AND COALESCE(deleted,0)=0 AND temperature='decayed'",
                (project_id,),
            ).fetchall()
            if not dry_run:
                for row in rows:
                    conn.execute(
                        "DELETE FROM memories WHERE memory_id=?", (row["memory_id"],)
                    )
                    conn.execute(
                        "UPDATE memory_history SET action='hard_deleted' "
                        "WHERE memory_id=? AND action='delete'",
                        (row["memory_id"],),
                    )

        elif action == "downgrade_unrecalled":
            old_cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - 90 * 86400),
            )
            rows = conn.execute(
                "SELECT memory_id, text FROM memories WHERE project_id=? "
                "AND COALESCE(deleted,0)=0 AND temperature='warm' "
                "AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
                (project_id, old_cutoff),
            ).fetchall()
            if not dry_run:
                for row in rows:
                    conn.execute(
                        "UPDATE memories SET temperature='cold', updated_at=? "
                        "WHERE memory_id=?",
                        (now_iso, row["memory_id"]),
                    )

        else:
            return {"error": "UNKNOWN_ACTION", "action": action}

        return {
            "action": action,
            "agent_key": agent_key,
            "project_id": project_id,
            "affected_count": len(rows),
            "dry_run": dry_run,
            "affected_memory_ids": [r["memory_id"] for r in rows],
        }
    finally:
        conn.close()


def unregister_agent(agent_key: str) -> dict[str, Any]:
    """Delete an agent from the registry. Does NOT touch the agent's memories."""
    profile = get_agent(agent_key)
    if profile is None:
        return {"error": "AGENT_NOT_FOUND", "agent_key": agent_key}

    conn = _reg_connect()
    try:
        conn.execute("DELETE FROM agent_registry WHERE agent_key=?", (agent_key,))
        conn.commit()
        return {
            "agent_key": agent_key,
            "project_id": profile["project_id"],
            "deleted": True,
        }
    finally:
        conn.close()


def gc_all_agents(
    *, action: str = "delete_decayed", dry_run: bool = True
) -> dict[str, Any]:
    """Run the same GC action across every registered agent."""
    agents = list_registered_agents()
    per_agent: dict[str, dict[str, Any]] = {}
    total = 0
    for agent in agents:
        result = agent_gc(agent["agent_key"], action=action, dry_run=dry_run)
        per_agent[agent["agent_key"]] = result
        total += result.get("affected_count", 0)
    return {"action": action, "per_agent": per_agent, "total_affected": total, "dry_run": dry_run}


# --------------------------------------------------------------------------
# Agent memory listing (paginated)
# --------------------------------------------------------------------------


def list_agent_memories(
    agent_key: str,
    *,
    status: str | None = None,
    layer: str | None = None,
    temperature: str | None = None,
    memory_type: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict[str, Any]:
    profile = get_agent(agent_key)
    if profile is None:
        return {"error": "AGENT_NOT_FOUND", "agent_key": agent_key}

    project_id = profile["project_id"]
    conn = _mem_conn()
    if conn is None:
        return {"error": "MEM_DB_UNAVAILABLE", "agent_key": agent_key}

    try:
        where = "WHERE project_id=? AND COALESCE(deleted,0)=0"
        params: list[Any] = [project_id]

        if status:
            where += " AND status=?"
            params.append(status)
        if layer:
            where += " AND layer=?"
            params.append(layer)
        if temperature:
            where += " AND temperature=?"
            params.append(temperature)
        if memory_type:
            where += " AND memory_type=?"
            params.append(memory_type)

        # Total count
        count_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories {where}", params
        ).fetchone()
        total = count_row["n"] if count_row else 0

        # Paginated list
        offset = (page - 1) * per_page
        rows = conn.execute(
            f"SELECT memory_id, text, memory_type, layer, temperature, "
            f"scope, status, importance, created_at, updated_at, last_recalled_at, "
            f"expires_at, supersedes "
            f"FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, per_page, offset],
        ).fetchall()

        memories = [
            {
                "memory_id": r["memory_id"],
                "text": (r["text"] or "")[:300],
                "memory_type": r["memory_type"],
                "layer": r["layer"],
                "temperature": r["temperature"],
                "scope": r["scope"],
                "status": r["status"],
                "importance": r["importance"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "last_recalled_at": r["last_recalled_at"],
                "expires_at": r["expires_at"],
                "supersedes": r["supersedes"],
            }
            for r in rows
        ]

        return {
            "agent_key": agent_key,
            "project_id": project_id,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page) if total else 1,
            "memories": memories,
        }
    finally:
        conn.close()
