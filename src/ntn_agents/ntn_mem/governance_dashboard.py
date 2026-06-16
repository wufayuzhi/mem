"""Governance dashboard / audit for NTN MEM.

Provides aggregated views of the memory store sliced by:
- agent_key (actor_role/owner_role/role)
- layer   (hot / warm / cold)
- temperature (hot / warm / cold)
- memory_type (procedural / episodic / semantic / shared_kb)
- status, importance buckets, vitality, expiry, and more.

All queries operate on the live ``memories`` table with optional time filtering.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .app import _connect


def _bucket_importance(val: int | None) -> str:
    if val is None:
        return "unset"
    if val >= 80:
        return "high (80-100)"
    if val >= 50:
        return "medium (50-79)"
    if val >= 20:
        return "low (20-49)"
    return "critical-low (<20)"


def _bucket_vitality(val: int | None) -> str:
    if val is None:
        return "unset"
    if val >= 10:
        return "high (10+)"
    if val >= 5:
        return "medium (5-9)"
    if val >= 1:
        return "low (1-4)"
    return "zero"


def dashboard_summary(
    *,
    since_days: int | None = None,
) -> dict[str, Any]:
    """Return a complete governance snapshot over all memories.

    Args:
        since_days: if set, filter to memories created in the last N days.
    """
    conn = _connect()
    try:
        where_clause = ""
        params: list[Any] = []
        if since_days is not None and since_days > 0:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - since_days * 86400),
            )
            where_clause = " WHERE created_at >= ? AND COALESCE(deleted,0)=0"
            params.append(cutoff)
        else:
            where_clause = " WHERE COALESCE(deleted,0)=0"
        where_and = where_clause + " AND"

        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories{where_clause}", params
        ).fetchone()["n"]

        # ---- agent_key breakdown (aggregate on role/actor_role/owner_role) ----
        agent_keys: dict[str, int] = {}
        agent_keys_total = conn.execute(
            f"SELECT COALESCE(role,'<null>') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        agent_keys_by_role = {row["k"]: row["cnt"] for row in agent_keys_total}

        # actor_role breakdown
        actor_keys_total = conn.execute(
            f"SELECT COALESCE(actor_role,'<null>') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        agent_keys_by_actor = {row["k"]: row["cnt"] for row in actor_keys_total}

        # ---- layer breakdown ----
        layers_raw = conn.execute(
            f"SELECT COALESCE(layer,'unset') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        by_layer = {row["k"]: row["cnt"] for row in layers_raw}

        # ---- temperature breakdown ----
        temps_raw = conn.execute(
            f"SELECT COALESCE(temperature,'unset') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        by_temperature = {row["k"]: row["cnt"] for row in temps_raw}

        # ---- memory_type breakdown ----
        types_raw = conn.execute(
            f"SELECT COALESCE(memory_type,'unset') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        by_memory_type = {row["k"]: row["cnt"] for row in types_raw}

        # ---- importance buckets ----
        importance_buckets: dict[str, int] = {}
        for row in conn.execute(
            f"SELECT importance FROM memories{where_clause}", params
        ).fetchall():
            bucket = _bucket_importance(row["importance"])
            importance_buckets[bucket] = importance_buckets.get(bucket, 0) + 1

        # ---- vitality buckets ----
        vitality_buckets: dict[str, int] = {}
        for row in conn.execute(
            f"SELECT vitality FROM memories{where_clause}", params
        ).fetchall():
            bucket = _bucket_vitality(row["vitality"])
            vitality_buckets[bucket] = vitality_buckets.get(bucket, 0) + 1

        # ---- status breakdown ----
        statuses = conn.execute(
            f"SELECT COALESCE(status,'unset') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        by_status = {row["k"]: row["cnt"] for row in statuses}

        # ---- expired / expiring ----
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        expired = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE expires_at IS NOT NULL AND expires_at < ? AND COALESCE(deleted,0)=0",
            (now_iso,),
        ).fetchone()["n"]

        expiring_7d = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE expires_at IS NOT NULL AND expires_at >= ? AND expires_at < ? AND COALESCE(deleted,0)=0",
            (now_iso, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 7 * 86400))),
        ).fetchone()["n"]

        # ---- scope breakdown ----
        scopes_raw = conn.execute(
            f"SELECT COALESCE(scope,'unset') AS k, COUNT(*) AS cnt FROM memories{where_clause} GROUP BY k ORDER BY cnt DESC",
            params,
        ).fetchall()
        by_scope = {row["k"]: row["cnt"] for row in scopes_raw}

        # ---- protected count ----
        protected = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories{where_clause} AND protected=1", params
        ).fetchone()["n"]

        # ---- deleted (if not filtered) ----
        deleted_total = 0
        if since_days is None:
            deleted_total = conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE deleted=1"
            ).fetchone()["n"]

        # ---- supersedes chains ----
        supersedes_count = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories{where_clause} AND supersedes IS NOT NULL AND supersedes != ''",
            params,
        ).fetchone()["n"]

        # ---- recall activity ----
        never_recalled = conn.execute(
            f"SELECT COUNT(*) AS n FROM memories{where_clause} AND (last_recalled_at IS NULL OR last_recalled_at = '')",
            params,
        ).fetchone()["n"]

        return {
            "total": total,
            "filter_since_days": since_days,
            "by_agent_role": agent_keys_by_role,
            "by_actor_role": agent_keys_by_actor,
            "by_layer": by_layer,
            "by_temperature": by_temperature,
            "by_memory_type": by_memory_type,
            "by_scope": by_scope,
            "by_status": by_status,
            "by_importance": importance_buckets,
            "by_vitality": vitality_buckets,
            "protected": protected,
            "superseded": supersedes_count,
            "expired": expired,
            "expiring_within_7d": expiring_7d,
            "never_recalled": never_recalled,
            "deleted_total": deleted_total,
        }
    finally:
        conn.close()


def dashboard_agent_audit(
    agent_key: str,
    *,
    limit: int = 100,
    include_deleted: bool = False,
) -> dict[str, Any]:
    """Return detailed audit data for a single agent/role.

    Covers memories where role=agent_key OR actor_role=agent_key OR owner_role=agent_key.
    """
    conn = _connect()
    try:
        base = "SELECT * FROM memories WHERE" if include_deleted else "SELECT * FROM memories WHERE COALESCE(deleted,0)=0 AND"
        sql = base + " (role=? OR actor_role=? OR owner_role=?) ORDER BY created_at DESC LIMIT ?"
        rows = conn.execute(sql, (agent_key, agent_key, agent_key, limit)).fetchall()

        if not rows:
            return {"agent_key": agent_key, "found": False}

        memories = [_row_to_memory_slim(row) for row in rows]

        layer_count: dict[str, int] = {}
        type_count: dict[str, int] = {}
        temp_count: dict[str, int] = {}
        scope_count: dict[str, int] = {}
        total_importance = 0

        for m in memories:
            l = m.get("layer") or "unset"
            layer_count[l] = layer_count.get(l, 0) + 1
            t = m.get("memory_type") or "unset"
            type_count[t] = type_count.get(t, 0) + 1
            temp = m.get("temperature") or "unset"
            temp_count[temp] = temp_count.get(temp, 0) + 1
            s = m.get("scope") or "unset"
            scope_count[s] = scope_count.get(s, 0) + 1
            total_importance += m.get("importance") or 0

        return {
            "agent_key": agent_key,
            "found": True,
            "total_memories": len(memories),
            "avg_importance": round(total_importance / len(memories), 1) if memories else 0,
            "by_layer": layer_count,
            "by_memory_type": type_count,
            "by_temperature": temp_count,
            "by_scope": scope_count,
            "most_recent_5": memories[:5],
        }
    finally:
        conn.close()


def dashboard_layer_temperature_matrix(
    *,
    since_days: int | None = None,
) -> dict[str, Any]:
    """Return a cross-tabulation of layer × temperature."""
    conn = _connect()
    try:
        where_clause = ""
        params: list[Any] = []
        if since_days is not None and since_days > 0:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - since_days * 86400),
            )
            where_clause = " WHERE created_at >= ? AND COALESCE(deleted,0)=0"
            params.append(cutoff)
        else:
            where_clause = " WHERE COALESCE(deleted,0)=0"

        rows = conn.execute(
            f"SELECT COALESCE(layer,'unset') AS l, COALESCE(temperature,'unset') AS t, COUNT(*) AS cnt "
            f"FROM memories{where_clause} GROUP BY l, t ORDER BY l, t",
            params,
        ).fetchall()

        matrix: dict[str, dict[str, int]] = {}
        for row in rows:
            l = row["l"]
            t = row["t"]
            if l not in matrix:
                matrix[l] = {}
            matrix[l][t] = row["cnt"]

        return {"matrix": matrix}
    finally:
        conn.close()


def dashboard_stale_memories(
    *,
    not_recalled_days: int = 30,
    limit: int = 50,
) -> dict[str, Any]:
    """Find memories that haven't been recalled recently (potential candidates for archiving/downgrading)."""
    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() - not_recalled_days * 86400),
    )
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM memories WHERE COALESCE(deleted,0)=0 AND "
            "(last_recalled_at IS NULL OR last_recalled_at = '' OR last_recalled_at < ?) "
            "AND (expires_at IS NULL OR expires_at = '') "
            "AND protected=0 "
            "ORDER BY importance ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()

        memories = [_row_to_memory_slim(row) for row in rows]
        return {
            "threshold_days": not_recalled_days,
            "cutoff": cutoff,
            "count": len(memories),
            "memories": memories,
        }
    finally:
        conn.close()


def _row_to_memory_slim(row: Any) -> dict[str, Any]:
    """Convert an sqlite3.Row to a slim dict with metadata parsed."""
    mem = {
        "memory_id": row["memory_id"],
        "text": (row["text"] or "")[:200],  # truncate for dashboard
        "role": row["role"],
        "actor_role": row["actor_role"],
        "owner_role": row["owner_role"],
        "layer": row["layer"],
        "scope": row["scope"],
        "memory_type": row["memory_type"],
        "temperature": row["temperature"],
        "importance": row["importance"],
        "vitality": row["vitality"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_recalled_at": row["last_recalled_at"],
        "expires_at": row["expires_at"],
        "supersedes": row["supersedes"],
        "protected": bool(row["protected"]),
    }
    # parse metadata_json for additional tags if present
    raw_md = row.get("metadata_json") or "{}"
    if isinstance(raw_md, str):
        try:
            md = json.loads(raw_md)
        except (json.JSONDecodeError, ValueError):
            md = {}
    else:
        md = raw_md or {}
    mem["metadata_tags"] = list(md.keys())[:5]  # key names only
    return mem


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTN MEM Governance Dashboard")
    sub = parser.add_subparsers(dest="command", required=True)

    summary_p = sub.add_parser("summary", help="Full governance summary")
    summary_p.add_argument("--since-days", type=int, help="Filter to recent N days")

    agent_p = sub.add_parser("agent", help="Audit a single agent")
    agent_p.add_argument("agent_key", help="Agent key to audit")
    agent_p.add_argument("--limit", type=int, default=100)
    agent_p.add_argument("--include-deleted", action="store_true")

    matrix_p = sub.add_parser("matrix", help="Layer × temperature matrix")
    matrix_p.add_argument("--since-days", type=int)

    stale_p = sub.add_parser("stale", help="Stale memories (not recalled)")
    stale_p.add_argument("--days", type=int, default=30)
    stale_p.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()

    if args.command == "summary":
        result = dashboard_summary(since_days=args.since_days)
    elif args.command == "agent":
        result = dashboard_agent_audit(args.agent_key, limit=args.limit, include_deleted=args.include_deleted)
    elif args.command == "matrix":
        result = dashboard_layer_temperature_matrix(since_days=args.since_days)
    elif args.command == "stale":
        result = dashboard_stale_memories(not_recalled_days=args.days, limit=args.limit)

    print(json.dumps(result, ensure_ascii=False, indent=2))
