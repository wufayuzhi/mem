"""
Unified structured-logging module for NTN MEM.
Writes to an independent SQLite table so it never blocks memory operations.

Every `except Exception` on the hot path MUST call `log_error()` here.
Provides query endpoints via `query_errors()` and `query_error_stats()`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone

_SYSLOG = logging.getLogger("ntn-mem.logger")
_LOG_DB_PATH: str = os.environ.get("NTN_MEM_LOG_DB", "/data/mem_log.db")
_LOG_CLEANUP_DAYS: int = 30

# ── Level constants ──
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

LEVEL_NAMES = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
NAME_LEVELS = {v: k for k, v in LEVEL_NAMES.items()}

_write_lock = threading.Lock()


def _log_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mem_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,           -- ISO-8601 with timezone
            level INTEGER NOT NULL DEFAULT 20,
            module TEXT NOT NULL DEFAULT '',
            operation TEXT NOT NULL DEFAULT '',   -- e.g. "search", "add_memory"
            summary TEXT DEFAULT '',              -- short description of what happened
            request_snippet TEXT DEFAULT '',       -- truncated JSON of request body
            status TEXT DEFAULT '',               -- "ok", "degraded", "error"
            error_type TEXT DEFAULT '',           -- exception class name
            error_detail TEXT DEFAULT '',         -- exception message (no traceback)
            traceback TEXT DEFAULT '',            -- short traceback
            duration_ms INTEGER DEFAULT 0,
            extra TEXT DEFAULT '{}',              -- JSON dict for arbitrary extra fields
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_logs_ts ON mem_logs(ts DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_logs_level ON mem_logs(level)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_logs_operation ON mem_logs(operation)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mem_logs_error_type ON mem_logs(error_type)
    """)


def _duration_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def log(
    level: int,
    module: str = "",
    operation: str = "",
    summary: str = "",
    request_snippet: str = "",
    status: str = "",
    error_type: str = "",
    error_detail: str = "",
    traceback_str: str = "",
    duration_ms: int = 0,
    extra: dict | None = None,
) -> int | None:
    """Write a log entry. Returns the row ID on success, None on failure."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        extra_json = json.dumps(extra or {}, ensure_ascii=False)
        with _write_lock:
            conn = _log_db()
            cur = conn.execute(
                """INSERT INTO mem_logs
                   (ts, level, module, operation, summary, request_snippet,
                    status, error_type, error_detail, traceback, duration_ms, extra)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now, level, module, operation, summary[:500], request_snippet[:1000],
                 status, error_type[:200], error_detail[:1000], traceback_str[:2000],
                 duration_ms, extra_json),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return row_id
    except Exception as exc:
        _SYSLOG.error("log_write_failed: %s", exc)
        return None


def log_error(
    module: str = "",
    operation: str = "",
    summary: str = "",
    request_snippet: str = "",
    exc: Exception | None = None,
    extra: dict | None = None,
    log_stdout: bool = True,
) -> int | None:
    """Convenience: log an ERROR entry with exception info.

    Args:
        module: Source module name (e.g. "app", "embed_worker")
        operation: Operation name (e.g. "search_memory", "add_memory")
        summary: Human-readable description
        request_snippet: Truncated request body
        exc: The caught exception (extracts type, message, traceback)
        extra: Extra JSON fields
        log_stdout: Also print to stderr via logging module (default True)
    """
    error_type = type(exc).__name__ if exc else ""
    error_detail = str(exc) if exc else ""
    tb = ""
    if exc:
        tb_lines = traceback.format_exception_only(type(exc), exc)
        tb = "".join(tb_lines).strip()[:2000]
    rid = log(
        level=ERROR,
        module=module,
        operation=operation,
        summary=summary,
        request_snippet=request_snippet,
        status="error",
        error_type=error_type,
        error_detail=error_detail,
        traceback_str=tb,
        extra=extra,
    )
    if log_stdout:
        _SYSLOG.error("[%s] %s: %s | %s", operation, error_type, error_detail, summary)
    return rid


def log_warn(
    module: str = "",
    operation: str = "",
    summary: str = "",
    request_snippet: str = "",
    exc: Exception | None = None,
    extra: dict | None = None,
) -> int | None:
    """Convenience: log a WARNING entry."""
    error_type = type(exc).__name__ if exc else ""
    error_detail = str(exc) if exc else ""
    rid = log(
        level=WARNING,
        module=module,
        operation=operation,
        summary=summary,
        request_snippet=request_snippet,
        status="degraded" if exc else "warning",
        error_type=error_type,
        error_detail=error_detail,
        extra=extra,
    )
    if exc:
        _SYSLOG.warning("[%s] %s: %s", operation, error_type, error_detail)
    else:
        _SYSLOG.warning("[%s] %s", operation, summary)
    return rid


def log_info(
    module: str = "",
    operation: str = "",
    summary: str = "",
    duration_ms: int = 0,
    extra: dict | None = None,
) -> int | None:
    """Convenience: log an INFO entry."""
    rid = log(
        level=INFO,
        module=module,
        operation=operation,
        summary=summary,
        status="ok",
        duration_ms=duration_ms,
        extra=extra,
    )
    return rid


def log_operation(
    module: str = "",
    operation: str = "",
    summary: str = "",
    request_snippet: str = "",
    status: str = "ok",
    duration_ms: int = 0,
    extra: dict | None = None,
) -> int | None:
    """Log a generic operation with status (ok/degraded/error)."""
    return log(
        level=INFO if status == "ok" else WARNING,
        module=module,
        operation=operation,
        summary=summary,
        request_snippet=request_snippet,
        status=status,
        duration_ms=duration_ms,
        extra=extra,
    )


# ── Query helpers ──

def query_errors(
    since_hours: int = 24,
    level_min: int = ERROR,
    limit: int = 50,
    operation: str | None = None,
) -> list[dict]:
    """Query recent errors/warnings flagged by the log system.

    Returns list of dicts sorted by ts DESC.
    """
    try:
        conn = _log_db()
        where = "level >= ? AND ts >= datetime('now', ?)"
        params: list = [level_min, f"-{since_hours} hours"]
        if operation:
            where += " AND operation = ?"
            params.append(operation)
        rows = conn.execute(
            f"SELECT id, ts, level, module, operation, summary, "
            f"error_type, error_detail, duration_ms, status, extra "
            f"FROM mem_logs WHERE {where} ORDER BY ts DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        _SYSLOG.error("query_errors failed: %s", exc)
        return []


def query_error_stats(
    since_hours: int = 24,
) -> dict:
    """Aggregated error stats: counts by operation and error_type."""
    try:
        conn = _log_db()
        # Total count
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM mem_logs WHERE level >= ? AND ts >= datetime('now', ?)",
            [ERROR, f"-{since_hours} hours"],
        ).fetchone()
        total_errors = total["cnt"] if total else 0

        # By operation
        by_op = conn.execute(
            "SELECT operation, COUNT(*) as cnt FROM mem_logs "
            "WHERE level >= ? AND ts >= datetime('now', ?) "
            "GROUP BY operation ORDER BY cnt DESC",
            [ERROR, f"-{since_hours} hours"],
        ).fetchall()

        # By error_type
        by_type = conn.execute(
            "SELECT error_type, COUNT(*) as cnt FROM mem_logs "
            "WHERE level >= ? AND ts >= datetime('now', ?) AND error_type != '' "
            "GROUP BY error_type ORDER BY cnt DESC",
            [ERROR, f"-{since_hours} hours"],
        ).fetchall()

        # Recent 10
        recent = conn.execute(
            "SELECT id, ts, operation, summary, error_type, error_detail, module "
            "FROM mem_logs WHERE level >= ? AND ts >= datetime('now', ?) "
            "ORDER BY ts DESC LIMIT 10",
            [ERROR, f"-{since_hours} hours"],
        ).fetchall()

        # Warnings count
        warnings_total = conn.execute(
            "SELECT COUNT(*) as cnt FROM mem_logs WHERE level = ? AND ts >= datetime('now', ?)",
            [WARNING, f"-{since_hours} hours"],
        ).fetchone()

        conn.close()
        return {
            "total_errors": total_errors,
            "total_warnings": warnings_total["cnt"] if warnings_total else 0,
            "by_operation": {r["operation"]: r["cnt"] for r in by_op},
            "by_error_type": {r["error_type"]: r["cnt"] for r in by_type if r["error_type"]},
            "recent_errors": [dict(r) for r in recent],
            "since_hours": since_hours,
        }
    except Exception as exc:
        _SYSLOG.error("query_error_stats failed: %s", exc)
        return {"error": str(exc)}


def query_logs(
    level_min: int = INFO,
    since_hours: int = 24,
    limit: int = 50,
    operation: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Generic log query."""
    try:
        conn = _log_db()
        where = "level >= ? AND ts >= datetime('now', ?)"
        params: list = [level_min, f"-{since_hours} hours"]
        if operation:
            where += " AND operation = ?"
            params.append(operation)
        if status:
            where += " AND status = ?"
            params.append(status)
        rows = conn.execute(
            f"SELECT id, ts, level, module, operation, summary, "
            f"status, error_type, error_detail, duration_ms, extra "
            f"FROM mem_logs WHERE {where} ORDER BY ts DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        _SYSLOG.error("query_logs failed: %s", exc)
        return []


def cleanup_old_logs(days: int | None = None) -> int:
    """Delete logs older than N days. Returns count of deleted rows."""
    days = days or _LOG_CLEANUP_DAYS
    try:
        conn = _log_db()
        cur = conn.execute(
            "DELETE FROM mem_logs WHERE ts < datetime('now', ?)",
            [f"-{days} days"],
        )
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        _SYSLOG.info("log cleanup: deleted %d rows older than %d days", deleted, days)
        return deleted
    except Exception as exc:
        _SYSLOG.error("log cleanup failed: %s", exc)
        return 0
