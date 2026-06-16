"""
画像蒸馏 — Hermes 主模型侧画像更新写入接口
=========================================
每轮对话结束时，Hermes 主模型判断是否学到用户新特征，
有则 POST /v1/memory/profile/update 写入一条带 [画像更新] 前缀的记忆。

不主动调 LLM，不增加额外费用。纯粹是"主模型写笔记"的写入 API。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── 画像更新查 / 写 ──

_PROFILE_PREFIX = "[画像更新]"


def build_profile_memory(agent_key: str, text: str) -> dict[str, Any]:
    """构建一条画像更新记忆，返回 add_memory 可用的 dict。"""
    return {
        "agent_key": agent_key,
        "text": f"{_PROFILE_PREFIX} {text.strip()}",
        "temperature": "hot",           # 画像更新永远是 hot
        "importance": 80,               # 高重要性
        "metadata": json.dumps({
            "type": "profile_update",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }),
    }


def get_recent_updates(agent_key: str, conn, limit: int = 5) -> list[dict]:
    """获取最近 N 条画像更新（按时间倒序）。"""
    import sqlite3
    cur = conn.execute(
        """SELECT text, created_at FROM memories
           WHERE text LIKE ? AND project_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (f"{_PROFILE_PREFIX}%", f"private-{agent_key}", limit),
    )
    rows = cur.fetchall()
    return [
        {"text": r[0], "created_at": r[1]}
        for r in rows
    ]


def route_profile_update(environ, start_response) -> list[bytes]:
    """POST /v1/memory/profile/update
    请求体: {"agent_key": "...", "insight": "..."}
    """
    try:
        length = int(environ.get("CONTENT_LENGTH", "0"))
        body = json.loads(environ["wsgi.input"].read(length)) if length else {}
    except Exception:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "invalid JSON body"}).encode()]

    agent_key = body.get("agent_key", "")
    insight = body.get("insight", "")
    if not agent_key or not insight:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "agent_key and insight are required"}).encode()]

    # 直接调 add_memory，不走 HTTP 回环
    from .app import add_memory
    payload = {
        "agent_key": agent_key,
        "role": "assistant",
        "text": f"{_PROFILE_PREFIX} {insight.strip()}",
        "project_id": f"private-{agent_key}",
        "temperature": "hot",
        "importance": 80,
        "metadata": json.dumps({
            "type": "profile_update",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }),
    }
    try:
        result = add_memory(payload)
        logger.info(f"画像更新已写入: agent={agent_key} insight={insight[:60]}")
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps({"ok": True, "insight": insight[:100], "memory": result}).encode()]
    except Exception as e:
        logger.warning(f"画像更新写入失败: {e}")
        start_response("502 Bad Gateway", [("Content-Type", "application/json")])
        return [json.dumps({"error": "write failed", "detail": str(e)}).encode()]


def route_get_updates(environ, start_response) -> list[bytes]:
    """GET /v1/memory/profile/updates?agent_key=xxx&limit=5
    获取最近的画像更新记录。
    """
    from urllib.parse import parse_qs
    params = parse_qs(environ.get("QUERY_STRING", ""))
    agent_key = params.get("agent_key", [""])[0]
    limit = int(params.get("limit", ["5"])[0])
    if not agent_key:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "agent_key is required"}).encode()]

    from .app import _connect
    conn = _connect()
    try:
        updates = get_recent_updates(agent_key, conn, limit)
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps({"updates": updates}).encode()]
    finally:
        conn.close()
