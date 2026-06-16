"""时间戳工具（内联版，消除对 ntn_common 的依赖）。

死线:
    - 一律 UTC, ISO 8601, 毫秒精度
    - 格式: `YYYY-MM-DDTHH:MM:SS.fffZ` (末尾 Z 必带)
    - 禁存 Unix 时间戳 INT, 禁存本地时区字符串
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 毫秒精度字符串。

    >>> now_iso()  # 形如 '2026-05-20T09:53:00.123Z'
    """
    dt = datetime.now(timezone.utc)
    ms = dt.microsecond // 1000
    return dt.strftime(f"%Y-%m-%dT%H:%M:%S.{ms:03d}Z")


def now_unix() -> int:
    """返回当前 UTC Unix 秒（仅给 escalated_at / auto_denied_at 使用）。"""
    return int(datetime.now(timezone.utc).timestamp())


def parse_iso(s: str) -> datetime:
    """解析 ISO 8601 UTC 字符串为 datetime(timezone-aware)。

    传入非法格式抛 ValueError。
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def iso_diff_seconds(a: str, b: str) -> float:
    """返回 a - b 的秒差（用于审计日志的耗时统计）。"""
    return (parse_iso(a) - parse_iso(b)).total_seconds()
