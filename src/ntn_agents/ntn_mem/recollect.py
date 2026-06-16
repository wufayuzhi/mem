"""
Active recoll (recall + collate) for NTN MEM — 渐进式记忆提取.

TWO-LEVEL PROTOCOL:

Level 1 — /v1/memory/recollect (概要)
  快速搜索三个库 → MEM 侧 LLM 生成 300-500 字概要 →
  告诉主模型: "有×条私人记忆和×条经验库记录，简要内容如下..."
  主模型判断是否需要更多细节。

Level 2 — /v1/memory/recollect/detail (详细)
  主模型决定需要更多上下文后，主动调此接口。
  根据 memory_id 或 query + archive 取详细原文。

Architecture:
  Agent --POST /v1/memory/recollect--> MEM
    <-- {"has_context": bool, "gist": "300-500字概要", "archives": {...}}

  Agent --POST /v1/memory/recollect/detail--> MEM
    <-- {"has_context": bool, "sources": {"private_memory": [...], ...}}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .agent_standard import standard_agent_profile
from .llm import LLMProvider

# Search config
_RECOLLECT_LIMIT = int(os.environ.get("NTN_MEM_RECOLLECT_LIMIT", "10"))
_RECOLLECT_DETAIL_LIMIT = int(os.environ.get("NTN_MEM_RECOLLECT_DETAIL_LIMIT", "5"))

# LLM gist config
_GIST_MAX_CHARS = int(os.environ.get("NTN_MEM_GIST_MAX_CHARS", "500"))
_MIN_SCORE = float(os.environ.get("NTN_MEM_RECOLLECT_MIN_SCORE", "0.4"))
_LLM_TIMEOUT = int(os.environ.get("NTN_MEM_LLM_TIMEOUT", "10"))  # 2s+ margin

# ── 场景感知 ↴ 对查询词做场景分类 ──
_SCENE_PATTERNS: dict[str, list[str]] = {
    "技术开发": ["写一个", "代码", "实现", "开发", "搭建", "创建项目", "编程", "debug", "编译", "部署", "git", "PR", "merge", "commit", "测试", "重构", "迁移", "升级"],
    "架构讨论": ["架构", "设计", "方案对比", "选型", "权衡", "vs", "对比", "架构图", "组件", "模块划分", "扩展性", "高可用", "容错"],
    "运维排障": ["故障", "报错", "502", "超时", "挂了", "启动不了", "崩溃", "OOM", "卡住", "日志", "告警", "监控", "恢复", "不能用了"],
    "决策管理": ["要不要", "选哪个", "做决策", "决定", "投入", "省钱", "成本", "预算", "值不值得", "优先", "方向", "计划"],
    "记忆/知识": ["记得", "之前", "查一下", "搜索", "找到", "有没有", "MEM", "记忆", "知识库", "历史记录"],
    "日常沟通": ["你好", "谢谢", "在吗", "完成", "好", "OK", "好的", "明白了", "收到"],
}


def _detect_scene(query: str) -> str:
    """Simple keyword-based scene classification for the current query.
    Returns a short Chinese scene label used in recollect prompt.
    """
    if not query:
        return "其他"
    q = query.lower()
    scores: dict[str, int] = {}
    for scene, keywords in _SCENE_PATTERNS.items():
        count = sum(1 for kw in keywords if kw.lower() in q)
        if count:
            scores[scene] = count
    if not scores:
        return "其他"
    best = max(scores, key=scores.get)
    return best


def _add_scene_boost(scene: str) -> list[str]:
    """Return boost_terms based on scene type.
    Called from search_memory to adjust ranking by scene affinity.
    """
    boost_map: dict[str, list[str]] = {
        "技术开发": ["代码", "实现", "开发", "功能", "bug修复", "技术"],
        "架构讨论": ["架构", "设计", "方案", "组件", "系统"],
        "运维排障": ["故障", "修复", "排查", "错误", "告警", "恢复"],
        "决策管理": ["决策", "投入", "省钱", "方向", "成本", "推荐"],
        "记忆/知识": ["记忆", "知识库", "历史", "记录", "经验"],
        "日常沟通": [],
    }
    return boost_map.get(scene, [])


# ── 决策方向检测 ↴ 用于搜索层决策偏置 ──
_DECISION_DIRECTIONS: dict[str, tuple[tuple[str, ...], list[str]]] = {
    "投入": (
        ("方案", "用这个", "先改", "推进", "直接", "买", "花钱", "购", "升级", "投资", "扩展"),
        ["方案", "推进", "实施", "花费", "投入", "采购", "升级", "买"],
    ),
    "省钱": (
        ("免费", "省", "就用现有的", "不花钱", "省钱", "便宜", "自建", "本地", "不用额外", "省时间"),
        ["免费", "省钱", "节约", "自建", "本地", "现有", "节省", "替代"],
    ),
    "放弃": (
        ("不需要", "不测试", "算了", "跳过", "取消", "不做了"),
        ["不需要", "跳过", "取消", "放弃", "停止"],
    ),
}


def _detect_decision_direction(query: str) -> dict | None:
    """基于关键词检测 query 中的决策方向信号，返回偏置词或 None。"""
    query_lower = query.lower()
    for direction, (keywords, boost_words) in _DECISION_DIRECTIONS.items():
        for kw in keywords:
            if kw in query_lower:
                return {"direction": direction, "boost": boost_words}
    return None


def _search_memory(data: dict) -> dict:
    from .app import search_memory
    return search_memory(data)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _filter_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule-based filter: remove superseded, decayed, cold-and-abandoned."""
    filtered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for mem in memories:
        mid = mem.get("memory_id") or ""
        if not mid or mid in seen_ids:
            continue
        seen_ids.add(mid)

        # Rule A: superseded_by — replaced by newer fact
        if mem.get("superseded_by"):
            continue

        # Rule B: decayed
        if mem.get("temperature") == "decayed":
            continue

        # Rule C: cold AND never recalled OR not recalled in 30 days
        temp = mem.get("temperature") or "unknown"
        if temp == "cold":
            last_recalled = mem.get("last_recalled_at")
            if not last_recalled:
                continue
            try:
                lr = last_recalled.replace("Z", "+00:00") if "Z" in last_recalled else last_recalled
                recalled_dt = datetime.fromisoformat(lr)
                delta = (_now_utc() - recalled_dt).days
                if delta > 30:
                    continue
            except (ValueError, TypeError):
                pass

        # Build safe output dict
        safe_keys = [
            "memory_id", "text", "memory_type", "temperature", "importance",
            "layer", "created_at", "updated_at", "project_id", "source",
            "score", "tags", "source_path",
        ]
        entry: dict[str, Any] = {}
        for k in safe_keys:
            v = mem.get(k)
            if v is not None:
                entry[k] = v

        filtered.append(entry)

    return filtered


def _deduplicate(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for m in memories:
        text = (m.get("text") or "").strip()
        key = text[:200]
        if key and key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _search_all_archives(
    query: str,
    agent_key: str | None,
    limit: int,
    min_score: float = _MIN_SCORE,
) -> dict[str, Any]:
    """Search all three archives and return organised, filtered results.

    Returns:
        dict with: sources_by_label, all_tagged, total, queried_projects
    """
    profile = standard_agent_profile({"agent_key": agent_key or "unknown"})
    private_project = profile.private_memory_project
    shared_projects = profile.shared_knowledge_projects  # ["knowledge_reserve", "experience_reserve"]

    all_projects = [private_project] + list(shared_projects)
    source_labels = {private_project: "private_memory"}
    for sp in shared_projects:
        source_labels[sp] = sp

    archive_results: dict[str, list[dict[str, Any]]] = {}
    queried_projects: list[str] = []

    for project_id in all_projects:
        if not project_id or project_id in queried_projects:
            continue
        queried_projects.append(project_id)

        search_result = _search_memory({
            "query": query,
            "caller_role": agent_key,
            "target_role": agent_key if project_id == private_project else None,
            "project_id": project_id,
            "limit": limit * 2,  # search more, filter later
            "include_decayed": False,
        })

        raw = search_result.get("results") or []
        filtered = _filter_memories(raw)
        archive_results[project_id] = filtered

    # Organise by source label
    sources: dict[str, list[dict[str, Any]]] = {}
    for project_id, items in archive_results.items():
        label = source_labels.get(project_id, project_id)
        if label not in sources:
            sources[label] = []
        sources[label].extend(items)

    # Tag and dedup
    all_tagged: list[dict[str, Any]] = []
    for label, items in sources.items():
        for item in items:
            tagged_item = dict(item)
            tagged_item["_mem_archive"] = label
            all_tagged.append(tagged_item)

    total = len(all_tagged)
    deduped = _deduplicate(all_tagged)

    # Filter by min score
    deduped = [m for m in deduped if (m.get("score") or 0) >= min_score]

    # Sort by score descending
    deduped.sort(key=lambda x: -(x.get("score") or 0))

    # Regroup
    regrouped: dict[str, list[dict[str, Any]]] = {}
    for item in deduped:
        label = item.pop("_mem_archive", "unknown")
        if label not in regrouped:
            regrouped[label] = []
        regrouped[label].append(item)

    return {
        "sources_by_label": regrouped,
        "all_tagged": deduped[:20],  # max 20 for gist generation
        "total": total,
        "total_deduped": len(deduped),
        "queried_projects": queried_projects,
    }


# ── Level 1: Gist Generation ──────────────────────────────────────────────

def _generate_gist(
    query: str,
    regrouped: dict[str, list[dict[str, Any]]],
    total_deduped: int,
) -> str:
    """Have the MEM-side LLM (8B) produce a 300-500 char gist.

    The LLM categorises what was found: private memories, knowledge base
    entries, experience records. It writes ONE paragraph per archive type
    that summarises the key points WITHOUT deep analysis.
    """
    if not regrouped:
        return ""

    # Build a compact input for the LLM
    gist_data: dict[str, list[dict[str, str]]] = {}
    for label, items in regrouped.items():
        entries = []
        for m in items[:8]:  # at most 8 entries per archive for gist
            text = (m.get("text") or "").strip()[:300]  # 300 chars each
            temp = m.get("temperature") or "?"
            imp = m.get("importance") or "?"
            when = (m.get("created_at") or "")[:10]
            sp = m.get("source_path") or "-"
            entries.append({
                "text": text,
                "temp": str(temp),
                "imp": str(imp),
                "date": when,
                "source": sp,
            })
        gist_data[label] = entries

    archives_desc = []
    for label, entries in gist_data.items():
        label_cn = {"private_memory": "私人记忆", "knowledge_reserve": "知识储备", "experience_reserve": "经验储备"}.get(label, label)
        archives_desc.append(f"【{label_cn}】{len(entries)}条")

    prompt = f"""你是一个记忆系统助手，任务是为搜索到的历史记录生成结构化4段式摘要。

用户的当前查询：{query}

当前对话语境：{_detect_scene(query)}

搜索到 {total_deduped} 条相关记录，归档分布如下：
{' / '.join(archives_desc)}

每条记录的格式：text=内容, temp=新鲜度(hot/warm/cold), imp=重要性(0-100), date=日期。

{json.dumps(gist_data, ensure_ascii=False, indent=2)}

请按以下4段格式输出（中文）：

第1段：【画像】— 从搜索结果中提炼出与用户画像相关的信息（关注方向、决策倾向、沟通风格）。如果无画像信息就写"无新画像数据"。
第2段：【事实】— 置信度高的具体事实，标注置信度。每条约30-60字。
第3段：【经验/知识】— 知识库和经验储备中的相关文档或经验总结。如果无就跳过。
第4段：【待办】— 搜索结果中如果有"帮我、回头、找时间、下次"等承诺信号的记录，提取出来。如果无就跳过。

要求：
1. 每段只用一句话，不超过60字
2. 高置信度事实标记【高置信】，普通标记【中可信】
3. 开头不用概述句，直接从【画像】开始
4. 整体控制在200-300字
5. 如果某段无内容就写"无"并跳过

示例输出格式：
【画像】近期关注：MEM系统架构优化 | 决策倾向轻量化、容错优先
【事实】【高置信】嵌入通路已修复，3457条全量重嵌入bge-m3 1024维 @2026-06-13
【事实】【中可信】用户偏好中文、源证据、root-cause-then-repair
【知识】ccb多Agent编排架构：5层7类型4通信路径
【经验】近期经验：一次成功重嵌入（耗时47分钟）
【待办】"回头验证一下recollect的效果" @2天前"""

    try:
        llm = LLMProvider(timeout=_LLM_TIMEOUT)
        gist = llm.chat_str(
            [{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        # Clean up
        gist = gist.strip()
        if len(gist) > _GIST_MAX_CHARS:
            gist = gist[:_GIST_MAX_CHARS] + "…"
        return gist
    except RuntimeError as e:
        # LLM failed — return a basic text summary
        parts = []
        for label, items in regrouped.items():
            label_cn = {"private_memory": "私人记忆", "knowledge_reserve": "知识储备", "experience_reserve": "经验储备"}.get(label, label)
            item_lines = []
            for m in items[:3]:
                text = (m.get("text") or "")[:80]
                when = (m.get("created_at") or "")[:10]
                temp = m.get("temperature") or "?"
                item_lines.append(f"{text}… ({temp}|{when})")
            parts.append(f"{label_cn}({len(items)}条): {'; '.join(item_lines)}")
        return " | ".join(parts)


def recount_gist(
    query: str,
    agent_key: str | None = None,
    detail: bool = False,
    memory_ids: list[str] | None = None,
    archive_filter: str | None = None,
    limit: int = _RECOLLECT_DETAIL_LIMIT,
) -> dict[str, Any]:
    """Active recall with progressive detail — the unified backend.

    Args:
        query: User's message / context question.
        agent_key: Agent identifier.
        detail: If True, returns full memory details instead of gist.
        memory_ids: If set, fetch only these specific memories by ID.
        archive_filter: If set (e.g. "private_memory"), only return that archive.
        limit: Max memories per archive for detail mode.

    Returns:
        Level 1 (gist):
            {has_context, gist, archives: {label: count}, queried_projects}
        Level 2 (detail):
            {has_context, sources: {label: [...]}, summary: {...}}
    """
    if not query or not query.strip():
        return {
            "has_context": False,
            "gist": "",
            "archives": {},
            "error": "empty_query",
        }

    # ── Step 1: Search all archives ──
    search_limit = limit * 2 if detail else _RECOLLECT_LIMIT
    archive_data = _search_all_archives(query, agent_key, search_limit)
    regrouped = archive_data["sources_by_label"]
    total_deduped = archive_data["total_deduped"]
    has_context = total_deduped > 0

    # ── Step 2: If detail mode, return full sources ──
    if detail:
        _per_archive_limit = {
            "private_memory": 5,
            "knowledge_reserve": 3,
            "experience_reserve": 3,
        }
        sources: dict[str, list[dict[str, Any]]] = {}
        archive_counts: dict[str, int] = {}
        for label, items in regrouped.items():
            cap = _per_archive_limit.get(label, 3)
            kept = items[:cap]
            # Truncate each text
            for entry in kept:
                text = (entry.get("text") or "").strip()
                if len(text) > 800:
                    entry["text"] = text[:800] + "…"
                    entry["_truncated"] = True
            sources[label] = kept
            archive_counts[label] = len(kept)

        return {
            "has_context": has_context,
            "summary": {
                "total": archive_data["total"],
                "total_deduped": total_deduped,
                "queried_projects": archive_data["queried_projects"],
                "archives": archive_counts,
            },
            "sources": sources,
        }

    # ── Step 3: Gist mode — generate MEM-side LLM summary ──
    gist = _generate_gist(query, regrouped, total_deduped)

    archive_counts: dict[str, int] = {}
    for label, items in regrouped.items():
        archive_counts[label] = len(items)

    return {
        "has_context": has_context,
        "gist": gist,
        "archives": archive_counts,
        "queried_projects": archive_data["queried_projects"],
    }


# Deprecated: kept for backward compat, delegates to recount_gist
def recollect(
    query: str,
    agent_key: str | None = None,
    limit: int = _RECOLLECT_LIMIT,
) -> dict[str, Any]:
    """[DEPRECATED] Use recount_gist() for gist mode or recount_gist(..., detail=True) for detail.

    Returns gist (summary overview) of what was found across archives.
    """
    return recount_gist(query=query, agent_key=agent_key, limit=limit)
