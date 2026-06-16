"""Procedural memory → Hermes skill auto-conversion.

Scans NTN MEM for ``memory_type=procedural``, ``status=verified`` memories
and generates Hermes SKILL.md files under ``~/.hermes/skills/``.

Clustering heuristic:
- Same ``role`` + ``project_id`` + ``scope`` are grouped into one skill.
- Text summaries are concatenated as numbered steps.
- Naming: ``<role>-<short-topic>`` or falls back to ``procedural-<project_id>-<scope>``.

State is persisted to ``/data/mem-procedural-skill-state.json`` to track
which memory_ids have already been baked into a skill (avoid duplicates).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .app import _connect, _row_to_memory

DEFAULT_STATE_PATH = "/data/mem-procedural-skill-state.json"
DEFAULT_SKILLS_ROOT = os.path.expanduser("~/.hermes/skills")
CATEGORY = "procedural"  # Hermes skill category folder


def _row_to_memory_safe(row: Any) -> dict[str, Any]:
    """Convert a SQLite row to a memory dict."""
    return _row_to_memory(row)


@dataclass
class SkillGenerationState:
    last_run_at: str = ""
    total_processed: int = 0
    skills_created: int = 0
    memories_generated: int = 0
    already_materialized: list[str] | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.already_materialized is None:
            self.already_materialized = []


def _read_state(path: str) -> SkillGenerationState:
    p = Path(path)
    if not p.exists():
        return SkillGenerationState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        state = SkillGenerationState(**{k: raw.get(k, v) for k, v in asdict(SkillGenerationState()).items()})
        if state.already_materialized is None:
            state.already_materialized = []
        return state
    except (OSError, ValueError, json.JSONDecodeError):
        return SkillGenerationState()


def _write_state(path: str, state: SkillGenerationState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _extract_topic(memories: list[dict[str, Any]]) -> str:
    """Derive a short topic name from a cluster of memories."""
    texts = [m.get("text") or m.get("summary") or "" for m in memories[:3]]
    combined = " ".join(texts)

    # Look for the most specific topic keyword
    for keyword in ("步骤", "procedure", "workflow", "runbook", "如何", "命令", "修复流程", "如何", "配置"):
        if keyword in combined:
            # Extract a short snippet after the keyword
            idx = combined.index(keyword)
            snippet = combined[idx : idx + 30].strip()
            return snippet[:40].rstrip("。，,.")

    # Fallback: first memory's first meaningful segment
    if texts:
        first = re.sub(r"[。！？，\n]", " ", texts[0]).strip()
        if len(first) > 40:
            first = first[:40]
        return first
    return "procedure"


def _slugify(name: str) -> str:
    """Convert a topic into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]", "_", name.lower().replace(" ", "_"))
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:48] or "procedure"


def _generate_skill_md(
    name: str,
    memories: list[dict[str, Any]],
) -> str:
    """Generate a SKILL.md from a list of procedural memories."""
    role = memories[0].get("owner_role") or memories[0].get("role") or "unknown"
    description = memories[0].get("summary") or _extract_topic(memories)

    steps = []
    notes = []
    for mem in memories:
        text = mem.get("text") or ""
        summary = mem.get("summary") or ""
        if summary and summary != text:
            steps.append(f"1. {summary.strip()}")
            notes.append(f"   - {text.strip()}" if text else "")
        else:
            steps.append(f"1. {text.strip()}")

    body = f"""# {name.replace("_", " ").title()}

**Source:** NTN MEM procedural memory (role: {role})

## Steps

{chr(10).join(steps)}

## Details

"""
    if notes:
        body += chr(10).join(notes) + chr(10) + "\n"
    else:
        body += "See individual memory entries for full details.\n\n"

    body += "## Metadata\n\n"
    body += f"- Generated from {len(memories)} memories at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    body += f"- Agent roles: {role}\n"
    body += "- Category: procedural\n"

    return body


def _generate_frontmatter(name: str, memories: list[dict[str, Any]]) -> str:
    """Generate YAML frontmatter for the SKILL.md."""
    role = memories[0].get("owner_role") or memories[0].get("role") or "unknown"
    description = memories[0].get("summary") or _extract_topic(memories)
    description = description.replace('"', "'")[:100]

    return f"""---
name: {_slugify(name)}
description: "{description}"
category: {CATEGORY}
source: ntn-mem-procedural
agent_roles: [{role}]
memory_count: {len(memories)}
generated_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
---

"""


def _cluster_memories(memories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group memories by (role, project_id, scope) into clusters."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mem in memories:
        key = f"{mem.get('owner_role') or mem.get('role') or 'unknown'}:{mem.get('project_id') or 'default'}:{mem.get('scope') or 'role_private'}"
        clusters[key].append(mem)
    return dict(clusters)


def generate_skills(
    *,
    limit: int = 500,
    min_importance: int = 30,
    min_memories_per_skill: int = 1,
    skills_root: str | None = None,
    state_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan procedural memories and generate Hermes skills.

    Returns a summary dict of what was created or would be created.
    """
    state_path = state_path or DEFAULT_STATE_PATH
    skills_root = skills_root or DEFAULT_SKILLS_ROOT
    state = _read_state(state_path)

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM memories WHERE memory_type='procedural' AND status='verified' "
            "AND COALESCE(deleted,0)=0 AND (importance IS NULL OR importance >= ?) ORDER BY created_at DESC LIMIT ?",
            (min_importance, limit),
        ).fetchall()
    finally:
        conn.close()

    procedural_memories = [_row_to_memory_safe(row) for row in rows]
    state.total_processed = len(procedural_memories)

    # Filter out already materialized
    new_memories = [m for m in procedural_memories if m.get("memory_id") not in (state.already_materialized or [])]
    if not new_memories:
        state.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_state(state_path, state)
        return {
            "accepted": True,
            "dry_run": dry_run,
            "scanned": len(procedural_memories),
            "new_memories": 0,
            "skills_created": 0,
            "memories_generated": 0,
            "message": "All procedural memories already materialized as skills.",
        }

    clusters = _cluster_memories(new_memories)
    skills_created = 0
    memories_generated = 0

    for cluster_key, cluster_memories in clusters.items():
        if len(cluster_memories) < min_memories_per_skill:
            continue

        topic = _extract_topic(cluster_memories)
        skill_name = _slugify(f"{cluster_key.replace(':', '-')}-{topic}")

        # Build SKILL.md content
        frontmatter = _generate_frontmatter(skill_name, cluster_memories)
        body = _generate_skill_md(skill_name, cluster_memories)
        skill_content = frontmatter + body

        if dry_run:
            skills_created += 1
            memories_generated += len(cluster_memories)
            continue

        # Write to disk
        skill_dir = Path(skills_root) / CATEGORY / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(skill_content, encoding="utf-8")

        # Track materialized memory_ids
        for mem in cluster_memories:
            mid = mem.get("memory_id")
            if mid and mid not in (state.already_materialized or []):
                state.already_materialized.append(mid)

        skills_created += 1
        memories_generated += len(cluster_memories)

    state.skills_created += skills_created
    state.memories_generated += memories_generated
    state.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_state(state_path, state)

    return {
        "accepted": True,
        "dry_run": dry_run,
        "scanned": len(procedural_memories),
        "new_memories": len(new_memories),
        "skills_created": skills_created,
        "memories_generated": memories_generated,
        "clusters": len(clusters),
        "already_materialized": len(state.already_materialized),
    }


def read_generation_state(state_path: str | None = None) -> dict[str, Any]:
    """Read current skill generation state without triggering a scan."""
    state = _read_state(state_path or DEFAULT_STATE_PATH)
    return asdict(state)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Hermes skills from procedural memories")
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--limit", type=int, default=500, help="Max procedural memories to scan")
    parser.add_argument("--min-importance", type=int, default=30, help="Min importance threshold")
    parser.add_argument("--skills-root", default=DEFAULT_SKILLS_ROOT, help="Hermes skills root directory")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="State file path")

    args = parser.parse_args()
    result = generate_skills(
        limit=args.limit,
        min_importance=args.min_importance,
        skills_root=args.skills_root,
        state_path=args.state_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
