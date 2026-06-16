"""Entity-relationship graph index for NTN MEM.

Builds and queries a directed graph of entities extracted from memory text
and metadata. Supports:
- Entity extraction from text (@mentions, known patterns)
- Relationship extraction (co-occurrence, explicit relation patterns)
- SQLite-backed index for persistence
- Incremental rebuild (scan only memories since last cursor)
- Path traversal and neighborhood queries
- API endpoints for trigger + query
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .app import _connect, _row_to_memory

DEFAULT_STATE_PATH = "/data/mem-graph-state.json"

# Regex patterns
_MENTION_PAT = re.compile(r"@(\w[\w.-]*)")
_ENTITY_PAT = re.compile(
    r"(?:实体|项目|服务|系统|模块|组件|容器|环境|端口|配置|数据库|API|角色|用户)\s*[：:]\s*(\S+)"
)
_RELATION_PAT = re.compile(
    r"(?:关联|依赖|属于|包含|通信|同步|注册|调用|连接|映射)\s*(?:于|到|至|给|与|和)?\s*(\S+)"
)


@dataclass
class GraphBuildState:
    last_run_at: str = ""
    total_memories_scanned: int = 0
    entities_built: int = 0
    relations_built: int = 0
    last_cursor_id: str = ""  # last memory_id processed
    errors: int = 0
    last_error: str | None = None


def _read_state(path: str) -> GraphBuildState:
    p = Path(path)
    if not p.exists():
        return GraphBuildState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        state = GraphBuildState(**{k: raw.get(k, v) for k, v in asdict(GraphBuildState()).items()})
        return state
    except (OSError, ValueError, json.JSONDecodeError):
        return GraphBuildState()


def _write_state(path: str, state: GraphBuildState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _ensure_graph_tables(conn: Any) -> None:
    """Create graph index tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_graph_entities (
            entity TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            memory_id TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            weight INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_graph_entities_entity ON memory_graph_entities(entity);
        CREATE INDEX IF NOT EXISTS idx_graph_entities_memory ON memory_graph_entities(memory_id);

        CREATE TABLE IF NOT EXISTS memory_graph_relations (
            source_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            first_seen TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_relations_source ON memory_graph_relations(source_entity);
        CREATE INDEX IF NOT EXISTS idx_graph_relations_target ON memory_graph_relations(target_entity);
        CREATE INDEX IF NOT EXISTS idx_graph_relations_type ON memory_graph_relations(relation_type);
        CREATE INDEX IF NOT EXISTS idx_graph_relations_memory ON memory_graph_relations(memory_id);
        """
    )


def _extract_entities(text: str, metadata: dict[str, Any] | None = None) -> set[str]:
    """Extract entity names from text and metadata."""
    entities: set[str] = set()

    # From @mentions
    for m in _MENTION_PAT.finditer(text):
        entities.add(m.group(1))

    # From structured entity patterns (e.g. "实体：xxx")
    for m in _ENTITY_PAT.finditer(text):
        entities.add(m.group(1))

    # From explicit entities in metadata
    if metadata:
        meta_entities = metadata.get("entities") or metadata.get("entity") or []
        if isinstance(meta_entities, list):
            for e in meta_entities:
                if isinstance(e, str) and e.strip():
                    entities.add(e.strip())
        elif isinstance(meta_entities, str):
            entities.update(_MENTION_PAT.findall(meta_entities))

        # From @ tags in tags or keywords fields
        for tag_field in ("tags", "keywords", "labels"):
            vals = metadata.get(tag_field)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v.strip():
                        entities.add(v.strip())
            elif isinstance(vals, str):
                entities.update(_MENTION_PAT.findall(vals))

    return entities


def _extract_relations(text: str, entities: set[str]) -> list[tuple[str, str, str]]:
    """Extract (source_entity, relation_type, target_entity) tuples from text."""
    relations: list[tuple[str, str, str]] = []

    if len(entities) < 2:
        return relations

    # For top-3 most frequent entities, check co-occurrence patterns
    for e1 in sorted(entities)[:3]:
        for e2 in sorted(entities)[:3]:
            if e1 >= e2:
                continue
            # Check if there's any relation keyword between them
            pattern = re.compile(
                re.escape(e1) + r".{0,30}?(?:关联|依赖|属于|包含|通信|同步|注册|调用|连接|映射|使用|部署).{0,30}?" + re.escape(e2),
                re.IGNORECASE,
            )
            if pattern.search(text):
                # Try to capture the relation word
                rel_match = re.search(
                    re.escape(e1) + r".{0,30}?(" + "|".join(["关联", "依赖", "属于", "包含", "通信", "同步", "注册", "调用", "连接", "映射", "使用", "部署"]) + r").{0,30}?" + re.escape(e2),
                    text,
                    re.IGNORECASE,
                )
                relation_type = rel_match.group(1) if rel_match else "co_occurs"
                relations.append((e1, relation_type, e2))

    # From explicit relation patterns in metadata
    meta_relations = re.findall(r"([\w.-]+)\s*[→-]{1,2}\s*([\w.-]+)\s*\((\w+)\)", text)
    for src, tgt, rel_type in meta_relations:
        relations.append((src.strip(), rel_type.strip(), tgt.strip()))

    return relations


def _update_scores(conn: Any, entity: str, increment: int = 1) -> None:
    """Increment entity weight across all occurrences."""
    conn.execute(
        "UPDATE memory_graph_entities SET weight = weight + ? WHERE entity = ?",
        (increment, entity),
    )


def _ingest_memory_to_graph(conn: Any, memory: dict[str, Any]) -> tuple[int, int]:
    """Extract entities and relations from a single memory and insert into graph tables.
    Returns (entities_added, relations_added).
    """
    memory_id = memory.get("memory_id")
    if not memory_id:
        return 0, 0

    text = memory.get("text") or memory.get("summary") or ""
    metadata_raw = memory.get("metadata") or memory.get("metadata_json") or "{}"
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, ValueError):
            metadata = {}
    else:
        metadata = metadata_raw or {}

    entities = _extract_entities(text, metadata)
    relations = _extract_relations(text, entities)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    entities_added = 0
    for entity in entities:
        # Check if we already have this memory_id+entity pair
        existing = conn.execute(
            "SELECT 1 FROM memory_graph_entities WHERE entity=? AND memory_id=? LIMIT 1",
            (entity, memory_id),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO memory_graph_entities (entity, memory_id, first_seen) VALUES (?, ?, ?)",
                (entity, memory_id, now),
            )
            entities_added += 1
        # Always update weight
        conn.execute(
            "UPDATE memory_graph_entities SET weight = weight + 1 WHERE entity=? AND memory_id=?",
            (entity, memory_id),
        )

    relations_added = 0
    for src_ent, rel_type, tgt_ent in relations:
        existing = conn.execute(
            "SELECT 1 FROM memory_graph_relations WHERE source_entity=? AND relation_type=? AND target_entity=? AND memory_id=? LIMIT 1",
            (src_ent, rel_type, tgt_ent, memory_id),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO memory_graph_relations (source_entity, relation_type, target_entity, memory_id, first_seen) VALUES (?, ?, ?, ?, ?)",
                (src_ent, rel_type, tgt_ent, memory_id, now),
            )
            relations_added += 1
        conn.execute(
            "UPDATE memory_graph_relations SET weight = weight + 1 WHERE source_entity=? AND relation_type=? AND target_entity=? AND memory_id=?",
            (src_ent, rel_type, tgt_ent, memory_id),
        )

    return entities_added, relations_added


def rebuild_graph(
    *,
    limit: int = 5000,
    state_path: str | None = None,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Scan memories and rebuild/update the graph index.

    Incremental by default: starts from last_cursor_id.
    Use force_rebuild=True to clear and rebuild from scratch.
    """
    state_path = state_path or DEFAULT_STATE_PATH
    state = _read_state(state_path)

    conn = _connect()
    try:
        _ensure_graph_tables(conn)

        if force_rebuild:
            conn.execute("DELETE FROM memory_graph_entities")
            conn.execute("DELETE FROM memory_graph_relations")
            state.last_cursor_id = ""

        # Determine scan range
        raw_state = state.last_cursor_id
        if raw_state and not force_rebuild:
            cursor = conn.execute(
                "SELECT created_at FROM memories WHERE memory_id=?",
                (raw_state,),
            ).fetchone()
            cursor_time = cursor["created_at"] if cursor else "1970-01-01"
            rows = conn.execute(
                "SELECT * FROM memories WHERE created_at >= ? AND COALESCE(deleted,0)=0 ORDER BY created_at ASC LIMIT ?",
                (cursor_time, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories WHERE COALESCE(deleted,0)=0 ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()

        memories = [_row_to_memory(row) for row in rows]
        total_entities = 0
        total_relations = 0
        last_id = raw_state

        for mem in memories:
            if not dry_run:
                e_added, r_added = _ingest_memory_to_graph(conn, mem)
                total_entities += e_added
                total_relations += r_added
            mid = mem.get("memory_id")
            if mid:
                last_id = mid

        if not dry_run:
            conn.commit()

        state.total_memories_scanned += len(memories)
        state.entities_built += total_entities
        state.relations_built += total_relations
        state.last_cursor_id = last_id or state.last_cursor_id
        state.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_state(state_path, state)

    except Exception as exc:
        if not dry_run:
            conn.rollback()
        state.last_error = str(exc)
        state.errors += 1
        _write_state(state_path, state)
        return {
            "accepted": False,
            "error": str(exc),
            "dry_run": dry_run,
        }
    finally:
        conn.close()

    return {
        "accepted": True,
        "dry_run": dry_run,
        "force_rebuild": force_rebuild,
        "memories_scanned": len(memories),
        "new_entities": total_entities,
        "new_relations": total_relations,
        "total_entities_lifetime": state.entities_built,
        "total_relations_lifetime": state.relations_built,
        "total_memories_lifetime": state.total_memories_scanned,
        "last_cursor": state.last_cursor_id,
    }


def query_graph_neighborhood(
    entity: str,
    max_depth: int = 2,
    limit: int = 50,
) -> dict[str, Any]:
    """Return the graph neighborhood around an entity."""
    conn = _connect()
    try:
        _ensure_graph_tables(conn)

        # Find the entity
        entity_row = conn.execute(
            "SELECT entity, weight FROM memory_graph_entities WHERE entity=? LIMIT 1",
            (entity,),
        ).fetchone()
        if not entity_row:
            return {"entity": entity, "found": False, "neighbors": [], "relations": []}

        # Direct neighbors (outgoing)
        outgoing = conn.execute(
            """SELECT r.source_entity, r.relation_type, r.target_entity, r.weight, COUNT(*) as memory_count
               FROM memory_graph_relations r
               WHERE r.source_entity=? OR r.target_entity=?
               GROUP BY r.source_entity, r.relation_type, r.target_entity
               ORDER BY r.weight DESC LIMIT ?""",
            (entity, entity, limit),
        ).fetchall()

        neighbors: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for row in outgoing:
            src = row["source_entity"]
            tgt = row["target_entity"]
            rel = row["relation_type"]
            pair = (src, tgt)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            neighbor = tgt if src == entity else src
            relations.append({
                "source": src,
                "target": tgt,
                "relation_type": rel,
                "weight": row["weight"],
                "memory_count": row["memory_count"],
            })
            if src == entity or tgt == entity:
                neighbors.append({
                    "entity": neighbor,
                    "relation": rel if src == entity else f"inverse_{rel}",
                    "weight": row["weight"],
                })

        # Related memories via this entity
        related_memories = conn.execute(
            """SELECT e.memory_id, e.weight
               FROM memory_graph_entities e
               WHERE e.entity=?
               ORDER BY e.weight DESC LIMIT 20""",
            (entity,),
        ).fetchall()

        memory_refs = [
            {
                "memory_id": row["memory_id"],
                "weight": row["weight"],
            }
            for row in related_memories
        ]

        return {
            "entity": entity,
            "found": True,
            "total_weight": entity_row["weight"],
            "neighbors": neighbors,
            "relations": relations,
            "related_memories": memory_refs,
        }
    finally:
        conn.close()


def query_graph_entities(
    *,
    limit: int = 100,
    min_weight: int = 1,
    sort: str = "weight",
) -> dict[str, Any]:
    """List all entities in the graph with their weights."""
    conn = _connect()
    try:
        _ensure_graph_tables(conn)
        rows = conn.execute(
            "SELECT entity, SUM(weight) as total_weight, COUNT(DISTINCT memory_id) as memory_count "
            "FROM memory_graph_entities "
            "WHERE weight >= ? "
            "GROUP BY entity "
            "ORDER BY total_weight DESC LIMIT ?",
            (min_weight, limit),
        ).fetchall()

        entities = [
            {
                "entity": row["entity"],
                "total_weight": row["total_weight"],
                "memory_count": row["memory_count"],
            }
            for row in rows
        ]

        return {"total": len(entities), "entities": entities}
    finally:
        conn.close()


def query_graph_stats() -> dict[str, Any]:
    """Return summary statistics about the graph."""
    conn = _connect()
    try:
        _ensure_graph_tables(conn)
        entity_count = conn.execute(
            "SELECT COUNT(DISTINCT entity) as cnt FROM memory_graph_entities"
        ).fetchone()["cnt"]
        relation_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM memory_graph_relations"
        ).fetchone()["cnt"]
        top_entities = conn.execute(
            "SELECT entity, SUM(weight) as total_weight FROM memory_graph_entities "
            "GROUP BY entity ORDER BY total_weight DESC LIMIT 10"
        ).fetchall()
        top_relations = conn.execute(
            "SELECT source_entity, relation_type, target_entity, SUM(weight) as total_weight "
            "FROM memory_graph_relations "
            "GROUP BY source_entity, relation_type, target_entity "
            "ORDER BY total_weight DESC LIMIT 10"
        ).fetchall()

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "top_entities": [
                {"entity": r["entity"], "weight": r["total_weight"]} for r in top_entities
            ],
            "top_relations": [
                {"source": r["source_entity"], "relation": r["relation_type"], "target": r["target_entity"], "weight": r["total_weight"]}
                for r in top_relations
            ],
        }
    finally:
        conn.close()


def read_graph_state(state_path: str | None = None) -> dict[str, Any]:
    """Read current graph build state."""
    state = _read_state(state_path or DEFAULT_STATE_PATH)
    return asdict(state)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NTN MEM Graph Index")
    sub = parser.add_subparsers(dest="command", required=True)

    rebuild_parser = sub.add_parser("rebuild", help="Rebuild graph index")
    rebuild_parser.add_argument("--force", action="store_true", help="Clear and rebuild from scratch")
    rebuild_parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    rebuild_parser.add_argument("--limit", type=int, default=5000, help="Max memories to scan")
    rebuild_parser.add_argument("--state-path", default=DEFAULT_STATE_PATH)

    query_parser = sub.add_parser("query", help="Query entity neighborhood")
    query_parser.add_argument("entity", help="Entity name to query")
    query_parser.add_argument("--depth", type=int, default=2)
    query_parser.add_argument("--limit", type=int, default=50)

    entities_parser = sub.add_parser("entities", help="List all entities")
    entities_parser.add_argument("--limit", type=int, default=100)
    entities_parser.add_argument("--min-weight", type=int, default=1)

    stats_parser = sub.add_parser("stats", help="Graph statistics")

    args = parser.parse_args()

    if args.command == "rebuild":
        result = rebuild_graph(
            limit=args.limit,
            state_path=args.state_path,
            dry_run=args.dry_run,
            force_rebuild=args.force,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "query":
        result = query_graph_neighborhood(args.entity, max_depth=args.depth, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "entities":
        result = query_graph_entities(limit=args.limit, min_weight=args.min_weight)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "stats":
        result = query_graph_stats()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
