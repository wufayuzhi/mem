#!/usr/bin/env python3
"""Migrate strategic knowledge docs to the 'knowledge_reserve' project.

Objective:
  Consolidate all operator-curated knowledge (Hermes Agent docs, CCB docs, etc.)
  under a single project so it's cleanly separated from per-agent private memory
  and daily experience extraction.

Source projects (from SQLite):
  - hermesagent-cn   (2033 records)
  - ccb-docs         (1221 records)

Target project:
  - knowledge_reserve   (scope=shared, memory_type=knowledge_doc)

Method:
  Phase A — SQLite UPDATE (fast, idempotent)
  Phase B — Qdrant payload update for the affected points
  Phase C — Verify

Run with --dry-run to preview changes.
Run with --execute to apply.
Run with --verify to check after execution.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from typing import Any

from urllib.request import Request, urlopen

# Config
MEM_DB = os.environ.get("NTN_MEM_DB", "/data/mem.db")
QDRANT_URL = os.environ.get("NTN_QDRANT_URL", "http://10.69.68.15:6333")
SOURCE_PROJECTS = ("hermesagent-cn", "ccb-docs")
TARGET_PROJECT = "knowledge_reserve"

# Scope and type overrides for the migrated records
TARGET_SCOPE = "shared"
TARGET_MEMORY_TYPE = "knowledge_doc"


def _memories_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MEM_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _qdrant_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{QDRANT_URL.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {}, method=method)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def dry_run() -> None:
    conn = _memories_conn()
    try:
        total = 0
        for src in SOURCE_PROJECTS:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
                (src,),
            ).fetchone()[0]
            print(f"  {src}: {count} records will migrate to '{TARGET_PROJECT}'")
            total += count

        # Check if target already has records
        existing = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
            (TARGET_PROJECT,),
        ).fetchone()[0]
        print(f"\n  '{TARGET_PROJECT}' already has {existing} records (will add {total} more)")

        # Check Qdrant
        try:
            qdrant_count = len(_qdrant_request("POST", "/collections/ntn_memories/points/scroll", {
                "limit": 5,
                "filter": {"must": [{"key": "project_id", "match": {"value": TARGET_PROJECT}}]},
                "with_payload": False,
            }).get("result", {}).get("points", []))
            print(f"  Qdrant: '{TARGET_PROJECT}' has some points (sample shows {qdrant_count}+)")
        except Exception as e:
            print(f"  Qdrant check failed: {e}")
    finally:
        conn.close()


def execute() -> None:
    conn = _memories_conn()
    total_migrated = 0
    try:
        for src in SOURCE_PROJECTS:
            # Get memory_ids for this source project
            rows = conn.execute(
                "SELECT memory_id, text, scope, memory_type FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
                (src,),
            ).fetchall()

            ids = [r["memory_id"] for r in rows]
            if not ids:
                print(f"  {src}: no records to migrate")
                continue

            # Phase A: SQLite UPDATE
            placeholders = ",".join("?" for _ in ids)
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"UPDATE memories SET project_id=?, scope=?, memory_type=?, updated_at=? WHERE memory_id IN ({placeholders})",
                (TARGET_PROJECT, TARGET_SCOPE, TARGET_MEMORY_TYPE, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *ids),
            )
            conn.execute("COMMIT")
            print(f"  {src}: SQLite migrated {len(ids)} records -> '{TARGET_PROJECT}'")
            total_migrated += len(ids)

            # Phase B: Qdrant payload update
            qdrant_points = []
            for mid in ids:
                qdrant_points.append({
                    "id": str(mid.replace("mem-", "", 1) if mid.startswith("mem-") else mid),
                    "payload": {
                        "project_id": TARGET_PROJECT,
                    },
                })

            # Qdrant set_payload in batches
            batch_size = 500
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i : i + batch_size]
                try:
                    result = _qdrant_request("POST", "/collections/ntn_memories/points/payload", {
                        "payload": {"project_id": TARGET_PROJECT},
                        "points": [p["id"] for p in batch],
                    })
                    print(f"    Qdrant batch {i//batch_size + 1}: {len(batch)} points updated")
                except Exception as e:
                    print(f"    Qdrant batch {i//batch_size + 1} failed: {e}")

            # Also overwrite old points (clear old project_id by setting the new one)
            # Qdrant set_payload UPSERTS — old project_id is replaced
            print(f"    Qdrant done for {src}")

    finally:
        conn.close()

    print(f"\n  Total migrated: {total_migrated} records -> '{TARGET_PROJECT}'")


def verify() -> None:
    conn = _memories_conn()
    try:
        print(f"=== Verification ===")

        # SQLite counts
        for src in SOURCE_PROJECTS:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
                (src,),
            ).fetchone()[0]
            print(f"  {src}: {count} records (should be 0)")

        target_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project_id=? AND COALESCE(deleted,0)=0",
            (TARGET_PROJECT,),
        ).fetchone()[0]
        print(f"  {TARGET_PROJECT}: {target_count} records (should be ~3254)")

        # Qdrant spot check
        try:
            result = _qdrant_request("POST", "/collections/ntn_memories/points/scroll", {
                "limit": 3,
                "filter": {"must": [{"key": "project_id", "match": {"value": TARGET_PROJECT}}]},
                "with_payload": True,
            })
            points = result.get("result", {}).get("points", [])
            print(f"  Qdrant: {len(points)} sample points with project_id={TARGET_PROJECT}")
            for p in points:
                print(f"    ID: {p['id']}, payload: {json.dumps(p.get('payload', {}), ensure_ascii=False)}")
        except Exception as e:
            print(f"  Qdrant check failed: {e}")
    finally:
        conn.close()


def main() -> int:
    if "--dry-run" in sys.argv:
        print(f"=== DRY RUN: would migrate {SOURCE_PROJECTS} -> '{TARGET_PROJECT}' ===\n")
        dry_run()
        print("\n  Run with --execute to apply.")
        return 0

    if "--execute" in sys.argv:
        print(f"=== Executing migration: {SOURCE_PROJECTS} -> '{TARGET_PROJECT}' ===\n")
        execute()
        print("\n  Run with --verify to check results.")
        return 0

    if "--verify" in sys.argv:
        verify()
        return 0

    print(f"Usage: {sys.argv[0]} [--dry-run | --execute | --verify]")
    print(f"  --dry-run   Preview changes")
    print(f"  --execute   Apply migration (irreversible)")
    print(f"  --verify    Check migration results")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
