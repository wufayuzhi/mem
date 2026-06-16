"""Batch re-embed: regenerate 1024-dimensional vectors for all memories
that currently have only local-hash (16d) or pending embeddings.

Usage:
    python3 -m ntn_agents.ntn_mem.scripts.reembed [--batch-size 50] [--limit 0] [--force]

This script reads from the SQLite mem.db, generates vectors via the configured
embedding provider (siliconflow BAAI/bge-m3), writes to Qdrant, and caches the
vectors in embedding_cache for future search-time use.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def _db_path() -> str:
    return os.environ.get("NTN_MEM_DB", "/data/mem.db")


def _load_secrets_env() -> None:
    """Load EnvironmentFile and Environment lines from the ntn-mem service."""
    env_path = "/etc/ntn-agents/secrets.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    if key not in os.environ:
                        os.environ[key.strip()] = val.strip()

    svc_path = "/etc/systemd/system/ntn-mem.service"
    if os.path.exists(svc_path):
        with open(svc_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Environment="):
                    kv = line[len("Environment="):].strip()
                    if kv and "=" in kv:
                        key, _, val = kv.partition("=")
                        if key not in os.environ:
                            os.environ[key.strip()] = val.strip()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Batch re-embed all pending memories")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default 50)")
    parser.add_argument("--limit", type=int, default=0, help="Max memories to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-embed even already-processed memories")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between batches (default 0.2s)")
    args = parser.parse_args()

    # Load secrets from EnvironmentFile (used by the ntn-mem service).
    _load_secrets_env()

    # Import MEM modules (standalone context).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from ntn_agents.ntn_mem.embedding import get_embedding_provider
    from ntn_agents.ntn_mem.qdrant import get_qdrant_client

    # Resolve embedding provider.
    provider = get_embedding_provider()
    print(f"Embedding provider: {type(provider).__name__} (name={provider.name})")

    # Resolve Qdrant client.
    qdrant = get_qdrant_client()
    if qdrant is None:
        print("WARNING: Qdrant not configured — will only cache embeddings in SQLite")
    else:
        print(f"Qdrant: {qdrant.url} collection={qdrant.collection}")

    # Connect to MEM DB.
    db = _db_path()
    print(f"SQLite DB: {db} ({os.path.getsize(db) / 1024 / 1024:.0f} MB)")

    conn = sqlite3.connect(db, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Find memories that need embedding.
    if args.force:
        rows = conn.execute(
            "SELECT memory_id, text, project_id FROM memories WHERE COALESCE(deleted,0)=0 ORDER BY created_at ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT memory_id, text, project_id FROM memories WHERE COALESCE(deleted,0)=0 "
            "AND (embedding_status IS NULL OR embedding_status='pending') "
            "ORDER BY created_at ASC"
        ).fetchall()

    if args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"\nFound {total} memories to re-embed\n")

    processed = 0
    errors = 0
    cached = 0
    qdranted = 0
    start_time = time.time()

    for i in range(0, total, args.batch_size):
        batch = rows[i : i + args.batch_size]
        batch_texts = [r["text"] for r in batch]

        # Generate vectors in batch.
        batch_vectors: list[list[float] | None] = []
        for text in batch_texts:
            try:
                vector = provider.embed(text)
                batch_vectors.append(vector)
            except Exception as exc:
                print(f"  Embedding failed for batch item: {exc}")
                batch_vectors.append(None)

        # Process each result.
        for j, row in enumerate(batch):
            memory_id = row["memory_id"]
            text = row["text"]
            vector = batch_vectors[j]
            project_id = row["project_id"]

            if vector is None:
                errors += 1
                continue

            # Write to embedding_cache.
            try:
                th = _text_hash(text)
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(text_hash, provider, embedding_json, created_at) VALUES (?, ?, ?, ?)",
                    (
                        th,
                        provider.name,
                        json.dumps(vector, ensure_ascii=False),
                        _now_iso(),
                    ),
                )
                conn.commit()
                cached += 1
            except Exception as exc:
                print(f"  Cache write error {memory_id}: {exc}")

            # Update embedding_status.
            try:
                conn.execute(
                    "UPDATE memories SET embedding_status='completed' WHERE memory_id=?",
                    (memory_id,),
                )
                conn.commit()
            except Exception:
                pass

            # Upsert to Qdrant.
            if qdrant is not None:
                try:
                    qdrant.upsert(
                        memory_id,
                        vector,
                        {
                            "memory_id": memory_id,
                            "project_id": project_id,
                        },
                    )
                    qdranted += 1
                except Exception as exc:
                    print(f"  Qdrant upsert error {memory_id}: {exc}")

            processed += 1

        # Progress.
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        pct = 100.0 * processed / total if total else 100.0
        print(
            f"  [{processed}/{total} {pct:.0f}%] "
            f"err={errors} cached={cached} qdrant={qdranted} "
            f"{rate:.1f} mem/s ETA {eta:.0f}s"
        )

        if args.delay > 0 and i + args.batch_size < total:
            time.sleep(args.delay)

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nDone: {processed} processed, {errors} errors, {cached} cached, {qdranted} qdranted in {elapsed:.0f}s")


def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
