"""HTTP server entry point for MEM WSGI app."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
from wsgiref.simple_server import WSGIServer
from wsgiref.simple_server import make_server as _make_server

from .app import application
from .logger import log_error, log_info, log_warn

_log = logging.getLogger("ntn-mem.embed-worker")

# ── embedding background worker ────────────────────────────────────────────
# Consumes the embedding_status=pending queue so newly written (or post-restart)
# memories get real vectors without waiting for a search-request trigger.

_EMBED_POLL_INTERVAL = 10  # seconds between polls
_EMBED_BATCH_SIZE = 20     # max texts to embed per poll cycle


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("NTN_MEM_DB", "/data/mem.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _embed_worker() -> None:
    """Background thread: poll for pending embeddings and process them."""
    try:
        from .embedding import get_embedding_provider
        from .qdrant import get_qdrant_client
    except ImportError:
        _log.error("embedding worker: failed to import modules")
        return

    # Warm up provider + qdrant once
    try:
        provider = get_embedding_provider()
        qdrant = get_qdrant_client()
    except Exception as exc:
        _log.warning("embedding worker: provider/qdrant init failed (%s), retrying next cycle", exc)
        log_warn(module="embed_worker", operation="init",
                 summary=f"provider/qdrant init failed", exc=exc)
        provider = qdrant = None

    _log.info(
        "embedding worker started (provider=%s, qdrant=%s)",
        getattr(provider, "name", "none") if provider else "none",
        "available" if qdrant else "unavailable",
    )
    log_info(module="embed_worker", operation="start",
             summary=f"provider={getattr(provider,'name','none')}, qdrant={'available' if qdrant else 'unavailable'}")


    while True:
        try:
            conn = _connect()
            rows = conn.execute(
                "SELECT memory_id, text, project_id, role, owner_role, "
                "source_role, scope, visibility, status, layer, memory_type "
                "FROM memories WHERE embedding_status='pending' "
                "ORDER BY created_at ASC LIMIT ?",
                (_EMBED_BATCH_SIZE,),
            ).fetchall()
            if not rows:
                conn.close()
                time.sleep(_EMBED_POLL_INTERVAL)
                continue

            _log.info("embedding worker: processing %d pending memories", len(rows))
            t_batch = time.time()

            for row in rows:
                text = row["text"]
                memory_id = row["memory_id"]
                t_item = time.time()
                try:
                    if provider is not None:
                        vector = provider.embed(text)
                        # Save real vector to embedding_cache
                        text_hash = _text_hash(f"{provider.name}:{text}")
                        existing = conn.execute(
                            "SELECT text_hash FROM embedding_cache WHERE text_hash=?",
                            (text_hash,),
                        ).fetchone()
                        if existing is None:
                            conn.execute(
                                "INSERT INTO embedding_cache "
                                "(text_hash, provider, embedding_json, created_at) VALUES (?, ?, ?, ?)",
                                (text_hash, provider.name, json.dumps(vector), _now_iso()),
                            )
                        # Upsert to Qdrant if available
                        if qdrant is not None:
                            try:
                                qdrant.upsert(
                                    memory_id,
                                    vector,
                                    {
                                        "memory_id": memory_id,
                                        "project_id": row["project_id"],
                                        "role": row["role"],
                                        "owner_role": row["owner_role"],
                                        "source_role": row["source_role"],
                                        "scope": row["scope"],
                                        "visibility": row["visibility"],
                                        "status": row["status"],
                                        "layer": row["layer"],
                                        "memory_type": row["memory_type"],
                                        "created_at": _now_iso(),
                                    },
                                )
                                log_info(module="embed_worker", operation="embed_success",
                                         summary=f"memory {memory_id[:12]} embedded ({row.get('project_id','?')})",
                                         duration_ms=int((time.time() - t_item) * 1000))
                            except Exception as exc:
                                _log.warning("qdrant upsert failed for %s: %s", memory_id, exc)
                                log_warn(module="embed_worker", operation="qdrant_upsert",
                                         summary=f"failed for {memory_id[:12]}", exc=exc)

                        # Mark pending→completed
                        conn.execute(
                            "UPDATE memories SET embedding_status='completed' WHERE memory_id=?",
                            (memory_id,),
                        )
                except Exception as exc:
                    # Don't retry immediately on failure; mark as errored to avoid busy-loop
                    _log.warning("embedding failed for %s: %s", memory_id, exc)
                    log_error(module="embed_worker", operation="embed_failed",
                              summary=f"memory {memory_id[:12]} failed",
                              exc=exc, extra={"memory_id": memory_id})
                    conn.execute(
                        "UPDATE memories SET embedding_status='errored' WHERE memory_id=?",
                        (memory_id,),
                    )
            conn.commit()
            conn.close()
            log_info(module="embed_worker", operation="embed_batch",
                     summary=f"processed {len(rows)} memories",
                     duration_ms=int((time.time() - t_batch) * 1000))

        except Exception as exc:
            _log.error("embedding worker cycle failed: %s", exc)
            log_error(module="embed_worker", operation="cycle_failed",
                      summary="Embedding worker cycle crashed",
                      exc=exc)
            time.sleep(_EMBED_POLL_INTERVAL * 3)  # back off on errors

        time.sleep(0.5)  # small delay between batches


def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_server(host: str = "0.0.0.0", port: int = 8081) -> WSGIServer:
    return _make_server(host, port, application)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ntn-mem")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args(argv)

    # Start embedding background worker
    thread = threading.Thread(target=_embed_worker, daemon=True, name="embed-worker")
    thread.start()
    _log.info("embedding worker thread started")

    server = make_server(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
