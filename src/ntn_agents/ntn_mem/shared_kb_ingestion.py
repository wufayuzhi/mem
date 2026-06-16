"""Shared KB document ingestion for NTN MEM.

Scans configured document directories and ingests files as ``memory_type=shared_kb``
memories with scope=shared, suitable for cross-agent knowledge retrieval.

Supports:
- Recursive directory scanning with glob patterns
- File extension-based type detection (md, txt, yaml, json)
- Deduplication via content hash + file path
- Configurable watch paths from env/config
- Manual trigger via API
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .app import _connect, _row_to_memory

DEFAULT_STATE_PATH = "/data/mem-shared-kb-state.json"
DEFAULT_WATCH_PATHS = [
    "/mnt/shared/agents/ntn-agents/4-代码/docs",
    "/mnt/shared/agents/ntn-agents/4-代码/README.md",
]
DEFAULT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Environment overrides
ENV_WATCH_PATHS = os.environ.get("NTN_MEM_SHARED_KB_WATCH", "")
ENV_EXTENSIONS = os.environ.get("NTN_MEM_SHARED_KB_EXTENSIONS", "")


def _parse_watch_paths() -> list[str]:
    if ENV_WATCH_PATHS:
        return [p.strip() for p in ENV_WATCH_PATHS.split(",") if p.strip()]
    return DEFAULT_WATCH_PATHS


def _parse_extensions() -> set[str]:
    if ENV_EXTENSIONS:
        return {f".{e.strip().lstrip('.')}" for e in ENV_EXTENSIONS.split(",") if e.strip()}
    return DEFAULT_EXTENSIONS


@dataclass
class IngestState:
    last_run_at: str = ""
    total_files_scanned: int = 0
    files_ingested: int = 0
    memories_created: int = 0
    errors: int = 0
    file_hashes_ingested: list[str] | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.file_hashes_ingested is None:
            self.file_hashes_ingested = []


def _read_state(path: str) -> IngestState:
    p = Path(path)
    if not p.exists():
        return IngestState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        state = IngestState(**{k: raw.get(k, v) for k, v in asdict(IngestState()).items()})
        if state.file_hashes_ingested is None:
            state.file_hashes_ingested = []
        return state
    except (OSError, ValueError, json.JSONDecodeError):
        return IngestState()


def _write_state(path: str, state: IngestState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _content_hash(content: str, file_path: str) -> str:
    """Return a stable hash that survives file renames."""
    return hashlib.sha256((content + "::" + file_path).encode("utf-8")).hexdigest()[:16]


def _chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split a long document into chunks for separate memories."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    # Try splitting by headings first
    parts = re.split(r"\n(#{1,3}\s)", text)
    current = ""
    for part in parts:
        if part.startswith("#") and part.strip().endswith(" "):
            # heading marker, will combine with next part
            current = part
        else:
            current += part
            if len(current) >= max_chars * 0.8 or len(current) + len(part) > max_chars:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())

    # If still too long, split at paragraph boundaries
    if len(chunks) == 1 and len(chunks[0]) > max_chars:
        chunks = []
        paragraphs = text.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > max_chars:
                if current.strip():
                    chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())

    return chunks


def _detect_doc_type(file_path: str) -> str:
    """Infer document type from extension."""
    ext = Path(file_path).suffix.lower()
    type_map = {
        ".md": "markdown",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
    }
    return type_map.get(ext, "unknown")


def _read_document(file_path: str) -> str | None:
    """Read a document file, returning None on error."""
    try:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        if p.stat().st_size > DEFAULT_MAX_FILE_SIZE:
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None


def ingest_documents(
    *,
    watch_paths: list[str] | None = None,
    extensions: set[str] | None = None,
    state_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan directories and ingest new/changed documents as shared_kb memories.

    Returns a summary dict with counts and any errors.
    """
    watch_paths = watch_paths or _parse_watch_paths()
    extensions = extensions or _parse_extensions()
    state_path = state_path or DEFAULT_STATE_PATH
    state = _read_state(state_path)

    conn = _connect()
    try:
        files_scanned = 0
        files_ingested = 0
        memories_created = 0
        errors = 0
        error_details: list[str] = []

        for watch_path_str in watch_paths:
            watch_path = Path(watch_path_str)
            if not watch_path.exists():
                error_details.append(f"watch_path not found: {watch_path_str}")
                errors += 1
                continue

            if watch_path.is_file():
                target_files = [watch_path]
            else:
                target_files = []
                for ext in extensions:
                    target_files.extend(watch_path.rglob(f"*{ext}"))

            for file_path in target_files:
                files_scanned += 1
                content = _read_document(str(file_path))
                if content is None:
                    continue

                # Check dedup
                content_hash_str = _content_hash(content, str(file_path))
                if content_hash_str in (state.file_hashes_ingested or []):
                    continue

                if dry_run:
                    memories_created += 1
                    continue

                doc_type = _detect_doc_type(str(file_path))
                relative_path = str(file_path.relative_to(watch_path)) if watch_path.is_dir() else str(file_path.name)
                title = file_path.stem.replace("_", " ").replace("-", " ").title()

                # Chunk large documents
                chunks = _chunk_text(content)
                for i, chunk in enumerate(chunks):
                    summary_prefix = chunk.strip()[:120].replace("\n", " ")

                    conn.execute(
                        """INSERT INTO memories
                        (memory_id, text, summary, memory_type, scope, project_id,
                         owner_role, source_role, layer, temperature, importance,
                         status, visibility, metadata, created_at, updated_at)
                        VALUES (?, ?, ?, 'shared_kb', 'shared', 'default',
                                'system', 'system', 'long_term', 'warm', 40,
                                'verified', 'shared_acl',
                                json(?), datetime('now'), datetime('now'))""",
                        (
                            _generate_memory_id(),
                            chunk,
                            f"[KB] {title}: {summary_prefix}",
                            json.dumps({
                                "doc_path": str(file_path),
                                "doc_type": doc_type,
                                "chunk": i if len(chunks) > 1 else None,
                                "total_chunks": len(chunks) if len(chunks) > 1 else None,
                                "content_hash": content_hash_str,
                                "title": title,
                                "relative_path": relative_path,
                            }, ensure_ascii=False),
                        ),
                    )
                    memories_created += 1

                state.file_hashes_ingested.append(content_hash_str)
                files_ingested += 1

        conn.commit()
    except Exception as exc:
        conn.rollback()
        state.last_error = str(exc)
        errors += 1
        error_details.append(str(exc))
    finally:
        conn.close()

    state.total_files_scanned += files_scanned
    state.files_ingested += files_ingested
    state.memories_created += memories_created
    state.errors += errors
    state.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_state(state_path, state)

    return {
        "dry_run": dry_run,
        "files_scanned": files_scanned,
        "files_ingested": files_ingested,
        "memories_created": memories_created,
        "errors": errors,
        "error_details": error_details if error_details else None,
        "total_files_lifetime": state.files_ingested,
        "total_memories_lifetime": state.memories_created,
    }


def read_ingest_state(state_path: str | None = None) -> dict[str, Any]:
    """Read current ingest state without triggering a scan."""
    state = _read_state(state_path or DEFAULT_STATE_PATH)
    return asdict(state)


def _generate_memory_id() -> str:
    """Generate a deterministic-ish memory_id for ingested docs."""
    import uuid
    return f"kb-{uuid.uuid4().hex[:12]}"


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents as shared_kb memories")
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--watch-paths", nargs="*", help="Override watch paths")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="State file path")

    args = parser.parse_args()
    result = ingest_documents(
        watch_paths=args.watch_paths or None,
        state_path=args.state_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
