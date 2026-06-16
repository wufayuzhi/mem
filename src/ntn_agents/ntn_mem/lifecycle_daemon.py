"""Long-running lifecycle daemon for NTN MEM governance.

Runs periodic lifecycle + supersede sweeps as a background process,
persisting cursor and configuration to disk so it survives restarts.
Designed to be launched as a sidecar alongside the MEM HTTP server.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .app import run_lifecycle, run_supersede

DEFAULT_STATE_PATH = "/data/mem-lifecycle-state.json"
DEFAULT_INTERVAL = 300  # 5 minutes between sweeps
DEFAULT_BATCH_LIMIT = 5000


@dataclass
class LifecycleState:
    last_run_at: str = ""  # ISO timestamp
    total_scanned: int = 0
    total_decayed: int = 0
    total_superseded: int = 0
    lifecycle_count: int = 0
    supersede_count: int = 0
    run_errors: int = 0
    last_error: str | None = None

    def record_lifecycle(self, result: dict[str, Any]) -> None:
        self.lifecycle_count += 1
        self.total_scanned += result.get("scanned", 0)
        self.total_decayed += result.get("decayed", 0)
        self.last_run_at = result.get("_completed_at", "")
        self.last_error = None

    def record_supersede(self, result: dict[str, Any]) -> None:
        self.supersede_count += 1
        self.total_superseded += result.get("superseded", 0)
        self.last_error = None

    def record_error(self, error: str) -> None:
        self.run_errors += 1
        self.last_error = error[:500]


def _read_state(path: str) -> LifecycleState:
    p = Path(path)
    if not p.exists():
        return LifecycleState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return LifecycleState(**{k: raw.get(k, v) for k, v in asdict(LifecycleState()).items()})
    except (OSError, ValueError, json.JSONDecodeError):
        return LifecycleState()


def _write_state(path: str, state: LifecycleState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class LifecycleDaemon:
    state_path: str = DEFAULT_STATE_PATH
    interval_seconds: int = DEFAULT_INTERVAL
    lifecycle_batch_limit: int = DEFAULT_BATCH_LIMIT
    supersede_enabled: bool = True
    lifecycle_config: dict[str, Any] | None = None
    dry_run: bool = False

    def run_once(self) -> dict[str, Any]:
        state = _read_state(self.state_path)
        try:
            lc_result = run_lifecycle({
                "limit": self.lifecycle_batch_limit,
                "dry_run": self.dry_run,
                "hot_days": (self.lifecycle_config or {}).get("hot_days", 7),
                "warm_days": (self.lifecycle_config or {}).get("warm_days", 30),
                "actor_role": "mem-lifecycle-daemon",
                "_completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            state.record_lifecycle(lc_result)
        except Exception as exc:
            state.record_error(f"lifecycle: {exc}")

        ss_result: dict[str, Any] | None = None
        if self.supersede_enabled:
            try:
                ss_result = run_supersede({
                    "dry_run": self.dry_run,
                    "actor_role": "mem-lifecycle-daemon",
                })
                state.record_supersede(ss_result)
            except Exception as exc:
                state.record_error(f"supersede: {exc}")

        _write_state(self.state_path, state)
        return {"lifecycle": lc_result, "supersede": ss_result}

    def run_once_resolved(self) -> dict[str, Any]:
        """Extended run_once that also runs procedural→skill and graph rebuild."""
        core = self.run_once()
        skill_result: dict[str, Any] | None = None
        try:
            from .procedural_to_skill import generate_skills
            skill_result = generate_skills()
        except Exception as exc:
            skill_result = {"error": str(exc)}
        graph_result: dict[str, Any] | None = None
        try:
            from .graph_index import rebuild_graph
            graph_result = rebuild_graph(force_rebuild=False)
        except Exception as exc:
            graph_result = {"error": str(exc)}
        return {**core, "procedural_to_skill": skill_result, "graph_rebuild": graph_result}

    def run_forever(self) -> None:
        while True:
            self.run_once_resolved()
            time.sleep(self.interval_seconds)


def read_lifecycle_state(state_path: str = DEFAULT_STATE_PATH) -> dict[str, Any]:
    """Read current lifecycle daemon state without modifying it."""
    state = _read_state(state_path)
    return asdict(state)


def main() -> None:
    daemon = LifecycleDaemon(
        state_path=os.environ.get("NTN_MEM_LIFECYCLE_STATE", DEFAULT_STATE_PATH),
        interval_seconds=_env_int("NTN_MEM_LIFECYCLE_INTERVAL", DEFAULT_INTERVAL),
        lifecycle_batch_limit=_env_int("NTN_MEM_LIFECYCLE_LIMIT", DEFAULT_BATCH_LIMIT),
        supersede_enabled=os.environ.get("NTN_MEM_LIFECYCLE_SUPERSEDE", "1") == "1",
        dry_run=os.environ.get("NTN_MEM_LIFECYCLE_DRY_RUN", "0") == "1",
        lifecycle_config={
            "hot_days": _env_int("NTN_MEM_LIFECYCLE_HOT_DAYS", 7),
            "warm_days": _env_int("NTN_MEM_LIFECYCLE_WARM_DAYS", 30),
        },
    )
    daemon.run_forever()


if __name__ == "__main__":
    main()
