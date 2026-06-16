"""Memory-tiering governance for NTN MEM.

This module is intentionally inside the MEM service boundary. Hermes plugins
only push/pull through MEM APIs; tiering decisions (layer, temperature,
retention, retrieval priority) are governed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


HOT = "hot"
WARM = "warm"
COLD = "cold"
PROTECTED = "protected"
DECAYED = "decayed"

EPISODIC = "episodic"
SEMANTIC = "semantic"
PROCEDURAL = "procedural"
SHARED_KB = "shared_kb"
TASK_CONTEXT = "task_context"


@dataclass(frozen=True)
class TierDecision:
    """Normalized governance decision persisted with each memory."""

    layer: str
    temperature: str
    memory_type: str
    importance: int
    protected: bool
    archived: bool
    expires_at: str | None


def _clamp_importance(value: Any, default: int = 50) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(0, min(100, number))


def _text_has_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def classify_memory(data: dict[str, Any]) -> TierDecision:
    """Classify inbound memory into tiering dimensions.

    Explicit caller-provided values win when safe. Otherwise the policy derives
    a conservative default from scope/source/type/importance:
    - task handoff/context -> episodic + hot/mid_term
    - stable facts/preferences/config/API conventions -> semantic + warm/long_term
    - procedures/how-to/fixes -> procedural + warm/long_term
    - shared knowledge -> shared_kb + warm/long_term
    - archived/low-importance -> cold
    - protected=true prevents lifecycle decay/auto-expiry
    """

    text = str(data.get("text") or data.get("summary") or "")
    scope = str(data.get("scope") or "role_private")
    explicit_type = data.get("memory_type") or data.get("type") or data.get("kind")
    memory_type = str(explicit_type or "").strip()

    if not memory_type:
        if scope == "task_context" or data.get("task_id"):
            memory_type = TASK_CONTEXT
        elif scope == "shared" or scope == "company":
            memory_type = SHARED_KB
        elif _text_has_any(text, ("步骤", "procedure", "workflow", "runbook", "如何", "命令", "修复流程")):
            memory_type = PROCEDURAL
        elif _text_has_any(text, ("偏好", "配置", "约定", "api", "schema", "事实", "remember", "记住")):
            memory_type = SEMANTIC
        else:
            memory_type = EPISODIC

    importance = _clamp_importance(data.get("importance") or data.get("vitality"), default=50)
    protected = bool(data.get("protected") or data.get("is_protected"))
    archived = bool(data.get("archived"))

    explicit_temperature = str(data.get("temperature") or "").strip().lower()
    if explicit_temperature in {HOT, WARM, COLD, PROTECTED, DECAYED}:
        temperature = explicit_temperature
    elif protected:
        temperature = PROTECTED
    elif archived or importance <= 20:
        temperature = COLD
    elif scope == "task_context" or memory_type == TASK_CONTEXT or importance >= 80:
        temperature = HOT
    else:
        temperature = WARM

    explicit_layer = str(data.get("layer") or "").strip()
    if explicit_layer:
        layer = explicit_layer
    elif temperature == COLD:
        layer = "cold"
    elif memory_type in {SEMANTIC, PROCEDURAL, SHARED_KB} or protected:
        layer = "long_term"
    else:
        layer = "mid_term"

    expires_at = data.get("expires_at")
    if protected:
        expires_at = None

    return TierDecision(
        layer=layer,
        temperature=temperature,
        memory_type=memory_type,
        importance=importance,
        protected=protected,
        archived=archived,
        expires_at=str(expires_at) if expires_at else None,
    )


def retrieval_weight(memory: dict[str, Any]) -> float:
    """Return an additive ranking weight for governed recall ordering."""

    temperature = str(memory.get("temperature") or WARM).lower()
    weight = {
        PROTECTED: 0.40,
        HOT: 0.30,
        WARM: 0.15,
        COLD: -0.10,
        DECAYED: -0.50,
    }.get(temperature, 0.0)
    try:
        importance = int(memory.get("importance") or memory.get("vitality") or 0)
    except (TypeError, ValueError):
        importance = 0
    weight += max(0, min(100, importance)) / 1000.0
    if memory.get("archived"):
        weight -= 0.05
    if memory.get("protected"):
        weight += 0.10
    return weight


def is_expired(memory: dict[str, Any], now: datetime | None = None) -> bool:
    """True when a non-protected memory is past expires_at."""

    if memory.get("protected"):
        return False
    expires_at = memory.get("expires_at")
    if not expires_at:
        return False
    current = now or datetime.now(timezone.utc)
    try:
        value = str(expires_at).replace("Z", "+00:00")
        return datetime.fromisoformat(value) <= current
    except ValueError:
        return False
