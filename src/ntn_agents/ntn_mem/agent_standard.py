"""Standard Agent memory onboarding and lifecycle policy.

This module defines the MEM-side standard flow for any new Agent that connects
through /v1/agents/memory/push and /pull. It deliberately stays inside the MEM
service boundary: clients provide agent_key + optional projects; MEM normalizes
identity, namespaces, and tiering/lifecycle defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

DEFAULT_SHARED_PROJECTS = ("knowledge_reserve", "experience_reserve")


@dataclass(frozen=True)
class AgentMemoryProfile:
    agent_key: str
    private_memory_project: str
    shared_knowledge_projects: list[str]
    push_endpoint: str = "/v1/agents/memory/push"
    pull_endpoint: str = "/v1/agents/memory/pull"
    tiering: str = "auto"
    lifecycle: str = "auto"
    acl_mode: str = "private_plus_shared_kb"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_agent_key(value: Any) -> str:
    agent_key = str(value or "").strip()
    if not agent_key:
        raise ValueError("AGENT_KEY_REQUIRED")
    return agent_key


def standard_agent_profile(data: dict[str, Any]) -> AgentMemoryProfile:
    """Return the standardized MEM onboarding profile for an Agent.

    Rules:
    - private memory namespace defaults to the stable agent_key;
    - shared KBs are explicit, or fall back to the MEM default shared KB;
    - SQL-HTTP events are not part of generic Agent onboarding;
    - tiering/lifecycle are automatic MEM-side responsibilities.
    """

    agent_key = normalize_agent_key(data.get("agent_key") or data.get("identity"))
    private_project = str(data.get("private_memory_project") or data.get("project_id") or agent_key).strip()
    # Auto-resolve via Private Memory Manager registry for registered agents.
    if not data.get("private_memory_project") and not data.get("project_id"):
        try:
            from .manager_private import get_agent as _get_registered_agent
            profile = _get_registered_agent(private_project)
            if profile:
                private_project = profile["project_id"]
        except Exception:
            pass
    shared = data.get("shared_knowledge_projects") or data.get("shared_projects") or DEFAULT_SHARED_PROJECTS
    if isinstance(shared, str):
        shared_projects = [shared]
    else:
        shared_projects = [str(item).strip() for item in shared if str(item).strip()]
    return AgentMemoryProfile(
        agent_key=agent_key,
        private_memory_project=private_project,
        shared_knowledge_projects=shared_projects,
    )


def apply_standard_push_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Agent push payload so tiering always runs automatically."""

    profile = standard_agent_profile(data)
    scope = data.get("scope") or "role_private"
    metadata = dict(data.get("metadata") or {})
    metadata.setdefault("agent_key", profile.agent_key)
    metadata.setdefault("private_memory_project", profile.private_memory_project)
    metadata.setdefault("shared_knowledge_projects", profile.shared_knowledge_projects)
    metadata.setdefault("standard_agent_memory_profile", profile.to_dict())
    metadata.setdefault("tiering_policy", "mem_auto_tiering_v1")
    return {
        **data,
        "agent_key": profile.agent_key,
        "role": data.get("role") or profile.agent_key,
        "owner_role": data.get("owner_role") or profile.agent_key,
        "actor_role": data.get("actor_role") or profile.agent_key,
        "source_role": data.get("source_role") or profile.agent_key,
        "project_id": data.get("project_id") or profile.private_memory_project,
        "scope": scope,
        "visibility": data.get("visibility") or ("private" if scope == "role_private" else "project_roles"),
        "source": data.get("source") or "agent_push",
        "metadata": metadata,
    }


def apply_standard_pull_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Agent pull payload for private + shared KB recall."""

    profile = standard_agent_profile(data)
    return {
        **data,
        "agent_key": profile.agent_key,
        "private_memory_project": profile.private_memory_project,
        "shared_knowledge_projects": profile.shared_knowledge_projects,
    }
