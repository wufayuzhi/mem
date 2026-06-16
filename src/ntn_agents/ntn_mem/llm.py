"""LLM inference provider for NTN MEM.

Wraps SiliconFlow's OpenAI-compatible /chat/completions API so MEM can:
  1. Summarise search results for agent recoll (recollect)
  2. Assess importance / generate summaries during memory writes
  3. Run nightly experience extraction

Shares the same API key + base_url as the embedding provider (embedding.py),
so no extra config is needed if SiliconFlow is already configured.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.request import Request, urlopen


class LLMProvider:
    """OpenAI-compatible LLM provider backed by SiliconFlow.

    Reads credentials from the same env vars as the embedding provider so that
    one SiliconFlow key serves both embedding and chat.  Falls back to reading
    /etc/ntn-agents/secrets.env for systems (such as MEM) where the secrets
    file is loaded via systemd EnvironmentFile but not directly visible to new
    Python subprocesses.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("NTN_MEM_EMBEDDING_BASE_URL")
            or self._fallback_secret("NTN_MEM_EMBEDDING_BASE_URL")
            or "https://api.siliconflow.cn/v1"
        )
        self.model = (
            model
            or os.environ.get("NTN_MEM_LLM_MODEL")
            or "THUDM/GLM-4-9B-0414"
        )
        self.api_key = (
            api_key
            or os.environ.get("NTN_MEM_EMBEDDING_API_KEY")
            or self._fallback_secret("NTN_MEM_EMBEDDING_API_KEY")
            or ""
        )
        self.timeout = timeout
        self.max_tokens = max_tokens

    _SECRETS_FILE = os.environ.get(
        "NTN_MEM_SECRETS_FILE", "/etc/ntn-agents/secrets.env"
    )

    @staticmethod
    def _fallback_secret(var_name: str) -> str | None:
        """Read a variable from the secrets file on disk."""
        try:
            with open(LLMProvider._SECRETS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{var_name}="):
                        val = line.split("=", 1)[1].strip("\"' \t\n\r")
                        return val if val else None
        except (OSError, FileNotFoundError):
            pass
        return None

    def _api_key_available(self) -> bool:
        return bool(self.api_key) and "***" not in self.api_key

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Call the LLM and return the full response dict.

        Args:
            messages: Standard chat messages [{"role": "user", "content": "..."}]
            system: Optional system prompt (prepended as a system message).
            max_tokens: Override the default max_tokens.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            A dict with "content", "reasoning_content", "usage", and "raw" fields.
            "raw" holds the full parsed API response for debugging.

        Raises:
            RuntimeError: If the API key is missing or the call fails.
        """
        if not self._api_key_available():
            raise RuntimeError(
                "LLMProvider: NTN_MEM_EMBEDDING_API_KEY not found. "
                "Check secrets file or set the env var."
            )

        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"LLMProvider: SiliconFlow API error: {exc}") from exc

        choices = raw.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"LLMProvider: empty choices: {json.dumps(raw, ensure_ascii=False)[:500]}"
            )

        message = choices[0].get("message", {})
        return {
            "content": (message.get("content") or "").strip(),
            "reasoning_content": (message.get("reasoning_content") or "").strip(),
            "usage": raw.get("usage", {}),
            "raw": raw,
        }

    def chat_str(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Convenience wrapper — returns just the text content."""
        result = self.chat(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["content"]

    def extract_json(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Call LLM and parse the response as JSON.

        Handles possible ```json ... ``` wrapping around the output.
        """
        text = self.chat_str(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # DeepSeek-R1 sometimes wraps reasoning in ... then outputs JSON
        # Strip everything before the first { if there's non-JSON prefix
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
        cleaned = text.strip()

        # Strip markdown JSON fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try full-string parse, then regex fallback
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(
                f"LLMProvider: failed to parse JSON from response:\n{text[:500]}"
            )
