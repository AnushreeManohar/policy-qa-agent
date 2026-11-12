"""Chat-model factory. Returns None in extractive mode (no model is called at all)."""

from __future__ import annotations

import json
import re
from typing import Any

from policyqa.config import Settings


class LLMError(RuntimeError):
    """Model provider failed (timeout, auth, rate limit); surfaced as 502 without provider details."""


def build_chat_model(settings: Settings) -> Any | None:
    if settings.llm_provider == "extractive":
        return None
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if settings.anthropic_api_key is None:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        return ChatAnthropic(model=settings.llm_model, temperature=settings.llm_temperature,
                             api_key=settings.anthropic_api_key.get_secret_value(),
                             timeout=settings.llm_timeout_s, max_retries=2, max_tokens=800)
    from langchain_openai import ChatOpenAI

    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if key is None and settings.llm_base_url is None:
        raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY (or LLM_BASE_URL for a local server)")
    return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature, api_key=key or "local",
                      base_url=settings.llm_base_url, timeout=settings.llm_timeout_s, max_retries=2)


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_json_object(text: str) -> dict | None:
    """Tolerant JSON extraction (models sometimes wrap JSON in prose or code fences)."""
    text = text.strip()
    for candidate in (text, *(m.group(0) for m in [_JSON_OBJ.search(text)] if m)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):  # Anthropic content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)
