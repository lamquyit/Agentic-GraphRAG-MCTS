"""
Async LLM client — thin wrapper around OpenAI-compatible API.
Used by every agent in the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
    return _client


async def call_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: str | None = None,
) -> str:
    """Plain text completion."""
    client = _get_client()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = await client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def call_llm_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    model: str | None = None,
) -> dict[str, Any]:
    """
    LLM call that expects a JSON object back.
    Uses response_format where supported, falls back to parsing.
    """
    client = _get_client()
    messages: list[dict[str, str]] = []
    sys_msg = (system_prompt or "") + "\nYou MUST respond with valid JSON only. No markdown fences."
    messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = await client.chat.completions.create(
            model=model or settings.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        # Fallback: some providers don't support response_format
        logger.warning("JSON mode not supported by provider, falling back to plain completion")
        resp = await client.chat.completions.create(
            model=model or settings.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content or "{}"

    # Parse — strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response: %s", raw[:200])
        return {"error": "json_parse_failed", "raw": raw}
