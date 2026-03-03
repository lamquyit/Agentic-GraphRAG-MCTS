"""
Groundedness Gate — verifies the draft answer is grounded in retrieved context.

Cross-references the answer against the retrieved graph nodes / chunks.
If hallucinations are detected, signals for regeneration with temperature=0.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_client import call_llm_json
from config import settings

logger = logging.getLogger(__name__)

GROUNDEDNESS_SYSTEM_PROMPT = """\
You are a Groundedness Verifier. Your job is to check whether an AI-generated answer
is fully grounded in (supported by) the provided source context.

You must:
1. Check every factual claim in the answer against the source context.
2. Identify any claims that are NOT present in or inferable from the context (hallucinations).
3. Score the overall groundedness from 0.0 (completely fabricated) to 1.0 (fully grounded).

Respond in JSON with this schema:
{
  "score": 0.0-1.0,
  "hallucinated_claims": ["claim1 not found in context", "claim2 fabricated"],
  "grounded_claims": ["claim1 supported by context", "claim2 found in source"],
  "reasoning": "Brief explanation of your assessment",
  "pass": true/false
}

Set "pass" to true ONLY if score >= the threshold (you will be told the threshold).
Be strict: if the answer adds specific numbers, dates, or facts not in the context, flag them."""


async def check_groundedness(
    draft_answer: str,
    retrieved_context: str,
) -> dict[str, Any]:
    """
    Groundedness Gate: cross-reference answer against retrieved context.

    Returns:
        dict with keys: score, hallucinated_claims, grounded_claims, reasoning, pass
    """
    if not retrieved_context.strip():
        logger.warning("No retrieved context — skipping groundedness check, auto-pass")
        return {
            "score": 1.0,
            "hallucinated_claims": [],
            "grounded_claims": [],
            "reasoning": "No context to verify against (direct RAG answer).",
            "pass": True,
        }

    threshold = settings.GROUNDEDNESS_THRESHOLD

    prompt = (
        f"## Source Context (ground truth)\n"
        f"{retrieved_context}\n\n"
        f"## AI-Generated Answer to verify\n"
        f"{draft_answer}\n\n"
        f"## Groundedness Threshold\n"
        f"The answer passes if score >= {threshold}\n\n"
        f"Check every claim in the answer against the source context."
    )

    result = await call_llm_json(prompt, system_prompt=GROUNDEDNESS_SYSTEM_PROMPT)

    # Ensure expected keys with defaults
    score = float(result.get("score", 0.0))
    result.setdefault("score", score)
    result.setdefault("hallucinated_claims", [])
    result.setdefault("grounded_claims", [])
    result.setdefault("reasoning", "")
    result["pass"] = score >= threshold

    logger.info(
        "Groundedness: score=%.2f, pass=%s, hallucinations=%d",
        score,
        result["pass"],
        len(result.get("hallucinated_claims", [])),
    )
    return result
