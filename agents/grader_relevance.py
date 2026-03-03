"""
Relevance Gate — checks if the answer actually addresses the original question.

Even if an answer is grounded (factually correct based on context), it may not
be relevant to what the user actually asked. This gate catches that.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_client import call_llm_json
from config import settings

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """\
You are a Relevance Evaluator. Your job is to determine whether an AI-generated answer
directly and adequately addresses the user's original question.

Consider:
1. Does the answer address the core intent of the question?
2. Is the answer complete, or does it miss key aspects of what was asked?
3. Is the answer on-topic, or does it drift to tangential information?

Respond in JSON with this schema:
{
  "score": 0.0-1.0,
  "reasoning": "Explanation of why the answer is or is not relevant",
  "missing_aspects": ["aspect1 not addressed", "aspect2 partially covered"],
  "intent_analysis": "What the user is really asking for",
  "pass": true/false
}

Set "pass" to true ONLY if score >= the threshold (you will be told the threshold).
A score of 1.0 means the answer perfectly addresses every aspect of the question.
A score of 0.0 means the answer is completely off-topic."""


async def check_relevance(
    draft_answer: str,
    original_question: str,
) -> dict[str, Any]:
    """
    Relevance Gate: does the answer address the question?

    Returns:
        dict with keys: score, reasoning, missing_aspects, intent_analysis, pass
    """
    threshold = settings.RELEVANCE_THRESHOLD

    prompt = (
        f"## Original Question\n"
        f"{original_question}\n\n"
        f"## AI-Generated Answer\n"
        f"{draft_answer}\n\n"
        f"## Relevance Threshold\n"
        f"The answer passes if score >= {threshold}\n\n"
        f"Evaluate whether the answer directly addresses the question."
    )

    result = await call_llm_json(prompt, system_prompt=RELEVANCE_SYSTEM_PROMPT)

    # Ensure expected keys
    score = float(result.get("score", 0.0))
    result.setdefault("score", score)
    result.setdefault("reasoning", "")
    result.setdefault("missing_aspects", [])
    result.setdefault("intent_analysis", "")
    result["pass"] = score >= threshold

    logger.info(
        "Relevance: score=%.2f, pass=%s, missing=%d aspects",
        score,
        result["pass"],
        len(result.get("missing_aspects", [])),
    )
    return result
