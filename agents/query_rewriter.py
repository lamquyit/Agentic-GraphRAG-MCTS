"""
CoT Query Rewriter — rewrites queries when the Relevance Gate fails.

Uses Chain-of-Thought reasoning to decompose the user's intent,
identify what went wrong, and generate improved search queries.
"""

from __future__ import annotations

import logging

from llm_client import call_llm

logger = logging.getLogger(__name__)

REWRITER_SYSTEM_PROMPT = """\
You are a Query Rewriting Specialist. When a RAG system fails to produce a relevant answer,
you analyze what went wrong and rewrite the query to improve retrieval.

Your approach:
1. Decompose the original question into its core sub-intents.
2. Analyze why the previous answer missed the mark (using the feedback).
3. Generate an improved query that:
   - Uses more specific terminology
   - Explicitly mentions key concepts the retrieval should target
   - May split complex questions into targeted sub-queries combined into one
   - Adds contextual keywords to improve vector search matching

Return ONLY the rewritten query text. No explanations, no JSON, just the improved query."""


async def rewrite_query(
    original_question: str,
    failed_answer: str,
    relevance_feedback: str,
    missing_aspects: list[str] | None = None,
) -> str:
    """
    CoT Query Rewriting: decompose intent and generate improved query.

    Returns:
        str — the rewritten query
    """
    missing_str = ""
    if missing_aspects:
        missing_str = "\n".join(f"  - {a}" for a in missing_aspects)
        missing_str = f"\n\n## Missing Aspects\n{missing_str}"

    prompt = (
        f"## Original Question\n{original_question}\n\n"
        f"## Previous Answer (not relevant enough)\n{failed_answer}\n\n"
        f"## Relevance Feedback\n{relevance_feedback}"
        f"{missing_str}\n\n"
        f"Rewrite the query to better capture the user's intent and improve retrieval."
    )

    rewritten = await call_llm(
        prompt,
        system_prompt=REWRITER_SYSTEM_PROMPT,
        temperature=0.3,
    )

    rewritten = rewritten.strip().strip('"').strip("'")

    logger.info(
        "Query rewritten: '%s' → '%s'",
        original_question[:60],
        rewritten[:80],
    )
    return rewritten
