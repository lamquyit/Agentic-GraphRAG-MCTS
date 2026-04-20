"""
Executor Agent — Deep-read phase.

Takes located sections from the Planner and executes a focused retrieval
through RAG-Anything, constructing context-enriched queries.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_bridge import aquery_rag_with_context, get_chunks_for_entities

logger = logging.getLogger(__name__)


async def run_executor(
    question: str,
    enhanced_query: str,
    selected_sections: list[dict[str, Any]],
    *,
    mode: str = "hybrid",
    storage_name: str = "rag_storage",
) -> dict[str, Any]:
    """
    Read phase: deep-read data from located sections via RAG-Anything.

    Returns:
        dict with keys: draft_answer, retrieved_context
    """
    # Build focused context from selected sections
    extra_context_parts: list[str] = []

    if selected_sections:
        # Gather entity names from all selected sections
        all_entities: list[str] = []
        for sec in selected_sections:
            all_entities.extend(sec.get("relevant_entities", []))

        if all_entities:
            chunks = get_chunks_for_entities(all_entities, storage_name=storage_name)
            for chunk in chunks:
                extra_context_parts.append(
                    f"[From {chunk['file_path']}]: {chunk['content'][:500]}"
                )
            logger.info("Executor: gathered %d chunks for %d entities", len(chunks), len(all_entities))

    # Use enhanced query from Planner, enriched with section context
    query_to_use = enhanced_query or question

    # If we have extra context from sections, prepend it
    if extra_context_parts:
        section_context = "\n\n".join(extra_context_parts)
        query_to_use = (
            f"Relevant document context:\n{section_context}\n\n"
            f"Question: {query_to_use}"
        )

    # Execute through RAG-Anything
    try:
        res = await aquery_rag_with_context(
            query_to_use,
            mode=mode,
            storage_name=storage_name,
        )
        draft_answer = res.get("answer", "")
        extra_context_parts.extend(res.get("contexts", []))
    except Exception as e:
        logger.error("RAG query failed: %s", e)
        draft_answer = f"Error during retrieval: {e}"

    # Collect retrieved context (from extra chunks + query result)
    retrieved_context = "\n\n".join(extra_context_parts) if extra_context_parts else ""

    logger.info("Executor: generated draft answer (%d chars)", len(draft_answer))

    return {
        "draft_answer": draft_answer,
        "retrieved_context": retrieved_context,
    }
