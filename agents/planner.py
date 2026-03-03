"""
Planner Agent — Locate-then-Read strategy.

Scans the content graph hierarchy (entity names, document sections) to identify
which sections are most relevant to a question BEFORE triggering deep retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_client import call_llm_json
from rag_bridge import get_entity_summary

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a Document Planner Agent. Your job is to analyze a user's question and determine
which document sections are most relevant to answer it.

You will be given a list of documents with their entity names (topics, sections, headings)
and a short preview of their content. You must select which documents/entities should be
read in depth to construct a comprehensive answer.

Respond in JSON with this schema:
{
  "reasoning": "Your chain-of-thought analysis of the question and why you selected these sections",
  "needs_planning": true/false,
  "selected_sections": [
    {
      "doc_id": "doc-xxx",
      "file_path": "filename.txt",
      "relevant_entities": ["Entity1", "Entity2"],
      "reason": "Why this section is relevant"
    }
  ],
  "enhanced_query": "An improved version of the question incorporating section context"
}

If the question is simple and targets a single obvious topic, set needs_planning=false
and return just the enhanced_query without selecting sections."""


async def run_planner(question: str) -> dict[str, Any]:
    """
    Locate phase: scan graph structure to determine which sections to read.

    Returns:
        dict with keys: needs_planning, selected_sections, enhanced_query, reasoning
    """
    # Get entity summaries from the content graph
    summaries = get_entity_summary()

    if not summaries:
        logger.warning("No entity summaries found — skipping planning, using direct query")
        return {
            "needs_planning": False,
            "selected_sections": [],
            "enhanced_query": question,
            "reasoning": "No document structure available for planning.",
        }

    # Build the context for the planner
    sections_text = ""
    for i, s in enumerate(summaries, 1):
        entities_str = ", ".join(s["entity_names"])
        sections_text += (
            f"\n--- Document {i} ---\n"
            f"Doc ID: {s['doc_id']}\n"
            f"File: {s['file_path']}\n"
            f"Topics/Entities: {entities_str}\n"
            f"Preview: {s['preview']}\n"
        )

    prompt = (
        f"## User Question\n{question}\n\n"
        f"## Available Document Sections\n{sections_text}\n\n"
        "Analyze the question and select the most relevant sections to read."
    )

    result = await call_llm_json(prompt, system_prompt=PLANNER_SYSTEM_PROMPT)

    # Ensure expected keys
    result.setdefault("needs_planning", False)
    result.setdefault("selected_sections", [])
    result.setdefault("enhanced_query", question)
    result.setdefault("reasoning", "")

    logger.info(
        "Planner: needs_planning=%s, selected %d sections",
        result["needs_planning"],
        len(result["selected_sections"]),
    )
    return result
