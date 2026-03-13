"""
LangGraph state schema for the Agentic RAG pipeline.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgenticRAGState(TypedDict, total=False):
    """
    Typed state flowing through the LangGraph StateGraph.

    All fields are optional (total=False) so nodes only need to set
    the fields they produce.
    """

    # ── Input ────────────────────────────────────────────────
    question: str                       # Original user question
    mode: str                           # RAG query mode (hybrid, local, global, ...)

    # ── Planning ─────────────────────────────────────────────
    current_query: str                  # Current query (may be rewritten)
    located_sections: list[dict[str, Any]]  # Sections selected by Planner
    planner_reasoning: str              # Planner's CoT reasoning

    # ── Execution ────────────────────────────────────────────
    retrieved_context: str              # Context chunks from RAG-Anything
    draft_answer: str                   # Current draft answer

    # ── Verification ─────────────────────────────────────────
    groundedness_result: dict[str, Any]  # Groundedness Gate output
    relevance_result: dict[str, Any]     # Relevance Gate output

    # ── Control flow ─────────────────────────────────────────
    final_answer: str                   # Final verified answer
    retry_count: int                    # Current retry iteration
    max_retries: int                    # Max allowed retries
    generation_temperature: float       # LLM temperature for current attempt
    status: str                         # Pipeline status (running / completed / max_retries_exceeded)
