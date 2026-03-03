"""
LangGraph StateGraph — Agentic RAG Pipeline.

Wires together: Planner → Executor → Groundedness Gate → Relevance Gate
with conditional retry loops (regenerate / rewrite query).

Flow:
    START → plan → execute → check_groundedness ─┬─ PASS → check_relevance ─┬─ PASS → END
                                                  │                          │
                                                  └─ FAIL → regenerate ──┘   └─ FAIL → rewrite_query → plan
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from graph.state import AgenticRAGState
from agents.planner import run_planner
from agents.executor import run_executor
from agents.grader_groundedness import check_groundedness
from agents.grader_relevance import check_relevance
from agents.query_rewriter import rewrite_query
from config import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# NODE FUNCTIONS — each takes state and returns partial state update
# ══════════════════════════════════════════════════════════════

async def node_plan(state: AgenticRAGState) -> dict[str, Any]:
    """Locate phase: scan graph structure to decide which sections to read."""
    query = state.get("current_query") or state["question"]

    result = await run_planner(query)

    return {
        "located_sections": result.get("selected_sections", []),
        "current_query": result.get("enhanced_query", query),
        "planner_reasoning": result.get("reasoning", ""),
    }


async def node_execute(state: AgenticRAGState) -> dict[str, Any]:
    """Read phase: deep-read through RAG-Anything with focused context."""
    result = await run_executor(
        question=state["question"],
        enhanced_query=state.get("current_query", state["question"]),
        selected_sections=state.get("located_sections", []),
        mode=state.get("mode", "hybrid"),
        user_roles=state.get("user_roles"),
        company_id=state.get("company_id"),
    )

    return {
        "draft_answer": result["draft_answer"],
        "retrieved_context": result["retrieved_context"],
    }


async def node_check_groundedness(state: AgenticRAGState) -> dict[str, Any]:
    """Groundedness Gate: verify answer is grounded in context."""
    result = await check_groundedness(
        draft_answer=state.get("draft_answer", ""),
        retrieved_context=state.get("retrieved_context", ""),
    )

    return {"groundedness_result": result}


async def node_check_relevance(state: AgenticRAGState) -> dict[str, Any]:
    """Relevance Gate: verify answer addresses the question."""
    result = await check_relevance(
        draft_answer=state.get("draft_answer", ""),
        original_question=state["question"],
    )

    return {"relevance_result": result}


async def node_regenerate(state: AgenticRAGState) -> dict[str, Any]:
    """Re-generate answer with temperature=0 after groundedness failure."""
    retry = state.get("retry_count", 0) + 1
    logger.info("Regenerating (attempt %d/%d) with temp=0", retry, state.get("max_retries", settings.MAX_RETRIES))

    # Re-execute with lower temperature — the executor re-queries RAG
    result = await run_executor(
        question=state["question"],
        enhanced_query=state.get("current_query", state["question"]),
        selected_sections=state.get("located_sections", []),
        mode=state.get("mode", "hybrid"),
        user_roles=state.get("user_roles"),
        company_id=state.get("company_id"),
    )

    return {
        "draft_answer": result["draft_answer"],
        "retrieved_context": result["retrieved_context"],
        "retry_count": retry,
        "generation_temperature": settings.REGEN_TEMPERATURE,
    }


async def node_rewrite_query(state: AgenticRAGState) -> dict[str, Any]:
    """CoT Query Rewriting after relevance failure."""
    retry = state.get("retry_count", 0) + 1
    relevance = state.get("relevance_result", {})

    new_query = await rewrite_query(
        original_question=state["question"],
        failed_answer=state.get("draft_answer", ""),
        relevance_feedback=relevance.get("reasoning", ""),
        missing_aspects=relevance.get("missing_aspects"),
    )

    logger.info("Query rewritten (attempt %d): %s", retry, new_query[:80])

    return {
        "current_query": new_query,
        "retry_count": retry,
    }


async def node_finalize(state: AgenticRAGState) -> dict[str, Any]:
    """Set the final answer and mark completed."""
    return {
        "final_answer": state.get("draft_answer", ""),
        "status": "completed",
    }


async def node_max_retries_exceeded(state: AgenticRAGState) -> dict[str, Any]:
    """Return best-effort answer when max retries are exceeded."""
    logger.warning("Max retries exceeded — returning best-effort answer")
    return {
        "final_answer": state.get("draft_answer", "Could not generate a verified answer."),
        "status": "max_retries_exceeded",
    }


# ══════════════════════════════════════════════════════════════
# CONDITIONAL EDGES — routing logic
# ══════════════════════════════════════════════════════════════

def route_after_groundedness(state: AgenticRAGState) -> str:
    """After groundedness check: pass → check_relevance, fail → regenerate or give up."""
    gr = state.get("groundedness_result", {})
    if gr.get("pass", False):
        return "check_relevance"

    # Check retry budget
    if state.get("retry_count", 0) >= state.get("max_retries", settings.MAX_RETRIES):
        return "max_retries_exceeded"

    return "regenerate"


def route_after_relevance(state: AgenticRAGState) -> str:
    """After relevance check: pass → finalize, fail → rewrite or give up."""
    rel = state.get("relevance_result", {})
    if rel.get("pass", False):
        return "finalize"

    # Check retry budget
    if state.get("retry_count", 0) >= state.get("max_retries", settings.MAX_RETRIES):
        return "max_retries_exceeded"

    return "rewrite_query"


# ══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def build_agentic_rag_graph() -> StateGraph:
    """Build and compile the Agentic RAG StateGraph."""

    graph = StateGraph(AgenticRAGState)

    # ── Add nodes ────────────────────────────────────────────
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("check_groundedness", node_check_groundedness)
    graph.add_node("check_relevance", node_check_relevance)
    graph.add_node("regenerate", node_regenerate)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("finalize", node_finalize)
    graph.add_node("max_retries_exceeded", node_max_retries_exceeded)

    # ── Set entry point ──────────────────────────────────────
    graph.set_entry_point("plan")

    # ── Linear edges ─────────────────────────────────────────
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "check_groundedness")

    # ── Conditional: after groundedness ──────────────────────
    graph.add_conditional_edges(
        "check_groundedness",
        route_after_groundedness,
        {
            "check_relevance": "check_relevance",
            "regenerate": "regenerate",
            "max_retries_exceeded": "max_retries_exceeded",
        },
    )

    # After regenerate → go back to groundedness check
    graph.add_edge("regenerate", "check_groundedness")

    # ── Conditional: after relevance ─────────────────────────
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
            "max_retries_exceeded": "max_retries_exceeded",
        },
    )

    # After rewrite → loop back to plan
    graph.add_edge("rewrite_query", "plan")

    # ── Terminal nodes ───────────────────────────────────────
    graph.add_edge("finalize", END)
    graph.add_edge("max_retries_exceeded", END)

    return graph.compile()


# Default recursion limit (max number of node transitions in one invocation)
DEFAULT_RECURSION_LIMIT = 25


# ── Convenience runner ──────────────────────────────────────

async def run_agentic_rag(
    question: str,
    *,
    mode: str = "hybrid",
    user_roles: list[str] | None = None,
    company_id: str | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """
    Run the full Agentic RAG pipeline.

    Returns the final state dict with:
        final_answer, status, retry_count, groundedness_result, relevance_result, etc.
    """
    compiled = build_agentic_rag_graph()
    retries = max_retries if max_retries is not None else settings.MAX_RETRIES

    initial_state: AgenticRAGState = {
        "question": question,
        "current_query": question,
        "mode": mode,
        "user_roles": user_roles,
        "company_id": company_id,
        "retry_count": 0,
        "max_retries": retries,
        "generation_temperature": 0.7,
        "status": "running",
        "located_sections": [],
        "retrieved_context": "",
        "draft_answer": "",
        "planner_reasoning": "",
        "groundedness_result": {},
        "relevance_result": {},
        "final_answer": "",
    }

    # Execute graph with recursion limit to prevent infinite loops
    final_state = await compiled.ainvoke(
        initial_state,
        config={"recursion_limit": DEFAULT_RECURSION_LIMIT},
    )

    return final_state
