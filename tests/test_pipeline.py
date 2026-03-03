"""
Integration test for the Agentic RAG pipeline.
Tests the full LangGraph flow with mocked RAG bridge and LLM calls.

Strategy: We mock `rag_bridge` and `rag_service` at `sys.modules` level
BEFORE importing any agents, to avoid triggering the heavy
lightrag import chain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Ensure project root on path ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Mock heavy modules BEFORE any agent imports ─────────────
_mock_rag_bridge = ModuleType("rag_bridge")
_mock_rag_bridge.get_entity_summary = MagicMock(return_value=[])
_mock_rag_bridge.query_rag = AsyncMock(return_value="mock answer")
_mock_rag_bridge.query_rag_sync = MagicMock(return_value="mock answer")
_mock_rag_bridge.get_chunks_for_entities = MagicMock(return_value=[])
_mock_rag_bridge.load_graph_structure = MagicMock(return_value={"entities": [], "chunks": [], "graph": None})
_mock_rag_bridge.get_rag_service = MagicMock()

_mock_rag_service = ModuleType("rag_service")
_mock_rag_service.RAGService = MagicMock()

sys.modules["rag_bridge"] = _mock_rag_bridge
sys.modules["rag_service"] = _mock_rag_service

# NOW safe to import agents and graph
import agents.planner  # noqa: E402
import agents.executor  # noqa: E402
import agents.grader_groundedness  # noqa: E402
import agents.grader_relevance  # noqa: E402
import agents.query_rewriter  # noqa: E402
from graph.agentic_rag_graph import (  # noqa: E402
    run_agentic_rag,
    node_plan, node_execute, node_check_groundedness,
    node_check_relevance, node_regenerate, node_rewrite_query,
    route_after_groundedness, route_after_relevance,
)


# ══════════════════════════════════════════════════════════════
# TEST 1: Happy path — full pipeline, everything passes first try
# ══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_full_pipeline_happy_path():
    """question → plan → execute → groundedness PASS → relevance PASS → final answer."""

    planner_response = {
        "needs_planning": True,
        "selected_sections": [
            {"doc_id": "doc-1", "file_path": "test.txt", "relevant_entities": ["BETA INC"], "reason": "Relevant"}
        ],
        "enhanced_query": "What are BETA INC employee benefits?",
        "reasoning": "Question targets BETA INC benefits.",
    }

    groundedness_response = {
        "score": 0.95,
        "hallucinated_claims": [],
        "grounded_claims": ["benefits claim supported"],
        "reasoning": "All good.",
        "pass": True,
    }

    relevance_response = {
        "score": 0.9,
        "reasoning": "Answer addresses benefits question.",
        "missing_aspects": [],
        "intent_analysis": "User wants benefits info.",
        "pass": True,
    }

    with patch("agents.planner.call_llm_json", new_callable=AsyncMock) as mock_plan, \
         patch("agents.executor.query_rag", new_callable=AsyncMock) as mock_rag, \
         patch("agents.executor.get_chunks_for_entities") as mock_chunks, \
         patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_gr, \
         patch("agents.grader_relevance.call_llm_json", new_callable=AsyncMock) as mock_rel, \
         patch("agents.planner.get_entity_summary") as mock_entities:

        mock_plan.return_value = planner_response
        mock_entities.return_value = [
            {"doc_id": "doc-1", "file_path": "test.txt", "entity_names": ["BETA INC"], "preview": "BETA handbook"}
        ]
        mock_chunks.return_value = [
            {"content": "BETA provides PVI health insurance.", "file_path": "test.txt"}
        ]
        mock_rag.return_value = "BETA provides PVI health insurance and training support up to 10M VND/year."
        mock_gr.return_value = groundedness_response
        mock_rel.return_value = relevance_response

        result = await run_agentic_rag("What benefits does BETA provide?")

        assert result["status"] == "completed"
        assert "PVI" in result["final_answer"]
        assert result["retry_count"] == 0


# ══════════════════════════════════════════════════════════════
# TEST 2: Groundedness retry — test individual node flow
#   Tests the regenerate loop by calling nodes directly
# ══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_groundedness_retry_logic():
    """
    Unit-level test of the retry flow:
    execute → groundedness FAIL → regenerate → groundedness PASS.
    Tests nodes and routing directly to avoid LangGraph event-loop issues.
    """
    # State after execute
    state_after_execute = {
        "question": "BETA benefits?",
        "current_query": "BETA benefits",
        "mode": "hybrid",
        "user_roles": None,
        "company_id": None,
        "draft_answer": "BETA gives 30 vacation days.",
        "retrieved_context": "BETA provides PVI health insurance.",
        "retry_count": 0,
        "max_retries": 3,
        "located_sections": [{"doc_id": "d1", "relevant_entities": ["BETA"]}],
    }

    # Step 1: Groundedness check FAILS (hallucination detected)
    groundedness_fail = {
        "score": 0.3,
        "hallucinated_claims": ["30 vacation days not in context"],
        "grounded_claims": [],
        "reasoning": "Hallucination detected.",
        "pass": False,
    }

    with patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_gr:
        mock_gr.return_value = groundedness_fail
        update = await node_check_groundedness(state_after_execute)

    state_after_gr = {**state_after_execute, **update}

    # Step 2: Routing should go to "regenerate"
    route = route_after_groundedness(state_after_gr)
    assert route == "regenerate", f"Expected 'regenerate', got '{route}'"

    # Step 3: Regenerate node
    with patch("agents.executor.query_rag", new_callable=AsyncMock) as mock_rag, \
         patch("agents.executor.get_chunks_for_entities") as mock_chunks:
        mock_rag.return_value = "BETA provides PVI health insurance."
        mock_chunks.return_value = [{"content": "PVI insurance", "file_path": "t.txt"}]
        regen_update = await node_regenerate(state_after_gr)

    state_after_regen = {**state_after_gr, **regen_update}
    assert state_after_regen["retry_count"] == 1

    # Step 4: Groundedness check PASSES now
    groundedness_pass = {
        "score": 0.9,
        "hallucinated_claims": [],
        "grounded_claims": ["PVI insurance confirmed"],
        "reasoning": "Grounded.",
        "pass": True,
    }

    with patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_gr:
        mock_gr.return_value = groundedness_pass
        update2 = await node_check_groundedness(state_after_regen)

    state_after_gr2 = {**state_after_regen, **update2}

    # Step 5: Routing should now go to "check_relevance"
    route2 = route_after_groundedness(state_after_gr2)
    assert route2 == "check_relevance", f"Expected 'check_relevance', got '{route2}'"


# ══════════════════════════════════════════════════════════════
# TEST 3: Relevance fails → query rewrite → re-plan → passes
# ══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_pipeline_with_relevance_rewrite():
    """relevance fails → query rewrite → re-plan → passes."""

    planner_response = {
        "needs_planning": False,
        "selected_sections": [],
        "enhanced_query": "leave policy",
        "reasoning": "Simple.",
    }

    groundedness_pass = {
        "score": 0.95, "hallucinated_claims": [], "grounded_claims": ["ok"],
        "reasoning": "Grounded.", "pass": True,
    }

    relevance_fail = {
        "score": 0.2, "reasoning": "Off-topic.",
        "missing_aspects": ["leave days"], "intent_analysis": "", "pass": False,
    }
    relevance_pass = {
        "score": 0.9, "reasoning": "Now relevant.",
        "missing_aspects": [], "intent_analysis": "", "pass": True,
    }

    call_count = {"relevance": 0}

    async def relevance_side_effect(*args, **kwargs):
        call_count["relevance"] += 1
        if call_count["relevance"] == 1:
            return relevance_fail
        return relevance_pass

    with patch("agents.planner.call_llm_json", new_callable=AsyncMock) as mock_plan, \
         patch("agents.executor.query_rag", new_callable=AsyncMock) as mock_rag, \
         patch("agents.executor.get_chunks_for_entities") as mock_chunks, \
         patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_gr, \
         patch("agents.grader_relevance.call_llm_json", new_callable=AsyncMock) as mock_rel, \
         patch("agents.query_rewriter.call_llm", new_callable=AsyncMock) as mock_rewrite, \
         patch("agents.planner.get_entity_summary") as mock_entities:

        mock_plan.return_value = planner_response
        mock_entities.return_value = []
        mock_chunks.return_value = []
        mock_rag.return_value = "Improved answer about leave days."
        mock_gr.return_value = groundedness_pass
        mock_rel.side_effect = relevance_side_effect
        mock_rewrite.return_value = "How many annual leave days for ACME employees?"

        result = await run_agentic_rag("Tell me about leave")

        assert result["status"] == "completed"
        assert result["retry_count"] >= 1


# ══════════════════════════════════════════════════════════════
# TEST 4: Max retries exceeded
# ══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_max_retries_routing():
    """When retry_count >= max_retries, routing should go to max_retries_exceeded."""

    state = {
        "retry_count": 3,
        "max_retries": 3,
        "groundedness_result": {"pass": False, "score": 0.2},
        "relevance_result": {"pass": False, "score": 0.2},
    }

    assert route_after_groundedness(state) == "max_retries_exceeded"
    assert route_after_relevance(state) == "max_retries_exceeded"


@pytest.mark.asyncio
async def test_routing_logic():
    """Test all routing branches."""

    # Groundedness pass → check_relevance
    assert route_after_groundedness({"groundedness_result": {"pass": True}}) == "check_relevance"

    # Groundedness fail, retries left → regenerate
    assert route_after_groundedness({"groundedness_result": {"pass": False}, "retry_count": 0, "max_retries": 3}) == "regenerate"

    # Relevance pass → finalize
    assert route_after_relevance({"relevance_result": {"pass": True}}) == "finalize"

    # Relevance fail, retries left → rewrite_query
    assert route_after_relevance({"relevance_result": {"pass": False}, "retry_count": 0, "max_retries": 3}) == "rewrite_query"
