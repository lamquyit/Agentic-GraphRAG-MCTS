"""
Unit tests for Groundedness and Relevance graders.
Uses monkeypatched LLM client to avoid real API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════
# GROUNDEDNESS GATE TESTS
# ══════════════════════════════════════════════════════════════

class TestGroundednessGate:
    """Test the Groundedness Gate with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_grounded_answer_passes(self):
        """An answer fully supported by context should pass."""
        mock_response = {
            "score": 0.95,
            "hallucinated_claims": [],
            "grounded_claims": ["Claim A is in context"],
            "reasoning": "All claims are supported.",
            "pass": True,
        }

        with patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            from agents.grader_groundedness import check_groundedness

            result = await check_groundedness(
                draft_answer="The company provides health insurance.",
                retrieved_context="The company provides PVI health insurance for all employees.",
            )

            assert result["pass"] is True
            assert result["score"] >= 0.7
            assert len(result["hallucinated_claims"]) == 0

    @pytest.mark.asyncio
    async def test_hallucinated_answer_fails(self):
        """An answer with fabricated claims should fail."""
        mock_response = {
            "score": 0.3,
            "hallucinated_claims": ["vacation days claim not in context"],
            "grounded_claims": [],
            "reasoning": "The answer mentions vacation days which are not in context.",
            "pass": False,
        }

        with patch("agents.grader_groundedness.call_llm_json", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            from agents.grader_groundedness import check_groundedness

            result = await check_groundedness(
                draft_answer="Employees get 30 vacation days and free gym membership.",
                retrieved_context="Employees work from 8:30 to 17:30.",
            )

            assert result["pass"] is False
            assert result["score"] < 0.7
            assert len(result["hallucinated_claims"]) > 0

    @pytest.mark.asyncio
    async def test_empty_context_auto_passes(self):
        """With no context, groundedness should auto-pass (direct RAG answer)."""
        from agents.grader_groundedness import check_groundedness

        result = await check_groundedness(
            draft_answer="Some answer",
            retrieved_context="",
        )
        assert result["pass"] is True
        assert result["score"] == 1.0


# ══════════════════════════════════════════════════════════════
# RELEVANCE GATE TESTS
# ══════════════════════════════════════════════════════════════

class TestRelevanceGate:
    """Test the Relevance Gate with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_relevant_answer_passes(self):
        """An answer that addresses the question should pass."""
        mock_response = {
            "score": 0.9,
            "reasoning": "The answer directly addresses the question about leave policy.",
            "missing_aspects": [],
            "intent_analysis": "User wants to know about leave days.",
            "pass": True,
        }

        with patch("agents.grader_relevance.call_llm_json", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            from agents.grader_relevance import check_relevance

            result = await check_relevance(
                draft_answer="Employees get 15 days of annual leave.",
                original_question="How many leave days do ACME employees get?",
            )

            assert result["pass"] is True
            assert result["score"] >= 0.7

    @pytest.mark.asyncio
    async def test_irrelevant_answer_fails(self):
        """An answer that doesn't address the question should fail."""
        mock_response = {
            "score": 0.2,
            "reasoning": "The answer talks about working hours, not leave policy.",
            "missing_aspects": ["leave policy not mentioned"],
            "intent_analysis": "User asked about leave days.",
            "pass": False,
        }

        with patch("agents.grader_relevance.call_llm_json", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            from agents.grader_relevance import check_relevance

            result = await check_relevance(
                draft_answer="Employees work from 8:30 to 17:30 with a 1 hour lunch break.",
                original_question="How many leave days do ACME employees get?",
            )

            assert result["pass"] is False
            assert result["score"] < 0.7
            assert len(result["missing_aspects"]) > 0


# ══════════════════════════════════════════════════════════════
# QUERY REWRITER TESTS
# ══════════════════════════════════════════════════════════════

class TestQueryRewriter:
    """Test the CoT Query Rewriter."""

    @pytest.mark.asyncio
    async def test_rewrite_improves_query(self):
        """Rewriter should produce a different, more specific query."""
        with patch("agents.query_rewriter.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "What is the annual leave policy for employees at ACME Corporation including days allowed and approval process?"
            from agents.query_rewriter import rewrite_query

            result = await rewrite_query(
                original_question="Tell me about leave",
                failed_answer="The company has various policies.",
                relevance_feedback="Too vague, doesn't specify leave days.",
                missing_aspects=["specific number of leave days", "approval process"],
            )

            assert len(result) > len("Tell me about leave")
            assert isinstance(result, str)
