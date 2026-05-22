"""
MCTS Orchestrator — high-level entry point for running MCTS search.

This is the function called from LangGraph's node_mcts_search().
It initializes the tree, runs the engine, and packages the result.
"""

from __future__ import annotations

import logging
from typing import Any

from mcts.mcts_node import MCTSNode, MCTSState, MCTSAction, MCTSActionType, MCTSResult
from mcts.mcts_engine import MCTSEngine

logger = logging.getLogger(__name__)


async def run_mcts_search(
    question: str,
    *,
    max_iterations: int = 8,
    max_depth: int = 4,
    exploration_weight: float = 1.414,
    groundedness_weight: float = 0.6,
    relevance_weight: float = 0.4,
    min_confidence: float = 0.7,
    parallel_expansion: bool = True,
    mode: str = "hybrid",
    storage_name: str = "rag_storage",
) -> MCTSResult:
    """
    Run a complete MCTS search from a user question.

    This is the main entry point called by the LangGraph pipeline.

    Args:
        question: The user's original question.
        max_iterations: Number of MCTS iterations (selection-expansion-simulation-backprop cycles).
        max_depth: Maximum depth of the search tree.
        exploration_weight: C parameter in UCT formula (higher = more exploration).
        groundedness_weight: Weight for groundedness in the reward function.
        relevance_weight: Weight for relevance in the reward function.
        min_confidence: Q-value threshold for early termination.
        parallel_expansion: Whether to expand children in parallel.
        mode: Initial RAG query mode (hybrid/local/global).
        storage_name: Name of the RAG storage directory.

    Returns:
        MCTSResult with the best answer, confidence, path, and tree stats.
    """
    logger.info(
        "Starting MCTS search: question='%s', iterations=%d, depth=%d, C=%.3f",
        question[:80], max_iterations, max_depth, exploration_weight,
    )

    # 1. Initialize root node
    root_state = MCTSState(
        query=question,
        original_question=question,
        mode=mode,
        storage_name=storage_name,
        depth=0,
    )
    root = MCTSNode(
        state=root_state,
        action=MCTSAction(type=MCTSActionType.ROOT, description="Search root"),
    )

    # 2. Create and run the engine
    engine = MCTSEngine(
        max_iterations=max_iterations,
        max_depth=max_depth,
        exploration_weight=exploration_weight,
        groundedness_weight=groundedness_weight,
        relevance_weight=relevance_weight,
        min_confidence=min_confidence,
        parallel_expansion=parallel_expansion,
    )

    best_node = await engine.search(root)

    # 3. Package the result
    path = MCTSEngine.trace_path(best_node)
    stats = engine.collect_stats(root)

    # Get final scores for the best answer
    groundedness_score = 0.0
    relevance_score = 0.0
    if best_node.state.draft_answer:
        groundedness_score = await engine._score_groundedness(
            best_node.state.draft_answer,
            best_node.state.retrieved_context,
        )
        relevance_score = await engine._score_relevance(
            best_node.state.draft_answer,
            question,
        )

    result = MCTSResult(
        answer=best_node.state.draft_answer or "MCTS could not generate an answer.",
        confidence=best_node.q_value,
        path=path,
        tree_stats=stats,
        groundedness_score=groundedness_score,
        relevance_score=relevance_score,
        iterations_used=min(max_iterations, stats.get("total_simulations", 0)),
        total_nodes=stats.get("total_nodes", 0),
    )

    logger.info(
        "MCTS search complete: confidence=%.3f, nodes=%d, answer=%d chars",
        result.confidence, result.total_nodes, len(result.answer),
    )

    return result
