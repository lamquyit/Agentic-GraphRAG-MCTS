"""
MCTS Action Space — defines available actions and executes them via existing agents.

Each action is a thin wrapper around the agents already implemented in
agents/planner.py, agents/executor.py, agents/query_rewriter.py, etc.
This module does NOT duplicate agent logic.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from mcts.mcts_node import MCTSAction, MCTSActionType, MCTSState

logger = logging.getLogger(__name__)

# RAG modes available for CHANGE_MODE action
_RAG_MODES = ["hybrid", "local", "global"]


def get_available_actions(node_state: MCTSState, max_depth: int = 4) -> list[MCTSAction]:
    """
    Determine which actions are available at a given state.

    Rules:
      - Depth 0 (root): SELECT_SECTIONS (must plan first)
      - Depth 1: DEEP_READ, REWRITE_QUERY
      - Depth 2+: REWRITE_QUERY, CHANGE_MODE, SYNTHESIZE
      - At max_depth: only SYNTHESIZE (force termination)
      - DEEP_READ requires sections to be selected
      - SYNTHESIZE requires context to exist
    """
    depth = node_state.depth
    actions: list[MCTSAction] = []

    # At max depth, only allow synthesis to terminate
    if depth >= max_depth:
        if node_state.retrieved_context:
            actions.append(MCTSAction(
                type=MCTSActionType.SYNTHESIZE,
                description="Force synthesis at max depth",
            ))
        return actions

    # Depth 0: must start with planning (section selection)
    if depth == 0:
        actions.append(MCTSAction(
            type=MCTSActionType.SELECT_SECTIONS,
            description="Initial section selection via Planner",
        ))
        return actions

    # General actions available at depth >= 1
    # 1. Rewrite query — always available (creates a new search direction)
    actions.append(MCTSAction(
        type=MCTSActionType.REWRITE_QUERY,
        description="Rewrite query for better retrieval",
    ))

    # 2. Select different sections — always available
    actions.append(MCTSAction(
        type=MCTSActionType.SELECT_SECTIONS,
        description="Re-plan with different section selection",
    ))

    # 3. Deep read — available only if sections have been selected
    if node_state.selected_sections:
        actions.append(MCTSAction(
            type=MCTSActionType.DEEP_READ,
            description="Deep-read selected sections via Executor",
        ))

    # 4. Change RAG mode — available if current mode can be changed
    other_modes = [m for m in _RAG_MODES if m != node_state.mode]
    for mode in other_modes:
        actions.append(MCTSAction(
            type=MCTSActionType.CHANGE_MODE,
            params={"new_mode": mode},
            description=f"Switch RAG mode to '{mode}'",
        ))

    # 5. Synthesize — available if we have context
    if node_state.retrieved_context:
        actions.append(MCTSAction(
            type=MCTSActionType.SYNTHESIZE,
            description="Synthesize answer from accumulated context",
        ))

    return actions


async def execute_action(action: MCTSAction, state: MCTSState) -> MCTSState:
    """
    Execute an MCTS action and return a NEW state (does not mutate input).

    Each action delegates to the existing agent implementations.
    """
    new_state = state.clone(depth=state.depth + 1)

    if action.type == MCTSActionType.SELECT_SECTIONS:
        new_state = await _action_select_sections(new_state)

    elif action.type == MCTSActionType.REWRITE_QUERY:
        new_state = await _action_rewrite_query(new_state)

    elif action.type == MCTSActionType.DEEP_READ:
        new_state = await _action_deep_read(new_state)

    elif action.type == MCTSActionType.CHANGE_MODE:
        new_mode = action.params.get("new_mode", "hybrid")
        new_state.mode = new_mode
        new_state.reasoning_history.append(f"Changed RAG mode to '{new_mode}'")

    elif action.type == MCTSActionType.SYNTHESIZE:
        new_state = await _action_synthesize(new_state)

    else:
        logger.warning("Unknown action type: %s", action.type)

    return new_state


# ══════════════════════════════════════════════════════════════
# PRIVATE — Action implementations (wrappers around existing agents)
# ══════════════════════════════════════════════════════════════

async def _action_select_sections(state: MCTSState) -> MCTSState:
    """Wrapper around agents/planner.py — run_planner()."""
    from agents.planner import run_planner

    result = await run_planner(
        question=state.query,
        storage_name=state.storage_name,
    )

    state.selected_sections = result.get("selected_sections", [])
    state.query = result.get("enhanced_query", state.query)
    state.reasoning_history.append(
        f"Planner selected {len(state.selected_sections)} sections: "
        f"{result.get('reasoning', '')[:120]}"
    )

    logger.debug("MCTS SELECT_SECTIONS: %d sections selected", len(state.selected_sections))
    return state


async def _action_rewrite_query(state: MCTSState) -> MCTSState:
    """Wrapper around agents/query_rewriter.py — rewrite_query()."""
    from agents.query_rewriter import rewrite_query

    new_query = await rewrite_query(
        original_question=state.original_question,
        failed_answer=state.draft_answer,
        relevance_feedback=f"MCTS exploration — trying alternative query formulation at depth {state.depth}",
        missing_aspects=None,
    )

    state.reasoning_history.append(
        f"Rewrote query: '{state.query[:60]}' → '{new_query[:60]}'"
    )
    state.query = new_query

    logger.debug("MCTS REWRITE_QUERY: '%s'", new_query[:80])
    return state


async def _action_deep_read(state: MCTSState) -> MCTSState:
    """Wrapper around agents/executor.py — run_executor()."""
    from agents.executor import run_executor

    result = await run_executor(
        question=state.original_question,
        enhanced_query=state.query,
        selected_sections=state.selected_sections,
        mode=state.mode,
        storage_name=state.storage_name,
    )

    state.draft_answer = result.get("draft_answer", "")
    # Accumulate context rather than replacing
    new_context = result.get("retrieved_context", "")
    if new_context:
        if state.retrieved_context:
            state.retrieved_context += "\n\n---\n\n" + new_context
        else:
            state.retrieved_context = new_context

    state.reasoning_history.append(
        f"Deep-read via Executor: got {len(state.draft_answer)} char answer, "
        f"{len(new_context)} char context"
    )

    logger.debug("MCTS DEEP_READ: answer=%d chars, context=%d chars",
                 len(state.draft_answer), len(new_context))
    return state


async def _action_synthesize(state: MCTSState) -> MCTSState:
    """Synthesize a final answer from accumulated context using LLM."""
    from llm_client import call_llm

    prompt = (
        f"Based on the following retrieved information, provide a comprehensive and "
        f"accurate answer to the question.\n\n"
        f"## Question\n{state.original_question}\n\n"
        f"## Retrieved Context\n{state.retrieved_context[:4000]}\n\n"
        f"Provide a clear, well-structured answer based ONLY on the context above."
    )

    answer = await call_llm(prompt, temperature=0.3)
    state.draft_answer = answer.strip()
    state.reasoning_history.append(
        f"Synthesized answer from {len(state.retrieved_context)} chars of context"
    )

    logger.debug("MCTS SYNTHESIZE: answer=%d chars", len(state.draft_answer))
    return state
