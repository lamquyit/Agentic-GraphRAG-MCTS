"""
MCTS Engine — core Selection / Expansion / Simulation / Backpropagation loop.

Implements the four canonical MCTS phases using UCT (Upper Confidence Bound
applied to Trees) for node selection and existing Grader agents for reward
estimation during simulation.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from mcts.mcts_node import MCTSNode, MCTSState, MCTSAction, MCTSActionType
from mcts.mcts_actions import get_available_actions, execute_action

logger = logging.getLogger(__name__)


class MCTSEngine:
    """
    Monte Carlo Tree Search engine for Agentic RAG.

    Uses UCT-based selection, agent-driven expansion, Grader-based simulation,
    and standard backpropagation to explore the retrieval-generation space.
    """

    def __init__(
        self,
        max_iterations: int = 8,
        max_depth: int = 4,
        exploration_weight: float = 1.414,
        groundedness_weight: float = 0.6,
        relevance_weight: float = 0.4,
        min_confidence: float = 0.7,
        parallel_expansion: bool = True,
    ):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.groundedness_weight = groundedness_weight
        self.relevance_weight = relevance_weight
        self.min_confidence = min_confidence
        self.parallel_expansion = parallel_expansion

        # Tracking
        self._total_nodes = 0
        self._total_simulations = 0

    # ══════════════════════════════════════════════════════════
    # 1. SELECTION — traverse from root to a promising leaf using UCT
    # ══════════════════════════════════════════════════════════

    def select(self, root: MCTSNode) -> MCTSNode:
        """
        Walk down the tree from root, at each level choosing the child
        with the highest UCT score, until reaching a leaf node.
        """
        node = root
        while not node.is_leaf:
            node = max(node.children, key=lambda c: self._uct_score(c))
        return node

    def _uct_score(self, node: MCTSNode) -> float:
        """
        UCT = Q(s)/N(s) + C * sqrt(ln(N_parent) / N(s))

        - Exploitation: Q(s)/N(s) — average reward
        - Exploration: C * sqrt(ln(N_parent) / N(s)) — bonus for less-visited nodes
        - Unvisited nodes get infinity (explore them first)
        """
        if node.visit_count == 0:
            return float("inf")

        exploitation = node.q_value
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visit_count) / node.visit_count
        )
        return exploitation + exploration

    # ══════════════════════════════════════════════════════════
    # 2. EXPANSION — create child nodes by executing available actions
    # ══════════════════════════════════════════════════════════

    async def expand(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Expand a leaf node by generating children for each available action.

        If parallel_expansion is True, all actions are executed concurrently
        via asyncio.gather to reduce latency.
        """
        available_actions = get_available_actions(
            node.state, max_depth=self.max_depth
        )

        if not available_actions:
            logger.debug("No actions available at node %s (depth=%d)", node.id, node.state.depth)
            return []

        if self.parallel_expansion:
            children = await self._expand_parallel(node, available_actions)
        else:
            children = await self._expand_sequential(node, available_actions)

        self._total_nodes += len(children)
        logger.debug(
            "Expanded node %s: %d children created (depth=%d)",
            node.id, len(children), node.state.depth,
        )
        return children

    async def _expand_parallel(
        self, node: MCTSNode, actions: list[MCTSAction]
    ) -> list[MCTSNode]:
        """Execute all actions in parallel and create child nodes."""
        tasks = [execute_action(action, node.state) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        children = []
        for action, result in zip(actions, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Action %s failed during expansion: %s", action.type.value, result
                )
                continue
            child = node.add_child(action=action, state=result)
            children.append(child)

        return children

    async def _expand_sequential(
        self, node: MCTSNode, actions: list[MCTSAction]
    ) -> list[MCTSNode]:
        """Execute actions one by one and create child nodes."""
        children = []
        for action in actions:
            try:
                new_state = await execute_action(action, node.state)
                child = node.add_child(action=action, state=new_state)
                children.append(child)
            except Exception as e:
                logger.warning(
                    "Action %s failed during expansion: %s", action.type.value, e
                )
        return children

    # ══════════════════════════════════════════════════════════
    # 3. SIMULATION — estimate the value of a node via rollout
    # ══════════════════════════════════════════════════════════

    async def simulate(self, node: MCTSNode) -> float:
        """
        Estimate the reward of a node by evaluating its current state.

        Uses the existing Grader agents (Groundedness + Relevance) as the
        reward function — no new scoring logic is introduced.

        Returns:
            float in [0.0, 1.0] — weighted combination of groundedness and relevance
        """
        state = node.state
        self._total_simulations += 1

        # If no draft answer yet, run a quick deep-read to get one
        if not state.draft_answer:
            try:
                from agents.executor import run_executor
                result = await run_executor(
                    question=state.original_question,
                    enhanced_query=state.query,
                    selected_sections=state.selected_sections,
                    mode=state.mode,
                    storage_name=state.storage_name,
                )
                state.draft_answer = result.get("draft_answer", "")
                if not state.retrieved_context:
                    state.retrieved_context = result.get("retrieved_context", "")
            except Exception as e:
                logger.warning("Simulation deep-read failed: %s", e)
                return 0.0

        # If still no answer, reward is 0
        if not state.draft_answer.strip():
            return 0.0

        # Score using existing graders
        groundedness_score = await self._score_groundedness(
            state.draft_answer, state.retrieved_context
        )
        relevance_score = await self._score_relevance(
            state.draft_answer, state.original_question
        )

        # Weighted combination
        reward = (
            self.groundedness_weight * groundedness_score
            + self.relevance_weight * relevance_score
        )

        logger.debug(
            "Simulation for node %s: groundedness=%.2f, relevance=%.2f, reward=%.3f",
            node.id, groundedness_score, relevance_score, reward,
        )
        return reward

    async def _score_groundedness(self, answer: str, context: str) -> float:
        """Delegate to existing grader_groundedness agent."""
        from agents.grader_groundedness import check_groundedness
        try:
            result = await check_groundedness(answer, context)
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.warning("Groundedness scoring failed: %s", e)
            return 0.0

    async def _score_relevance(self, answer: str, question: str) -> float:
        """Delegate to existing grader_relevance agent."""
        from agents.grader_relevance import check_relevance
        try:
            result = await check_relevance(answer, question)
            return float(result.get("score", 0.0))
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
            return 0.0

    # ══════════════════════════════════════════════════════════
    # 4. BACKPROPAGATION — update Q-values from leaf to root
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def backpropagate(node: MCTSNode, reward: float) -> None:
        """
        Propagate the simulation reward up from the given node to the root.

        Updates visit_count, total_reward, and q_value at each ancestor.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current.q_value = current.total_reward / current.visit_count
            current = current.parent

    # ══════════════════════════════════════════════════════════
    # MAIN LOOP — orchestrate the 4 phases
    # ══════════════════════════════════════════════════════════

    async def search(self, root: MCTSNode) -> MCTSNode:
        """
        Run the full MCTS search loop for max_iterations.

        Returns the best leaf node (by visit count — most robust choice).
        """
        self._total_nodes = 1  # root
        self._total_simulations = 0

        for iteration in range(self.max_iterations):
            logger.info(
                "MCTS iteration %d/%d (nodes=%d, sims=%d)",
                iteration + 1, self.max_iterations,
                self._total_nodes, self._total_simulations,
            )

            # 1. Selection
            leaf = self.select(root)

            # 2. Expansion
            children = await self.expand(leaf)

            if not children:
                # No expansion possible — simulate the leaf itself
                reward = await self.simulate(leaf)
                self.backpropagate(leaf, reward)
                continue

            # 3. Simulation — evaluate each new child
            # Run simulations in parallel for speed
            if self.parallel_expansion:
                sim_tasks = [self.simulate(child) for child in children]
                rewards = await asyncio.gather(*sim_tasks, return_exceptions=True)
                for child, reward in zip(children, rewards):
                    if isinstance(reward, Exception):
                        logger.warning("Simulation failed for node %s: %s", child.id, reward)
                        reward = 0.0
                    # 4. Backpropagation
                    self.backpropagate(child, reward)
            else:
                for child in children:
                    try:
                        reward = await self.simulate(child)
                    except Exception as e:
                        logger.warning("Simulation failed for node %s: %s", child.id, e)
                        reward = 0.0
                    self.backpropagate(child, reward)

            # Early termination: if we found a high-confidence answer
            best = self._get_best_leaf(root)
            if best and best.q_value >= self.min_confidence and best.visit_count >= 2:
                logger.info(
                    "Early termination: best node %s has Q=%.3f (>= %.3f) with N=%d",
                    best.id, best.q_value, self.min_confidence, best.visit_count,
                )
                break

        return self._get_best_leaf(root) or root

    def _get_best_leaf(self, root: MCTSNode) -> MCTSNode | None:
        """Find the leaf with the highest Q-value across the entire tree."""
        best = None
        best_score = -1.0
        stack = [root]

        while stack:
            node = stack.pop()
            if node.state.draft_answer and node.q_value > best_score:
                best = node
                best_score = node.q_value
            stack.extend(node.children)

        return best

    # ── Utility ────────────────────────────────────────────────

    def collect_stats(self, root: MCTSNode) -> dict[str, Any]:
        """Collect statistics about the search tree for debugging/logging."""
        total_nodes = 0
        max_depth = 0
        total_visits = 0
        actions_taken: dict[str, int] = {}

        stack = [root]
        while stack:
            node = stack.pop()
            total_nodes += 1
            total_visits += node.visit_count
            max_depth = max(max_depth, node.state.depth)

            action_name = node.action.type.value
            actions_taken[action_name] = actions_taken.get(action_name, 0) + 1

            stack.extend(node.children)

        return {
            "total_nodes": total_nodes,
            "max_depth_reached": max_depth,
            "total_visits": total_visits,
            "total_simulations": self._total_simulations,
            "iterations": self.max_iterations,
            "actions_distribution": actions_taken,
        }

    @staticmethod
    def trace_path(node: MCTSNode) -> list[dict[str, Any]]:
        """Trace the path from a node back to the root."""
        path = []
        current = node
        while current is not None:
            path.append({
                "node_id": current.id,
                "action": current.action.type.value,
                "action_desc": current.action.description,
                "query": current.state.query[:100],
                "depth": current.state.depth,
                "q_value": current.q_value,
                "visit_count": current.visit_count,
                "has_answer": bool(current.state.draft_answer),
            })
            current = current.parent

        path.reverse()  # root → leaf order
        return path
