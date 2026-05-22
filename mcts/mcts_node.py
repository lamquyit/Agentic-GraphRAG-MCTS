"""
MCTS Node, State, Action, and Result data structures.

MCTSNode represents a single node in the search tree.
MCTSState captures the retrieval/generation state at that node.
MCTSAction describes what action was taken to reach the node.
MCTSResult is the final output returned to the caller.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MCTSActionType(Enum):
    """Available actions at each MCTS node."""

    ROOT = "root"                        # Initial root node (no action taken)
    REWRITE_QUERY = "rewrite_query"      # Use query_rewriter to generate a new query
    SELECT_SECTIONS = "select_sections"  # Use planner to choose different sections
    DEEP_READ = "deep_read"              # Use executor to deep-read selected sections
    CHANGE_MODE = "change_mode"          # Switch RAG mode (hybrid / local / global)
    SYNTHESIZE = "synthesize"            # Synthesize final answer from accumulated context


@dataclass
class MCTSAction:
    """Describes the action taken to create a child node."""

    type: MCTSActionType
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __repr__(self) -> str:
        return f"MCTSAction({self.type.value}, {self.description or self.params})"


@dataclass
class MCTSState:
    """
    Retrieval-generation state at a specific node in the MCTS tree.

    This is independent of LangGraph's AgenticRAGState — it captures only
    the information needed for MCTS search decisions.
    """

    query: str                                         # Current query (may have been rewritten)
    original_question: str                             # Original user question (never changes)
    selected_sections: list[dict[str, Any]] = field(default_factory=list)
    retrieved_context: str = ""                        # Accumulated context from retrieval
    draft_answer: str = ""                             # Current draft answer
    depth: int = 0                                     # Depth in the tree
    mode: str = "hybrid"                               # RAG mode (hybrid / local / global)
    reasoning_history: list[str] = field(default_factory=list)  # Trace of reasoning steps
    storage_name: str = "rag_storage"

    def clone(self, **overrides) -> MCTSState:
        """Create a shallow copy with optional field overrides."""
        import copy
        new = copy.copy(self)
        # Deep-copy mutable collections FIRST to avoid cross-node mutation
        new.selected_sections = list(self.selected_sections)
        new.reasoning_history = list(self.reasoning_history)
        # Then apply overrides (so caller-provided lists are kept as-is)
        for k, v in overrides.items():
            setattr(new, k, v)
        return new


@dataclass
class MCTSNode:
    """
    A single node in the MCTS search tree.

    Tracks visit statistics (N, W, Q) and parent-child relationships
    for UCT-based selection.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: MCTSState = field(default_factory=lambda: MCTSState(query="", original_question=""))
    action: MCTSAction = field(default_factory=lambda: MCTSAction(type=MCTSActionType.ROOT))
    parent: Optional[MCTSNode] = field(default=None, repr=False)
    children: list[MCTSNode] = field(default_factory=list, repr=False)

    # ── MCTS statistics ──────────────────────────────────────
    visit_count: int = 0        # N(s) — number of visits
    total_reward: float = 0.0   # W(s) — sum of rewards
    q_value: float = 0.0        # Q(s) = W(s) / N(s)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_fully_expanded(self) -> bool:
        """A node is fully expanded if it has been visited and has children."""
        return self.visit_count > 0 and len(self.children) > 0

    def add_child(self, action: MCTSAction, state: MCTSState) -> MCTSNode:
        """Create and attach a child node."""
        child = MCTSNode(
            state=state,
            action=action,
            parent=self,
        )
        self.children.append(child)
        return child

    def best_child(self) -> Optional[MCTSNode]:
        """Return the child with the highest Q-value (exploitation only)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.q_value)

    def most_visited_child(self) -> Optional[MCTSNode]:
        """Return the child with the most visits (most robust choice)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visit_count)

    def __repr__(self) -> str:
        return (
            f"MCTSNode(id={self.id}, action={self.action.type.value}, "
            f"N={self.visit_count}, Q={self.q_value:.3f}, "
            f"children={len(self.children)}, depth={self.state.depth})"
        )


@dataclass
class MCTSResult:
    """Final output from a complete MCTS search."""

    answer: str                                   # Best answer found
    confidence: float                             # Q-value of the best path
    path: list[dict[str, Any]]                    # Trace of actions from root to best leaf
    tree_stats: dict[str, Any]                    # Statistics about the search tree
    groundedness_score: float = 0.0               # Groundedness of the best answer
    relevance_score: float = 0.0                  # Relevance of the best answer
    iterations_used: int = 0                      # Number of MCTS iterations executed
    total_nodes: int = 0                          # Total nodes in the tree

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "path": self.path,
            "tree_stats": self.tree_stats,
            "groundedness_score": self.groundedness_score,
            "relevance_score": self.relevance_score,
            "iterations_used": self.iterations_used,
            "total_nodes": self.total_nodes,
        }
