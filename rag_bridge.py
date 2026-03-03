"""
Bridge to RAG_base — imports RAGService and provides graph-structure access.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from config import settings

logger = logging.getLogger(__name__)

# ── Make RAG_base importable ────────────────────────────────
_rag_base = settings.RAG_BASE_PATH
if str(_rag_base) not in sys.path:
    sys.path.insert(0, str(_rag_base))
# Also need RAG-Anything on the path (RAG_base does this internally too)
_rag_anything_path = _rag_base / "RAG-Anything"
if str(_rag_anything_path) not in sys.path:
    sys.path.insert(0, str(_rag_anything_path))

from rag_service import RAGService  # noqa: E402

# ── Singleton ───────────────────────────────────────────────
_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService(_rag_base)
    return _service


# ── Query helpers ───────────────────────────────────────────

async def query_rag(
    question: str,
    *,
    mode: str = "hybrid",
    user_roles: list[str] | None = None,
    company_id: str | None = None,
) -> str:
    """Execute a query through RAG-Anything and return the answer string."""
    svc = get_rag_service()
    svc.initialize()
    return await svc.aquery(question, mode=mode, user_roles=user_roles, company_id=company_id)


def query_rag_sync(
    question: str,
    *,
    mode: str = "hybrid",
    user_roles: list[str] | None = None,
    company_id: str | None = None,
) -> str:
    svc = get_rag_service()
    svc.initialize()
    return svc.query(question, mode=mode, user_roles=user_roles, company_id=company_id)


# ── Graph structure access ──────────────────────────────────

def _storage_dir() -> Path:
    return _rag_base / "rag_storage"


def load_graph_structure() -> dict[str, Any]:
    """
    Load the content graph hierarchy from RAG_base storage.

    Returns dict with:
        entities: list[dict]   — entity names + doc associations
        chunks:   list[dict]   — text chunks with metadata
        graph:    nx.Graph     — full entity-relation graph (if graphml exists)
    """
    storage = _storage_dir()
    result: dict[str, Any] = {"entities": [], "chunks": [], "graph": None}

    # 1. Entities
    entities_path = storage / "kv_store_full_entities.json"
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities_raw = json.load(f)
        for doc_id, doc_data in entities_raw.items():
            result["entities"].append({
                "doc_id": doc_id,
                "entity_names": doc_data.get("entity_names", []),
            })

    # 2. Text chunks
    chunks_path = storage / "kv_store_text_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)
        for chunk_id, chunk_data in chunks_raw.items():
            result["chunks"].append({
                "chunk_id": chunk_id,
                "content": chunk_data.get("content", ""),
                "file_path": chunk_data.get("file_path", ""),
                "doc_id": chunk_data.get("full_doc_id", ""),
                "company_id": chunk_data.get("company_id"),
                "role_ids": chunk_data.get("role_ids", []),
                "tokens": chunk_data.get("tokens", 0),
            })

    # 3. GraphML (entity-relation graph)
    graphml_path = storage / "graph_chunk_entity_relation.graphml"
    if graphml_path.exists():
        try:
            result["graph"] = nx.read_graphml(graphml_path)
        except Exception as e:
            logger.warning("Failed to load graphml: %s", e)

    return result


def get_entity_summary() -> list[dict[str, Any]]:
    """
    Return a lightweight summary of all entities for the Planner Agent.
    Each item: {"doc_id": str, "file_path": str, "entity_names": list[str], "preview": str}
    """
    gs = load_graph_structure()
    entities = gs["entities"]
    chunks = gs["chunks"]

    # Build doc_id → chunk mapping
    doc_chunks: dict[str, list[dict]] = {}
    for c in chunks:
        doc_chunks.setdefault(c["doc_id"], []).append(c)

    summaries = []
    for ent in entities:
        doc_id = ent["doc_id"]
        related_chunks = doc_chunks.get(doc_id, [])
        file_path = related_chunks[0]["file_path"] if related_chunks else "unknown"
        preview = related_chunks[0]["content"][:200] if related_chunks else ""

        summaries.append({
            "doc_id": doc_id,
            "file_path": file_path,
            "entity_names": ent["entity_names"],
            "preview": preview,
        })

    return summaries


def get_chunks_for_entities(entity_names: list[str]) -> list[dict[str, Any]]:
    """
    Given a list of entity names, find and return matching text chunks.
    Matches by checking if any entity name appears in the chunk content.
    """
    gs = load_graph_structure()
    matching = []
    entity_lower = [e.lower() for e in entity_names]

    for chunk in gs["chunks"]:
        content_lower = chunk["content"].lower()
        if any(ent in content_lower for ent in entity_lower):
            matching.append(chunk)

    return matching
