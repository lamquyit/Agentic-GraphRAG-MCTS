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

# ── RAG Service Initialization ───────────────────────────────
_rag_services: dict[str, RAGService] = {}

def get_rag_service(storage_name: str = "rag_storage") -> RAGService:
    """Get or create a RAGService instance for a specific storage."""
    if storage_name not in _rag_services:
        _rag_services[storage_name] = RAGService(_rag_base, storage_name=storage_name)
    return _rag_services[storage_name]


# ── Query helpers ───────────────────────────────────────────

async def query_rag(
    question: str,
    *,
    mode: str = "hybrid",
    storage_name: str = "rag_storage"
) -> str:
    """Execute a query through RAG-Anything and return the answer string."""
    svc = get_rag_service(storage_name)
    svc.initialize()
    return await svc.aquery(question, mode=mode)


async def aquery_rag_with_context(
    question: str,
    *,
    mode: str = "hybrid",
    storage_name: str = "rag_storage"
) -> dict[str, Any]:
    """Execute query (async) and return {'answer': str, 'contexts': list[str]}"""
    svc = get_rag_service(storage_name)
    svc.initialize()
    return await svc.aquery_with_context(question, mode=mode)


def query_rag_sync(
    question: str,
    *,
    mode: str = "hybrid",
    storage_name: str = "rag_storage"
) -> str:
    """Execute query (sync) và trả về nội dung text."""
    svc = get_rag_service(storage_name)
    svc.initialize()
    return svc.query(question, mode=mode)


def query_rag_with_context(
    question: str,
    *,
    mode: str = "hybrid",
    storage_name: str = "rag_storage"
) -> dict[str, Any]:
    """Execute query and return {'answer': str, 'contexts': list[str]}"""
    svc = get_rag_service(storage_name)
    svc.initialize()
    return svc.query_with_context(question, mode=mode)


# ── Document processing helpers ─────────────────────────────

def calculate_md5_bytes(file_bytes: bytes) -> str:
    """Calculate MD5 hash of file content for deduplication."""
    import hashlib
    hash_md5 = hashlib.md5()
    hash_md5.update(file_bytes)
    return hash_md5.hexdigest()


def load_processed_metadata() -> dict[str, Any]:
    """Load processed_files.json from RAG_base."""
    meta_path = _rag_base / "processed_files.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": []}


def process_document(
    file_path: str,
    storage_name: str = "rag_storage",
    log_file: str = None
) -> str:
    """
    Index a document through RAG-Anything.

    Args:
        file_path: absolute path to the saved file
        storage_name: name of the storage folder (rag_storage, etc.)

    Returns:
        doc_id (MD5 hash of the file content)
    """
    svc = get_rag_service(storage_name)
    output_dir = _rag_base / "output_for_report"
    svc.process_document(
        file_path,
        str(output_dir),
        log_file=log_file
    )
    # Return the MD5 hash
    with open(file_path, "rb") as f:
        return calculate_md5_bytes(f.read())


def get_input_dir() -> Path:
    """Return the input directory for uploaded files (inside RAG_base)."""
    d = _rag_base / "input"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Graph structure access ──────────────────────────────────

def _storage_dir(storage_name: str = "rag_storage") -> Path:
    return _rag_base / storage_name


def load_graph_structure(storage_name: str = "rag_storage") -> dict[str, Any]:
    """
    Load the content graph hierarchy from RAG_base storage.

    Returns dict with:
        entities: list[dict]   — entity names + doc associations
        chunks:   list[dict]   — text chunks with metadata
        graph:    nx.Graph     — full entity-relation graph (if graphml exists)
    """
    storage = _storage_dir(storage_name)
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


def get_entity_summary(storage_name: str = "rag_storage") -> list[dict[str, Any]]:
    """
    Return a lightweight summary of all entities for the Planner Agent.
    Each item: {"doc_id": str, "file_path": str, "entity_names": list[str], "preview": str}
    """
    gs = load_graph_structure(storage_name)
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


def get_chunks_for_entities(entity_names: list[str], storage_name: str = "rag_storage") -> list[dict[str, Any]]:
    """
    Given a list of entity names, find and return matching text chunks.
    Matches by checking if any entity name appears in the chunk content.
    """
    gs = load_graph_structure(storage_name)
    matching = []
    entity_lower = [e.lower() for e in entity_names]

    for chunk in gs["chunks"]:
        content_lower = chunk["content"].lower()
        if any(ent in content_lower for ent in entity_lower):
            matching.append(chunk)

    return matching
