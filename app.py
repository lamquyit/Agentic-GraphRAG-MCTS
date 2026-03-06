"""
Agentic RAG — FastAPI Server

Wraps the LangGraph pipeline behind a REST API on port 8002.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings  # noqa: E402
from graph.agentic_rag_graph import run_agentic_rag  # noqa: E402

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_rag")

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG API",
    description=(
        "Self-Correcting RAG with Locate-then-Read planning, "
        "powered by LangGraph + RAG-Anything."
    ),
    version="1.0",
)


# ── Pydantic Models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = "What is the main topic of the uploaded document?"
    mode: str = "hybrid"
    user_roles: list[str] | None = None
    company_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    status: str
    retries_used: int = 0
    groundedness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    planner_reasoning: str = ""
    sections_located: int = 0
    elapsed_sec: float = 0.0


class IndexResponse(BaseModel):
    doc_id: str
    message: str
    filename: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────

@app.get("/", summary="Health Check")
def health_check():
    return {
        "status": "ok",
        "service": "Agentic RAG API",
        "features": [
            "Self-Correcting RAG (Groundedness + Relevance Gates)",
            "Locate-then-Read Planning",
            "CoT Query Rewriting",
            "Document Upload & Indexing",
        ],
    }


# ── Document Upload ──────────────────────────────────────────
@app.post(
    "/index-doc",
    summary="Upload and Index Document",
    description=(
        "Upload a file and index it for RAG retrieval.\n\n"
        "- `role_ids`: comma-separated role names (e.g. `developer,qa_lead`)\n"
        "- `company_id`: tenant isolation ID\n"
        "- Deduplication via MD5 hash — skips if already processed"
    ),
    response_model=IndexResponse,
)
async def index_doc(
    file: UploadFile = File(..., description="The document file to upload"),
    role_ids: str = Form("public", description="Comma-separated list of role IDs"),
    company_id: str = Form(..., description="Company ID for tenant isolation"),
):
    from rag_bridge import (
        calculate_md5_bytes,
        load_processed_metadata,
        process_document,
        get_input_dir,
    )

    try:
        content = await file.read()
        doc_md5 = calculate_md5_bytes(content)

        roles = [r.strip() for r in role_ids.split(",") if r.strip()]
        if not roles:
            roles = ["public"]

        # Dedup check
        metadata = load_processed_metadata()
        for f_meta in metadata.get("files", []):
            if f_meta.get("doc_id") == doc_md5 and f_meta.get("company_id") == company_id:
                return IndexResponse(
                    doc_id=doc_md5,
                    message="existed",
                    filename=f_meta.get("filename"),
                )

        # Save file to RAG_base/input/
        input_dir = get_input_dir()
        file_path = input_dir / file.filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Process through RAG-Anything
        logger.info("Indexing document: %s (roles=%s, company=%s)", file.filename, roles, company_id)
        process_document(
            str(file_path),
            role_ids=roles,
            company_id=company_id,
        )

        return IndexResponse(
            doc_id=doc_md5,
            message="processed",
            filename=file.filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Index error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query",
    summary="Agentic RAG Query",
    description=(
        "Execute a query through the full Agentic RAG pipeline:\n\n"
        "1. **Plan** — Locate relevant sections in the content graph\n"
        "2. **Execute** — Deep-read via RAG-Anything\n"
        "3. **Verify** — Groundedness Gate + Relevance Gate\n"
        "4. **Retry** — Auto-regenerate or rewrite query if gates fail"
    ),
    response_model=QueryResponse,
)
async def query(req: QueryRequest):
    try:
        t0 = time.perf_counter()

        final_state = await run_agentic_rag(
            question=req.question,
            mode=req.mode,
            user_roles=req.user_roles,
            company_id=req.company_id,
        )

        elapsed = time.perf_counter() - t0

        gr = final_state.get("groundedness_result", {})
        rel = final_state.get("relevance_result", {})

        return QueryResponse(
            answer=final_state.get("final_answer", ""),
            status=final_state.get("status", "unknown"),
            retries_used=final_state.get("retry_count", 0),
            groundedness_score=gr.get("score"),
            relevance_score=rel.get("score"),
            planner_reasoning=final_state.get("planner_reasoning", ""),
            sections_located=len(final_state.get("located_sections", [])),
            elapsed_sec=round(elapsed, 2),
        )

    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query/simple",
    summary="Simple RAG Query (bypass agentic pipeline)",
    description="Direct pass-through to RAG-Anything without verification gates.",
    response_model=QueryResponse,
)
async def query_simple(req: QueryRequest):
    """Bypass the agentic pipeline — useful for comparison / debugging."""
    from rag_bridge import query_rag

    try:
        t0 = time.perf_counter()
        answer = await query_rag(
            req.question,
            mode=req.mode,
            user_roles=req.user_roles,
            company_id=req.company_id,
        )
        elapsed = time.perf_counter() - t0

        return QueryResponse(
            answer=answer,
            status="completed (simple)",
            elapsed_sec=round(elapsed, 2),
        )
    except Exception as e:
        logger.exception("Simple query error")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)
