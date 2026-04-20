import os
import sys

# Ensure MinerU and other tools in the environment are in PATH
ENV_BIN = "/workspace/miniconda3/envs/ragvenv310/bin"
if ENV_BIN not in os.environ["PATH"]:
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ['PATH']}"

import asyncio
import threading
import time
import logging
from pathlib import Path

# Suppress harmless font warnings from Mineru/magic-pdf
logging.getLogger("root").addFilter(lambda record: "WenQuanYi font not found" not in record.getMessage())
logging.getLogger("magic-pdf").setLevel(logging.ERROR)
from dotenv import load_dotenv

# Ensure local RAG-Anything package is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG-Anything"))

from raganything import RAGAnything, RAGAnythingConfig
print(f"DEBUG: Loaded RAGAnything from {RAGAnything.__module__} at {sys.modules['raganything'].__file__}")
from lightrag.utils import EmbeddingFunc
from llm_provider import openai_complete_with_fallback, openai_embed_with_fallback, get_providers, qwen_rerank

# Metrics & Logging imports
from logging_util import setup_memory_logger, get_logs, parse_document_metrics, cleanup_logger
from processed_util import add_processed_file

load_dotenv()

class BackgroundLoop:
    """
    vai trò : chạy background thread có chứa asyncio event loop
    class này giúp chạy các tác vụ nặng ở dưới nền mà không làm treo giao diện UI
    """
    def __init__(self):
        self.loop = None
        self.thread = None

    def start(self):
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()
        def _run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def run_coro(self, coro):
        """Submit a coroutine to the background loop and wait for result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result()


class RAGFactory:
    """
    Factory Pattern: Responsible for creating and configuring the RAGAnything instance.
    Decouples configuration/creation logic from usage.
    vai trò : tạo ra RAGAnything instance và cấu hình các tham số cần thiết
    """
    @staticmethod
    def create_rag_sync(root_path: Path, storage_name: str = "rag_storage") -> RAGAnything:
        # Validate: cần ít nhất 1 API key
        providers = get_providers()  # raises RuntimeError nếu không có key nào
        primary = providers[0]
        print(f"🔑 Primary LLM provider: {primary.name} (fallback: {', '.join(p.name for p in providers[1:])} )")

        config = RAGAnythingConfig(
            working_dir=str(root_path / storage_name),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        def llm_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=None, 
            **kwargs
        ):
            # Increase max_tokens to prevent JSON truncation in large responses
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 4096
                
            history_messages = history_messages or []
            return openai_complete_with_fallback(
                model_type="llm",
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        def vision_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=None, 
            image_data=None, 
            messages=None, 
            **kwargs
        ):
            # Increase max_tokens to prevent JSON truncation in large responses
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 4096
                
            history_messages = history_messages or []
            if messages:
                return openai_complete_with_fallback(
                    model_type="vision",
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    **kwargs,
                )
            elif image_data:
                built_messages = []
                if system_prompt:
                    built_messages.append({"role": "system", "content": system_prompt})
                
                built_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                })
                
                return openai_complete_with_fallback(
                    model_type="vision",
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=built_messages,
                    **kwargs,
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed_with_fallback(texts),
        )

        return RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs={
                "rerank_model_func": qwen_rerank,
                "min_rerank_score": 0.2, # Hơi thấp để giữ lại nhiều context hơn cho Verifier
            }
        )


class RAGService:
    """
    Service Pattern: Facade for interacting with the RAG system.
    Manages the background loop and async execution, exposing synchronous methods to the UI.
    """
    def __init__(self, root_path: Path, storage_name: str = "rag_storage"):
        self.root_path = root_path
        self.storage_name = storage_name
        self.bg_loop = BackgroundLoop()
        self.bg_loop.start()
        self.rag = None # Lazy initialization

    def initialize(self):
        """Initializes the RAG instance if not already done."""
        if self.rag is None:
            # Run initialization on background thread to be safe with async calls inside
            self.rag = self.bg_loop.run_coro(self._build_and_load())
        return self.rag

    async def _build_and_load(self):
        print("DEBUG: Entering _build_and_load")
        rag = RAGFactory.create_rag_sync(self.root_path, storage_name=self.storage_name)
        print(f"DEBUG: RAGAnything created: {rag}")
        # Force RAGAnything to load existing LightRAG index
        init_res = await rag._ensure_lightrag_initialized()
        print(f"DEBUG: _ensure_lightrag_initialized result: {init_res}")
        print(f"DEBUG: rag.lightrag is now: {rag.lightrag}")
        return rag

    def process_document(self, file_path: str, output_dir: str, log_file: str = None):
        """Sync wrapper for processing a document với thu thập stdout/stderr và lưu log."""
        self.initialize()
        
        # Ensure log directory exists if log_file is provided
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # 1. Setup stdout capture using TeeIO
        import sys
        from io import StringIO
        
        class TeeIO:
            """Captures stdout while also printing to console and optionally a file."""
            def __init__(self, original, file_handle=None):
                self.original = original
                self.buffer = StringIO()
                self.file_handle = file_handle
            
            def write(self, text):
                self.original.write(text)
                self.buffer.write(text)
                if self.file_handle:
                    self.file_handle.write(text)
                    self.file_handle.flush()
            
            def flush(self):
                self.original.flush()
                if self.file_handle:
                    self.file_handle.flush()
            
            def getvalue(self):
                return self.buffer.getvalue()
        
        f_handle = open(log_file, "a", encoding="utf-8") if log_file else None
        tee_stdout = TeeIO(sys.stdout, f_handle)
        tee_stderr = TeeIO(sys.stderr, f_handle)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_stdout, tee_stderr
        
        start_time = time.perf_counter()
        try:
            # 2. Run processing
            result = self.bg_loop.run_coro(
                self.rag.process_document_complete(
                    file_path=file_path,
                    output_dir=output_dir,
                    parse_method="auto",
                )
            )
        finally:
            # Restore stdout/stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if f_handle:
                f_handle.close()
            
            elapsed = time.perf_counter() - start_time
            
            # 3. Get Stats directly from LightRAG KV stores (reliable source)
            try:
                # LightRAG stores actual data in these KV stores — read counts directly
                actual_rag_stats = {"chunks": 0, "entities": 0, "relations": 0}
                try:
                    lg = self.rag.lightrag  # the actual LightRAG instance inside RAGAnything
                    if lg is not None:
                        # text_chunks KV store contains chunk count
                        if hasattr(lg, "text_chunks") and lg.text_chunks:
                            actual_rag_stats["chunks"] = len(getattr(lg.text_chunks, "_data", {}))
                        # full_entities KV store
                        if hasattr(lg, "full_entities") and lg.full_entities:
                            actual_rag_stats["entities"] = len(getattr(lg.full_entities, "_data", {}))
                        # full_relations KV store
                        if hasattr(lg, "full_relations") and lg.full_relations:
                            actual_rag_stats["relations"] = len(getattr(lg.full_relations, "_data", {}))
                except Exception as _stats_e:
                    print(f"[DEBUG] Could not read LightRAG KV stats: {_stats_e}")

                logs = tee_stdout.getvalue() + "\n" + tee_stderr.getvalue()
                
                # Update debug_last_log.txt for quick check
                with open("debug_last_log.txt", "w", encoding="utf-8") as f:
                    f.write(logs)

                # 4. Parse metrics from logs (timing info)
                parsed_metrics = parse_document_metrics(logs)
                metrics = parsed_metrics.copy()
                
                # Inject real rag_stats (overrides the always-zero parsed version)
                metrics["rag_stats"] = actual_rag_stats

                print(f"[DEBUG] Final metrics: {metrics}")
                
                # 4. Save metadata with metrics
                add_processed_file(file_path, elapsed, metrics)

                
                # 5. Print Formatted Report to Console
                from processed_util import format_single_file_report
                
                # Construct temporary file_data for report formatting
                temp_file_data = {
                    "filename": Path(file_path).name,
                    "processing_time_sec": round(elapsed, 2),
                    "phase_stats": {}
                }
                
                # Flatten metrics for report
                # (metrics structure: key -> {stage, duration_sec})
                # We need simple key -> duration in seconds
                t_stats = temp_file_data["phase_stats"]
                
                # Mapping known keys
                if metrics.get("mineru_parsing_time"): t_stats["mineru_parsing"] = metrics["mineru_parsing_time"].get("duration_sec", 0)
                if metrics.get("content_analysis_time"): t_stats["content_analysis"] = metrics["content_analysis_time"].get("duration_sec", 0)
                if metrics.get("multimodal_processing_time"): t_stats["multimodal_processing"] = metrics["multimodal_processing_time"].get("duration_sec", 0)
                if metrics.get("multimodal_description_time"): t_stats["multimodal_description"] = metrics["multimodal_description_time"].get("duration_sec", 0)
                if metrics.get("multimodal_embedding_time"): t_stats["multimodal_embedding"] = metrics["multimodal_embedding_time"].get("duration_sec", 0)
                if metrics.get("multimodal_extraction_time"): t_stats["multimodal_extraction"] = metrics["multimodal_extraction_time"].get("duration_sec", 0)
                if metrics.get("crossmodal_graph_merging_time"): t_stats["crossmodal_graph_merging"] = metrics["crossmodal_graph_merging_time"].get("duration_sec", 0)
                if metrics.get("text_indexing_time"): t_stats["text_indexing"] = metrics["text_indexing_time"].get("duration_sec", 0)
                # Calculate Overhead for report
                
                # Report formatting
                report_str = format_single_file_report(temp_file_data)
                print(f"\n{report_str}\n")
                
            except Exception as e:
                print(f"Error retrieving/parsing metrics: {e}")
                import traceback
                traceback.print_exc()
            
        return result

    async def aquery(self, question: str, mode: str = "hybrid"):
        return await self.rag.aquery(question, mode=mode)

    def query(self, question: str, mode: str = "hybrid"):
        """Sync wrapper for querying."""
        self.initialize()
        return self.bg_loop.run_coro(
            self.rag.aquery(question, mode=mode)
        )

    async def aquery_with_context(self, question: str, mode: str = "hybrid"):
        """Returns both answer and retrieved contexts."""
        # 1. Get retrieved data
        from lightrag import QueryParam
        param = QueryParam(mode=mode)
        retrieval_data = await self.rag.lightrag.aquery_data(question, param=param)
        
        # 2. Get the answer (using standard aquery to benefit from RAGAnything's full pipeline)
        answer = await self.rag.aquery(question, mode=mode)
        
        # 3. Extract contexts
        contexts = [c.get("content", "") for c in retrieval_data.get("data", {}).get("chunks", [])]
        
        return {
            "answer": answer,
            "contexts": contexts
        }

    def query_with_context(self, question: str, mode: str = "hybrid"):
        """Sync wrapper for aquery_with_context."""
        self.initialize()
        return self.bg_loop.run_coro(self.aquery_with_context(question, mode=mode))

    def delete_document(self, doc_id: str) -> dict:
        """
        Xóa toàn bộ dữ liệu liên quan đến một document khỏi RAG storage.

        Args:
            doc_id: MD5 hash của file (được trả về bởi /index-doc).

        Returns:
            dict với các stats về số lượng records đã xóa.
        """
        import json
        import shutil
        import xml.etree.ElementTree as ET

        storage_dir = self.root_path / "rag_storage"
        processed_path = self.root_path / "processed_files.json"

        def load_json(p):
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        def save_json(p, d):
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Tìm full_doc_id trong storage ────────────────────────────────────
        chunks_path = storage_dir / "kv_store_text_chunks.json"
        chunks_data = load_json(chunks_path) or {}

        # full_doc_id có thể là "doc-<md5>" hoặc khớp trực tiếp
        full_doc_ids_to_remove = set()
        for _, chunk in chunks_data.items():
            if not isinstance(chunk, dict):
                continue
            fdid = chunk.get("full_doc_id", "")
            fp = chunk.get("file_path", "")
            # Match by doc_id embedded in full_doc_id, or match by file lookup
            if doc_id in fdid or fdid == f"doc-{doc_id}":
                full_doc_ids_to_remove.add(fdid)

        if not full_doc_ids_to_remove:
            return {
                "doc_id": doc_id,
                "status": "not_found",
                "message": f"doc_id '{doc_id}' not found in storage.",
            }

        stats = {
            "chunks_removed": 0,
            "vdb_chunks_removed": 0,
            "entities_removed": 0,
            "entities_updated": 0,
            "relations_removed": 0,
            "relations_updated": 0,
            "vdb_entities_removed": 0,
            "vdb_relations_removed": 0,
            "graph_nodes_removed": 0,
            "graph_edges_removed": 0,
        }

        # ── 1. Text chunks ────────────────────────────────────────────────────
        chunk_ids = {
            cid for cid, c in chunks_data.items()
            if isinstance(c, dict) and c.get("full_doc_id") in full_doc_ids_to_remove
        }
        for cid in chunk_ids:
            chunks_data.pop(cid)
        save_json(chunks_path, chunks_data)
        stats["chunks_removed"] = len(chunk_ids)

        # ── 2. vdb_chunks (matrix-aware) ──────────────────────────────────────
        def _remove_vdb_entries(vdb_path, ids_to_remove):
            """Remove entries from a NanoVectorDB file, updating both data and matrix."""
            import numpy as np
            import base64
            d = load_json(vdb_path)
            if not d:
                return 0
            old_data = d.get("data", [])
            old_ids = [e.get("__id__") for e in old_data]
            keep_mask = [eid not in ids_to_remove for eid in old_ids]
            new_data = [e for e, keep in zip(old_data, keep_mask) if keep]
            removed = len(old_data) - len(new_data)
            if removed == 0:
                return 0
            d["data"] = new_data
            # Rebuild matrix by slicing rows
            emb_dim = d.get("embedding_dim", 3072)
            mat_b64 = d.get("matrix", "")
            if mat_b64:
                try:
                    raw = base64.b64decode(mat_b64)
                    n_total = len(raw) // (emb_dim * 4)
                    if n_total == len(old_data):  # matrix is in sync with old data
                        mat = np.frombuffer(raw, dtype=np.float32).reshape(n_total, emb_dim)
                        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                        new_mat = mat[keep_indices]
                        d["matrix"] = base64.b64encode(new_mat.tobytes()).decode("ascii")
                    else:
                        print(f"[WARN] {vdb_path.name}: matrix rows ({n_total}) != data len ({len(old_data)}), skipping matrix update")
                except Exception as ex:
                    print(f"[WARN] {vdb_path.name}: matrix update failed: {ex}")
            save_json(vdb_path, d)
            return removed

        p = storage_dir / "vdb_chunks.json"
        stats["vdb_chunks_removed"] = _remove_vdb_entries(p, chunk_ids)

        # ── 3. Entities ───────────────────────────────────────────────────────
        p = storage_dir / "kv_store_full_entities.json"
        d = load_json(p) or {}
        removed_ent = set()
        for eid, e in list(d.items()):
            if not isinstance(e, dict):
                continue
            srcs = set(e.get("source_id", "").split("<SEP>"))
            remaining = srcs - chunk_ids
            if not remaining or remaining == {""}:
                d.pop(eid)
                removed_ent.add(eid)
                stats["entities_removed"] += 1
            elif srcs & chunk_ids:
                e["source_id"] = "<SEP>".join(remaining)
                stats["entities_updated"] += 1
        save_json(p, d)

        # ── 4. Relations ──────────────────────────────────────────────────────
        p = storage_dir / "kv_store_full_relations.json"
        d = load_json(p) or {}
        removed_rel = set()
        for rid, r in list(d.items()):
            if not isinstance(r, dict):
                continue
            srcs = set(r.get("source_id", "").split("<SEP>"))
            remaining = srcs - chunk_ids
            if not remaining or remaining == {""}:
                d.pop(rid)
                removed_rel.add(rid)
                stats["relations_removed"] += 1
            elif srcs & chunk_ids:
                r["source_id"] = "<SEP>".join(remaining)
                stats["relations_updated"] += 1
        save_json(p, d)

        # ── 5. vdb_entities ───────────────────────────────────────────────────
        if removed_ent:
            p = storage_dir / "vdb_entities.json"
            stats["vdb_entities_removed"] = _remove_vdb_entries(p, removed_ent)

        # ── 6. vdb_relationships ──────────────────────────────────────────────
        if removed_rel:
            p = storage_dir / "vdb_relationships.json"
            stats["vdb_relations_removed"] = _remove_vdb_entries(p, removed_rel)

        # ── 7. GraphML ────────────────────────────────────────────────────────
        gp = storage_dir / "graph_chunk_entity_relation.graphml"
        if gp.exists():
            try:
                ET.register_namespace("", "http://graphml.graphdrawing.org/graphml")
                tree = ET.parse(str(gp))
                root = tree.getroot()
                # Find graph element regardless of namespace
                graph = None
                for child in root:
                    if child.tag.endswith("}graph") or child.tag == "graph":
                        graph = child
                        break
                if graph is not None:
                    bad_ids = chunk_ids | removed_ent | removed_rel
                    nodes = [n for n in list(graph) if n.tag.endswith("}node") or n.tag == "node"
                             if n.get("id", "") in bad_ids]
                    edges = [e for e in list(graph) if e.tag.endswith("}edge") or e.tag == "edge"
                             if e.get("source", "") in bad_ids or e.get("target", "") in bad_ids]
                    for n in nodes:
                        graph.remove(n)
                    for e in edges:
                        graph.remove(e)
                    tree.write(str(gp), encoding="utf-8", xml_declaration=True)
                    stats["graph_nodes_removed"] = len(nodes)
                    stats["graph_edges_removed"] = len(edges)
            except Exception as ex:
                print(f"[WARN] GraphML update failed: {ex}")

        # ── 8. processed_files.json ───────────────────────────────────────────
        pf = load_json(processed_path)
        if pf and isinstance(pf, dict):
            before = len(pf.get("files", []))
            pf["files"] = [f for f in pf.get("files", []) if f.get("doc_id") != doc_id]
            if len(pf["files"]) < before:
                save_json(processed_path, pf)

        # ── Reload RAG instance để đồng bộ in-memory state ───────────────────
        # Reset rag instance để lần query tiếp theo sẽ load lại từ storage đã sạch
        self.rag = None

        return {
            "doc_id": doc_id,
            "status": "deleted",
            "full_doc_ids": list(full_doc_ids_to_remove),
            "stats": stats,
        }

