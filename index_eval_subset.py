import json
import os
import argparse
import time
from pathlib import Path
from rag_bridge import get_rag_service, process_document, settings, load_processed_metadata

print(">>> index_eval_subset.py is INITIALIZING...")

def index_mmlongbench(data_path, dataset_root):
    print(f"RAG_BASE_PATH from settings: {settings.RAG_BASE_PATH}")
    storage_name = "rag_storage_mmlongbench"
    print(f"Index will be saved to: {storage_name}")
    
    # Load already processed files - only consider SUCCESSFULLY indexed files (chunks > 0)
    processed = load_processed_metadata().get("files", [])
    
    # Chỉ skip nếu file đã được index THỰC SỰ (chunks > 0)
    successfully_indexed_paths = {
        f.get("path")
        for f in processed
        if f.get("path") and f.get("details", {}).get("rag_stats", {}).get("chunks", 0) > 0
    }
    # Theo dõi tất cả paths đã được ghi vào (dù không thành công) để log
    all_processed_paths = {f.get("path") for f in processed if f.get("path")}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from {data_path}")
    
    print(f"[Index] Successfully indexed paths (chunks>0): {len(successfully_indexed_paths)}")
    print(f"[Index] Total records in processed_files.json: {len(all_processed_paths)}")
    
    indexed_files = set()
    skipped_files = set()
    
    # MMLongBench-Doc data has "doc_id" field which is the filename
    for item in data:
        doc_id = item.get("doc_id")
        if doc_id:
            # Construct absolute path to the PDF
            # Documents are in MMLongBench-Doc/data/documents
            abs_path = os.path.join(dataset_root, "data", "documents", doc_id)
            
            # Chỉ skip nếu đã được index thực sự (chunks > 0)
            if abs_path in successfully_indexed_paths:
                if abs_path not in skipped_files:
                    print(f"Skipping already indexed MMLongBench document: {abs_path}")
                    skipped_files.add(abs_path)
                continue
            
            if os.path.exists(abs_path) and abs_path not in indexed_files:
                if abs_path in all_processed_paths:
                    print(f"Re-indexing failed MMLongBench document (chunks=0): {abs_path}")
                else:
                    print(f"Indexing MMLongBench document: {abs_path}")
                try:
                    process_document(abs_path, storage_name="rag_storage_mmlongbench")
                    indexed_files.add(abs_path)
                except Exception as e:
                    print(f"Error indexing {abs_path}: {e}")
            elif not os.path.exists(abs_path):
                print(f"Warning: File not found: {abs_path}")

    print(f"Indexed {len(indexed_files)} new/re-indexed documents for MMLongBench-Doc")
    print(f"Skipped {len(skipped_files)} already indexed documents")

def index_hotpot(data_path, base_dir):
    # Load already processed files - only consider SUCCESSFULLY indexed files (chunks > 0)
    processed = load_processed_metadata().get("files", [])
    
    # Chỉ skip nếu file đã được index THỰC SỰ (chunks > 0)
    successfully_indexed_paths = {
        f.get("path")
        for f in processed
        if f.get("path") and f.get("details", {}).get("rag_stats", {}).get("chunks", 0) > 0
    }
    # Cũng theo dõi tất cả paths đã được ghi vào (dù không thành công) để cập nhật
    all_processed_paths = {f.get("path") for f in processed if f.get("path")}
    
    print(f"[Index] Successfully indexed paths (chunks>0): {len(successfully_indexed_paths)}")
    print(f"[Index] Total records in processed_files.json: {len(all_processed_paths)}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from {data_path}")
    
    # For HotpotQA (distractor), the context is provided as a list of [title, sentences]
    # We'll save each context as a temporary text file and index it.
    temp_dir = Path(base_dir) / "temp_hotpot_docs"
    temp_dir.mkdir(exist_ok=True)
    
    indexed_titles = set()
    skipped_titles = set()
    
    for item in data:
        context = item.get("context", [])
        for title, sentences in context:
            if title not in indexed_titles and title not in skipped_titles:
                file_path = temp_dir / f"{title.replace('/', '_')}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(" ".join(sentences))
                
                abs_p = str(file_path.absolute())
                
                # Chỉ skip nếu đã được index thực sự (chunks > 0)
                if abs_p in successfully_indexed_paths:
                    print(f"Skipping already indexed HotpotQA context: {title}")
                    skipped_titles.add(title)
                    continue
                
                if abs_p in all_processed_paths:
                    print(f"Re-indexing failed context (chunks=0): {title}")
                else:
                    print(f"Indexing HotpotQA context: {title}")
                    
                try:
                    process_document(str(file_path.absolute()), storage_name="rag_storage_hotpot")
                    indexed_titles.add(title)
                except Exception as e:
                    print(f"Error indexing {title}: {e}")

    print(f"Indexed {len(indexed_titles)} new/re-indexed contexts for HotpotQA")
    print(f"Skipped {len(skipped_titles)} already indexed contexts")

def index_asqa(data_path, base_dir):
    """Index ASQA dataset — mirrors HotpotQA approach.
    
    Uses a 2-phase approach:
      Phase 1: Aggregate ALL passages per wikipage into .txt files
      Phase 2: Index the complete .txt files into rag_storage_asqa
    """
    storage_name = "rag_storage_asqa"
    print(f"\n{'='*60}")
    print(f"Starting ASQA indexing → storage: {storage_name}")
    print(f"{'='*60}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} ASQA samples from {data_path}")
    
    # Create temp dir for ASQA docs (like HotpotQA)
    temp_dir = Path(base_dir) / "temp_asqa_docs"
    temp_dir.mkdir(exist_ok=True)
    
    # ── Phase 1: Aggregate all passages per wikipage into .txt files ──
    print(f"\n--- Phase 1: Aggregating passages into .txt files ---")
    wikipage_passages = {}  # wikipage_title -> list of passages
    
    for item in data:
        annotations = item.get("annotations", [])
        for ann in annotations:
            knowledge_list = ann.get("knowledge", [])
            for knowledge in knowledge_list:
                content = (knowledge.get("content") or "").strip()
                wikipage = (knowledge.get("wikipage") or "").strip()
                
                if not content or not wikipage:
                    continue
                
                if wikipage not in wikipage_passages:
                    wikipage_passages[wikipage] = []
                # Avoid duplicate passages for the same wikipage
                if content not in wikipage_passages[wikipage]:
                    wikipage_passages[wikipage].append(content)
    
    # Write aggregated files
    files_to_index = {}  # wikipage_title -> abs_path
    for wikipage, passages in wikipage_passages.items():
        safe_name = wikipage.replace('/', '_').replace('\\', '_')
        file_path = temp_dir / f"{safe_name}.txt"
        
        # Always overwrite to ensure complete content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(passages))
        
        files_to_index[wikipage] = str(file_path.absolute())
    
    print(f"Aggregated {sum(len(p) for p in wikipage_passages.values())} passages "
          f"into {len(files_to_index)} unique wikipage files")
    
    # ── Phase 2: Index all .txt files ──
    print(f"\n--- Phase 2: Indexing {len(files_to_index)} files into {storage_name} ---")
    
    # Load already processed files
    processed = load_processed_metadata().get("files", [])
    successfully_indexed_paths = {
        f.get("path")
        for f in processed
        if f.get("path") and f.get("details", {}).get("rag_stats", {}).get("chunks", 0) > 0
    }
    all_processed_paths = {f.get("path") for f in processed if f.get("path")}
    
    print(f"[Index] Successfully indexed paths (chunks>0): {len(successfully_indexed_paths)}")
    print(f"[Index] Total records in processed_files.json: {len(all_processed_paths)}")
    
    indexed_count = 0
    skipped_count = 0
    error_count = 0
    
    for wikipage, abs_p in files_to_index.items():
        # Skip if already indexed successfully
        if abs_p in successfully_indexed_paths:
            print(f"Skipping already indexed: {wikipage}")
            skipped_count += 1
            continue
        
        if abs_p in all_processed_paths:
            print(f"Re-indexing (chunks=0): {wikipage}")
        else:
            print(f"Indexing: {wikipage}")
        
        try:
            process_document(abs_p, storage_name=storage_name)
            indexed_count += 1
        except Exception as e:
            print(f"Error indexing {wikipage}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"ASQA Indexing Complete!")
    print(f"  Indexed: {indexed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors:  {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["hotpot", "mmlongbench", "asqa", "both", "all"], default="both")
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Create logs directory
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    session_log = f"status_indexing_{args.dataset}_{int(time.time())}.log"
    log_path = str(logs_dir / session_log)
    print(f"Indexing session started. Logs will be saved to: {log_path}")

    # ── Tee stdout/stderr vào log file (vừa hiện terminal vừa lưu file) ──────
    import sys as _sys

    class _Tee:
        def __init__(self, stream, fh):
            self._s, self._f = stream, fh
        def write(self, data):
            self._s.write(data)
            self._f.write(data)
            self._f.flush()
        def flush(self):
            self._s.flush()
            self._f.flush()
        def __getattr__(self, attr):
            return getattr(self._s, attr)

    _log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    _sys.stdout = _Tee(_sys.__stdout__, _log_fh)
    _sys.stderr = _Tee(_sys.__stderr__, _log_fh)
    print(f">>> All output is being tee'd to: {log_path}")

    if args.dataset in ["mmlongbench", "both"]:
        # Use the file the user just edited
        mml_data = BASE_DIR / "MMLongBench-Doc" / "data" / "samples_15_docs.json"
        mml_root = BASE_DIR / "MMLongBench-Doc"
        if mml_data.exists():
            index_mmlongbench(str(mml_data), str(mml_root))
        else:
            print(f"MMLongBench data not found at {mml_data}")
            
    if args.dataset in ["hotpot", "both"]:
        hotpot_data = BASE_DIR / "data" / "hotpot" / "hotpot_dev_distractor_20.json"
        if hotpot_data.exists():
            print(f"Starting HotpotQA indexing...")
            index_hotpot(str(hotpot_data), str(BASE_DIR))
        else:
            print(f"HotpotQA data not found at {hotpot_data}")

    if args.dataset in ["asqa", "all"]:
        asqa_data = BASE_DIR / "data" / "asqa" / "asqa_dev_50.json"
        if asqa_data.exists():
            print(f"Starting ASQA indexing...")
            index_asqa(str(asqa_data), str(BASE_DIR))
        else:
            print(f"ASQA data not found at {asqa_data}")

    print("Indexing process FINISHED successfully.")
