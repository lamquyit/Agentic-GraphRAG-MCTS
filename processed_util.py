import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import hashlib

ROOT = Path(__file__).parent
PROCESSED_JSON = ROOT / "processed_files.json"
INPUT_FILE = ROOT / "input"

def calculate_md5(file_path: str) -> str:
    """Calculates MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_metadata():
    if not PROCESSED_JSON.exists():
        return {"files": []}
    with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"files": []}


def save_processed_metadata(data):
    PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_processed_file(file_path: str, elapsed: float, metrics: Optional[Dict] = None):
    data = load_processed_metadata()
    path_obj = Path(file_path)
    
    # Calculate file size in MB
    try:
        size_mb = round(path_obj.stat().st_size / (1024 * 1024), 2)
    except FileNotFoundError:
        size_mb = 0
        
    # Calculate MD5
    try:
        doc_id = calculate_md5(file_path)
    except FileNotFoundError:
        doc_id = ""

    record = {
        "filename": path_obj.name,
        "path": str(path_obj.resolve()),
        "doc_id": doc_id,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "processing_time_sec": round(elapsed, 2),
        "file_size_mb": size_mb,
        "details": metrics or {}
    }

    # Avoid duplicates: remove old entry for same path
    data["files"] = [
        f for f in data["files"]
        if f.get("path") != record["path"]
    ]

    data["files"].append(record)
    save_processed_metadata(data)


def generate_summary_report() -> Dict:
    """
    Generates a summary report from processed files.
    Returns a dictionary with statistics.
    """
    data = load_processed_metadata()
    files = data.get("files", [])
    
    if not files:
        return {}
    
    total_files = len(files)
    total_time = sum(f.get("processing_time_sec", 0) for f in files)
    total_size_mb = sum(f.get("file_size_mb", 0) for f in files)
    
    # Calculate Total Chunks (RAG Chunks)
    total_chunks = 0
    for f in files:
        rag_stats = f.get("details", {}).get("rag_stats", {})
        total_chunks += rag_stats.get("chunks", 0)
    
    # Calculate averages
    avg_time_per_file = total_time / total_files if total_files > 0 else 0
    
    # Speed: Seconds per Chunk
    # Avoid division by zero
    avg_speed_sec_per_chunk = total_time / total_chunks if total_chunks > 0 else 0
    
    # Phase Breakdown (Average)
    phase_stats = {
        "mineru_parsing": 0.0,
        "content_analysis": 0.0,
        "multimodal_processing": 0.0,
        "text_indexing": 0.0,
        "text_indexing": 0.0
    }
    
    count_valid_details = 0
    for f in files:
        details = f.get("details", {})
        if not details:
            continue
            
        # Extract times (safely)
        mineru = details.get("mineru_parsing_time") or {}
        analysis = details.get("content_analysis_time") or {}
        multimodal = details.get("multimodal_processing_time") or {}
        text_index = details.get("text_indexing_time") or {}
        
        t_mineru = mineru.get("duration_sec", 0)
        t_analysis = analysis.get("duration_sec", 0)
        t_multimodal = multimodal.get("duration_sec", 0)
        t_text = text_index.get("duration_sec", 0)
        
        # New: Cross-modal Graph Merging
        # It is usually a sub-component of Multimodal Processing time if captured sequentially,
        # but if we want to break it out, we should handle it carefully.
        # In processor.py, multimodal_processing_time is t5-t4, covering everything.
        # crossmodal_graph_merging_time is inside that duration.
        # So we should validly show it as "of which graph merging: X sec" or separate it if logic changed.
        # For now, let's just extract it for display.
        
        t_cross_merge = (details.get("crossmodal_graph_merging_time") or {}).get("duration_sec", 0)
        t_desc = (details.get("multimodal_description_time") or {}).get("duration_sec", 0)
        t_embed = (details.get("multimodal_embedding_time") or {}).get("duration_sec", 0)
        t_extract = (details.get("multimodal_extraction_time") or {}).get("duration_sec", 0)

        phase_stats["mineru_parsing"] += t_mineru
        phase_stats["content_analysis"] += t_analysis
        phase_stats["multimodal_processing"] += t_multimodal
        phase_stats["text_indexing"] += t_text
        phase_stats["crossmodal_graph_merging"] = phase_stats.get("crossmodal_graph_merging", 0.0) + t_cross_merge
        phase_stats["multimodal_description"] = phase_stats.get("multimodal_description", 0.0) + t_desc
        phase_stats["multimodal_embedding"] = phase_stats.get("multimodal_embedding", 0.0) + t_embed
        phase_stats["multimodal_extraction"] = phase_stats.get("multimodal_extraction", 0.0) + t_extract
        
        t_total = f.get("processing_time_sec", 0)
        # Remaining time is miscellaneous RAG overhead (merging graphs, etc.)
        # Note: t_multimodal already includes t_cross_merge in current processor logic?
        # Let's check: t4 is start of multimodal, t5 is end. 
        # t_merge is inside t4-t5. So t_multimodal INCLUDES t_cross_merge.
        # To avoid double counting in "Overhead" calculation, we rely on t_multimodal.
        

        
        count_valid_details += 1

    if count_valid_details > 0:
        for k in phase_stats:
            phase_stats[k] = round(phase_stats[k] / count_valid_details, 2)
            
    # Prepare latest file report data
    latest_file_data = {}
    if files:
        # Assuming the last file in the list is the latest
        last_f = files[-1]
        ld = last_f.get("details", {})
        
        t_min = (ld.get("mineru_parsing_time") or {}).get("duration_sec", 0)
        t_ana = (ld.get("content_analysis_time") or {}).get("duration_sec", 0)
        t_mul = (ld.get("multimodal_processing_time") or {}).get("duration_sec", 0)
        t_txt = (ld.get("text_indexing_time") or {}).get("duration_sec", 0)
        t_cro = (ld.get("crossmodal_graph_merging_time") or {}).get("duration_sec", 0)
        t_desc = (ld.get("multimodal_description_time") or {}).get("duration_sec", 0)
        t_embed = (ld.get("multimodal_embedding_time") or {}).get("duration_sec", 0)
        t_extract = (ld.get("multimodal_extraction_time") or {}).get("duration_sec", 0)
        
        t_tot = last_f.get("processing_time_sec", 0)

        
        latest_file_data = {
            "filename": last_f.get("filename"),
            "processing_time_sec": t_tot,
            "phase_stats": {
                "mineru_parsing": t_min,
                "content_analysis": t_ana,
                "multimodal_processing": t_mul,
                "text_indexing": t_txt,
                "crossmodal_graph_merging": t_cro,
                "multimodal_description": t_desc,
                "multimodal_embedding": t_embed,
                "multimodal_extraction": t_extract,
                "multimodal_extraction": t_extract
            }
        }
            
            
    return {
        "latest_file": latest_file_data,
        "total_files": total_files,
        "total_processed_size_mb": round(total_size_mb, 2),
        "total_chunks": total_chunks,
        "avg_time_per_file_sec": round(avg_time_per_file, 2),
        "avg_speed_sec_per_chunk": round(avg_speed_sec_per_chunk, 4),
        "avg_phase_breakdown_sec": phase_stats
    }


def format_single_file_report(file_data: Dict) -> str:
    """Formats stats for a single file into a report string."""
    if not file_data:
        return ""
        
    phases = file_data.get("phase_stats", {})
    return f"""
[LATEST PROCESSED FILE: {file_data.get('filename')}]
Total Time: {file_data.get('processing_time_sec')} sec
- MinerU Parsing:        {phases.get('mineru_parsing', 0)} sec
- Content Analysis:      {phases.get('content_analysis', 0)} sec
- Text Indexing:         {phases.get('text_indexing', 0)} sec
- Multimodal Processing: {phases.get('multimodal_processing', 0)} sec
  * Description Gen:     {phases.get('multimodal_description', 0)} sec
  * Embedding & Store:   {phases.get('multimodal_embedding', 0)} sec
  * Entity Extraction:   {phases.get('multimodal_extraction', 0)} sec
  * Graph Merging:       {phases.get('crossmodal_graph_merging', 0)} sec
"""


def get_report_text() -> str:
    """Returns a formatted string for the summary report."""
    summary = generate_summary_report()
    if not summary:
        return "No data available."
        
    phases = summary.get("avg_phase_breakdown_sec", {})
    
    
    # Get latest file stats
    latest_file_text = ""
    if summary.get("latest_file"):
        latest_file_text = format_single_file_report(summary["latest_file"])

    text = f"""==========================================
RAG PROCESSING REPORT
Generated at: {datetime.now().isoformat()}
==========================================
{latest_file_text}
"""
# [CUMULATIVE OVERVIEW]
# Total Files Processed: {summary['total_files']}
# Total Size: {summary['total_processed_size_mb']} MB
# Total Chunks Generated: {summary['total_chunks']}

# [PERFORMANCE ESTIMATION (AVERAGE)]
# Average Time per File: {summary['avg_time_per_file_sec']} seconds
# Average Processing Speed: {summary['avg_speed_sec_per_chunk']} seconds/chunk

# [AVERAGE PHASE BREAKDOWN]
# 1. MinerU Parsing:        {phases.get('mineru_parsing', 0)} sec
# 2. Content Analysis:      {phases.get('content_analysis', 0)} sec
# 3. Text Indexing:         {phases.get('text_indexing', 0)} sec
# 4. Multimodal Processing: {phases.get('multimodal_processing', 0)} sec
# 4. Multimodal Processing: {phases.get('multimodal_processing', 0)} sec

# ==========================================
# """
    # Save to file as requested
    report_path = ROOT / "report_summary.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving report summary to file: {e}")
        
    return text
def load_input_files():
    result = {"files": []}

    if not INPUT_FILE.exists():
        return result

    for item in INPUT_FILE.iterdir():
        if item.is_file():
            result["files"].append({
                "filename": item.name
            })

    return result

def update_processed_metadata():
    """
    Xóa các record trong PROCESSED_JSON
    nếu filename không tồn tại trong load_output_for_report()
    """
    if not PROCESSED_JSON.exists():
        return

    # 1. Load processed metadata
    with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
        processed_data = json.load(f)

    processed_files = processed_data.get("files", [])

    # 2. Load danh sách file hiện có
    output_data = load_input_files()
    print("Output for report files:", output_data.get("files", []))
    valid_filenames = {
        f["filename"] for f in output_data.get("files", [])
    }

    # 3. Lọc lại processed files
    filtered_files = [
        f for f in processed_files
        if f.get("filename") in valid_filenames
    ]

    # 4. Ghi lại file JSON
    processed_data["files"] = filtered_files

    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)