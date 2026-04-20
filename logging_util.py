# logging_util.py
import logging
import re
from io import StringIO
from typing import Dict, Tuple, Optional
from datetime import datetime


# =====================================================
# 1. Setup in-memory logger (BẮT BUỘC có timestamp)
# =====================================================

def setup_memory_logger(
    level=logging.INFO,
    fmt="%(asctime)s | %(levelname)s | %(message)s",) -> Tuple[StringIO, logging.Handler]:
    """
    Ghi toàn bộ log vào bộ nhớ để parse sau khi pipeline chạy xong
    """
    log_stream = StringIO()

    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

    return log_stream, handler


# =====================================================
# 2. Get raw logs
# =====================================================

def get_logs(log_stream: StringIO) -> str:
    return log_stream.getvalue()


# =====================================================
# 3. Parse content block information
# =====================================================

def parse_content_blocks(log_text: str) -> Optional[Dict]:
    """
    Parse:
    - Total content blocks
    - Content block types
    """

    total_blocks_pattern = (
        r"Total blocks in content_list:\s*(\d+)"
    )

    block_types_pattern = (
        r"-\s*(\w+):\s*(\d+)"
    )

    total_match = re.search(total_blocks_pattern, log_text)
    if not total_match:
        return None

    total_blocks = int(total_match.group(1))

    block_types = {}
    for m in re.finditer(block_types_pattern, log_text):
        block_types[m.group(1)] = int(m.group(2))

    return {
        "total_blocks": total_blocks,
        "block_types": block_types,
    }


# =====================================================
# 4. Parse MinerU parsing time (Stage 5 → 6)
# =====================================================

def parse_mineru_parsing_time(log_text: str) -> Optional[Dict]:
    """
    Thời gian MinerU xử lý OCR / layout:
    từ warning mineru đầu tiên → parsing complete
    """

    time_fmt = "%Y-%m-%d %H:%M:%S.%f"

    start_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+).*mineru"
    )

    end_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+).*Parsing .* complete!"
    )

    m_start = re.search(start_pattern, log_text)
    m_end = re.search(end_pattern, log_text)

    if not m_start or not m_end:
        return None

    t_start_str = m_start.group(1).replace(",", ".")
    t_end_str = m_end.group(1).replace(",", ".")

    t_start = datetime.strptime(t_start_str, time_fmt)
    t_end = datetime.strptime(t_end_str, time_fmt)

    return {
        "stage": "mineru_parsing",
        "duration_sec": round((t_end - t_start).total_seconds(), 3),
    }


# =====================================================
# 5. Parse Content Analysis time (Stage 7 → 8)
# =====================================================

def parse_content_analysis_time(log_text: str) -> Optional[Dict]:
    """
    Thời gian từ:
    - Content separation complete
    → Document processing complete
    """

    time_fmt = "%Y-%m-%d %H:%M:%S.%f"

    stage7_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+).*Content separation complete"
    )

    stage8_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+).*Document .* processing complete"
    )

    m7 = re.search(stage7_pattern, log_text)
    m8 = re.search(stage8_pattern, log_text)

    if not m7 or not m8:
        return None

    t7_str = m7.group(1).replace(",", ".")
    t8_str = m8.group(1).replace(",", ".")

    t7 = datetime.strptime(t7_str, time_fmt)
    t8 = datetime.strptime(t8_str, time_fmt)

    return {
        "stage": "content_analysis",
        "duration_sec": round((t8 - t7).total_seconds(), 3),
    }


# =====================================================
# 6. Parse RAG Statistics (Chunks, Entities, Relations)
# =====================================================

def parse_rag_stats(log_text: str) -> Dict:
    """
    Parse RAG Knowledge Graph stats
    Example log: "Chunk 1 of 5 extracted 10 Ent + 5 Rel"
    """
    chunk_pattern = r"(?:Chunk|Process) \d+ of (\d+)"
    ent_pattern = r"extracted (\d+) Ent"
    rel_pattern = r"\+ (\d+) Rel"

    chunks = 0
    entities = 0
    relations = 0

    # Tìm số chunk cao nhất (vì log ghi Chunk 1 of N...)
    for m in re.finditer(chunk_pattern, log_text):
        c = int(m.group(1))
        if c > chunks:
            chunks = c
    
    # Cộng dồn Entities & Relations
    for m in re.finditer(ent_pattern, log_text):
        entities += int(m.group(1))

    for m in re.finditer(rel_pattern, log_text):
        relations += int(m.group(1))

    return {
        "chunks": chunks,
        "entities": entities,
        "relations": relations
    }


# =====================================================
# 6. Parse ALL metrics (HÀM TỔNG)
# =====================================================

def parse_document_metrics(log_text: str) -> Dict:
    """
    Hàm tổng – gọi 1 lần sau process_document_complete
    """

    return {
        "content_blocks": parse_content_blocks(log_text),
        "mineru_parsing_time": parse_mineru_parsing_time(log_text),
        "content_analysis_time": parse_content_analysis_time(log_text),
        "rag_stats": parse_rag_stats(log_text),
    }


# =====================================================
# 7. Cleanup logger
# =====================================================

def cleanup_logger(handler: logging.Handler):
    logging.getLogger().removeHandler(handler)
