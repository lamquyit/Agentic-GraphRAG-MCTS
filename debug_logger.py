"""
Debug Logger Module
───────────────────
Ghi log debug chi tiết vào file `logs/debug_rerank.log` và `logs/debug_eval.log`
để dễ dàng kiểm tra sau khi chạy benchmark.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _make_logger(name: str, filename: str, level=logging.DEBUG) -> logging.Logger:
    """Tạo logger ghi ra file với format chi tiết."""
    logger = logging.getLogger(name)
    # Tránh duplicate handlers nếu module được import lại
    if logger.handlers:
        return logger
    logger.setLevel(level)
    
    fh = logging.FileHandler(LOG_DIR / filename, mode="a", encoding="utf-8")
    fh.setLevel(level)
    
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Cũng log ra console cho WARNING trở lên
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


# ── Pre-built loggers ────────────────────────────────────────────────────────

rerank_logger = _make_logger("debug.rerank", "debug_rerank.log")
eval_logger   = _make_logger("debug.eval",   "debug_eval.log")
provider_logger = _make_logger("debug.provider", "debug_provider.log")


def log_rerank_call(query: str, num_docs: int, top_n, provider_name: str, model_name: str):
    """Log khi bắt đầu gọi rerank."""
    rerank_logger.info(
        f"RERANK_CALL | provider={provider_name} model={model_name} "
        f"num_docs={num_docs} top_n={top_n} query={query[:120]}..."
    )


def log_rerank_response(raw_response: str, scored_docs: list, elapsed: float):
    """Log kết quả rerank (raw JSON + scored docs)."""
    rerank_logger.info(f"RERANK_RESPONSE | elapsed={elapsed:.2f}s scored_count={len(scored_docs)}")
    rerank_logger.debug(f"RERANK_RAW_RESPONSE | {raw_response[:500]}")
    for doc in scored_docs[:10]:  # Log top 10
        rerank_logger.debug(
            f"  idx={doc['index']} score={doc['relevance_score']:.4f}"
        )


def log_rerank_error(error: Exception, provider_name: str):
    """Log lỗi rerank."""
    rerank_logger.error(f"RERANK_ERROR | provider={provider_name} error={error}")


def log_eval_item(question: str, ground_truth: str, answer: str, em: float, f1: float, latency: float, scenario: str, error=None):
    """Log từng item trong quá trình eval."""
    status = "OK" if not error else f"ERROR: {error}"
    eval_logger.info(
        f"EVAL | scenario={scenario} em={em:.2f} f1={f1:.4f} "
        f"latency={latency:.2f}s status={status}"
    )
    eval_logger.debug(f"  Q: {question[:150]}")
    eval_logger.debug(f"  GT: {str(ground_truth)[:150]}")
    eval_logger.debug(f"  A: {str(answer)[:150]}")


def log_provider_call(provider_name: str, model_name: str, model_type: str, success: bool, error=None):
    """Log mỗi lần gọi LLM provider."""
    if success:
        provider_logger.debug(
            f"PROVIDER_OK | provider={provider_name} model={model_name} type={model_type}"
        )
    else:
        provider_logger.warning(
            f"PROVIDER_FAIL | provider={provider_name} model={model_name} "
            f"type={model_type} error={error}"
        )


def log_session_start(scenario: str, dataset: str, num_samples: int):
    """Log bắt đầu một session eval."""
    separator = "=" * 80
    eval_logger.info(separator)
    eval_logger.info(
        f"SESSION_START | scenario={scenario} dataset={dataset} "
        f"samples={num_samples} time={datetime.now().isoformat()}"
    )
    eval_logger.info(separator)
    rerank_logger.info(separator)
    rerank_logger.info(f"SESSION_START | scenario={scenario} dataset={dataset} time={datetime.now().isoformat()}")
    rerank_logger.info(separator)
