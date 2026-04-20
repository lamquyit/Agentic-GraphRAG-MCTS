"""
LLM Provider Fallback Module
──────────────────────────────
Mặc định gọi OpenRouter. Nếu lỗi → tự động fallback sang OpenAI.
Không gọi API thừa để check — chỉ fallback khi request thực sự thất bại.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Global usage tracker
_usage_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

def get_usage_stats():
    return _usage_stats

def reset_usage_stats():
    global _usage_stats
    _usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ── Provider Configuration ───────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Cấu hình cho một LLM provider."""
    name: str
    api_key: str
    base_url: Optional[str]
    llm_model: str           # model cho text completion
    vision_model: str        # model cho vision tasks
    embedding_model: str     # model cho embeddings
    rerank_model: str        # model cho reranking (LLM based)
    embedding_dim: int = 3072


# Model mapping: OpenRouter yêu cầu prefix "openai/" trước tên model
OPENROUTER_MODELS = {
    "llm": "openai/gpt-4o",
    "vision": "openai/gpt-4o",
    "embedding": "openai/text-embedding-3-large",
    "rerank": "qwen/qwen-2.5-7b-instruct",
}

OPENAI_MODELS = {
    "llm": "gpt-4o",
    "vision": "gpt-4o",
    "embedding": "text-embedding-3-large",
    "rerank": "gpt-4o-mini",
}


def get_providers() -> list[ProviderConfig]:
    """
    Trả về danh sách providers theo thứ tự ưu tiên:
      1. OpenRouter (nếu có OPENROUTER_API_KEY)
      2. OpenAI    (nếu có OPENAI_API_KEY)
    """
    providers = []

    openrouter_key = os.getenv("OPENAI_API_KEY_opr")
    if openrouter_key:
        providers.append(ProviderConfig(
            name="openrouter",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            llm_model=OPENROUTER_MODELS["llm"],
            vision_model=OPENROUTER_MODELS["vision"],
            embedding_model=OPENROUTER_MODELS["embedding"],
            rerank_model=OPENROUTER_MODELS["rerank"],
        ))

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers.append(ProviderConfig(
            name="openai",
            api_key=openai_key,
            base_url=None,
            llm_model=OPENAI_MODELS["llm"],
            vision_model=OPENAI_MODELS["vision"],
            embedding_model=OPENAI_MODELS["embedding"],
            rerank_model=OPENAI_MODELS["rerank"],
        ))

    if not providers:
        raise RuntimeError(
            "Không tìm thấy API key nào! "
            "Hãy set OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env"
        )

    return providers


def get_primary_provider() -> ProviderConfig:
    """Trả về provider đầu tiên (ưu tiên cao nhất)."""
    return get_providers()[0]


# ── Fallback Wrapper cho LLM Completion ──────────────────────────────────────

async def openai_complete_with_fallback(
    model_type: str,   # "llm", "vision", hoặc "rerank"
    prompt: str,
    system_prompt=None,
    history_messages=None,
    **kwargs,
):
    """
    Gọi openai_complete_if_cache với fallback qua các providers.
    
    - model_type: "llm" → dùng llm_model, "vision" → dùng vision_model, "rerank" → dùng rerank_model
    - Thử provider đầu tiên, nếu lỗi → thử provider tiếp theo
    - KHÔNG gọi API thừa để check — chỉ fallback khi request thất bại
    """
    from lightrag.llm.openai import openai_complete_if_cache

    providers = get_providers()
    history_messages = history_messages or []
    last_error = None

    for provider in providers:
        if model_type == "llm":
            model_name = provider.llm_model
        elif model_type == "vision":
            model_name = provider.vision_model
        elif model_type == "rerank":
            model_name = provider.rerank_model
        else:
            model_name = provider.llm_model
        try:
            result = await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=provider.api_key,
                base_url=provider.base_url,
                **kwargs,
            )
            
            # Attempt to track usage if returned (LightRAG might return usage in some versions/configs)
            # If result is just a string, we can't track exactly without a separate call or internal access.
            # But let's assume we might get a dict or we can estimate.
            # For now, we'll try to find usage in the response if it's there.
            if isinstance(result, dict) and "usage" in result:
                u = result["usage"]
                _usage_stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                _usage_stats["completion_tokens"] += u.get("completion_tokens", 0)
                _usage_stats["total_tokens"] += u.get("total_tokens", 0)
            
            return result
        except Exception as e:
            last_error = e
            print(f"⚠️ [{provider.name}] LLM call failed: {e}")
            if provider != providers[-1]:
                next_provider = providers[providers.index(provider) + 1]
                print(f"   ↳ Falling back to {next_provider.name}...")
                # Nếu model name khác giữa providers, cần loại bỏ model cũ khỏi kwargs
                # để tránh conflict
            continue

    raise RuntimeError(
        f"Tất cả providers đều thất bại. Lỗi cuối: {last_error}"
    )


# ── Fallback Wrapper cho Embedding ───────────────────────────────────────────

async def openai_embed_with_fallback(texts: list[str]) -> list:
    """
    Gọi openai_embed với fallback qua các providers.
    Tự động xử lý None/invalid values trong texts.
    """
    from lightrag.llm.openai import openai_embed

    if texts is None:
        return []

    # Sanitize inputs
    safe_texts = [str(t) if t is not None else "" for t in texts]
    if not safe_texts:
        return []

    providers = get_providers()
    last_error = None

    for provider in providers:
        try:
            result = await openai_embed(
                safe_texts,
                model=provider.embedding_model,
                api_key=provider.api_key,
                base_url=provider.base_url,
            )
            if result is None:
                print(f"⚠️ [{provider.name}] Embedding returned None")
                continue
            return result
        except Exception as e:
            last_error = e
            print(f"⚠️ [{provider.name}] Embedding failed: {e}")
            if provider != providers[-1]:
                next_provider = providers[providers.index(provider) + 1]
                print(f"   ↳ Falling back to {next_provider.name}...")
            continue

    # Nếu tất cả thất bại → trả zero vectors thay vì crash
    print(f"❌ All embedding providers failed. Returning zero vectors. Last error: {last_error}")
    dim = providers[0].embedding_dim if providers else 3072
    return [[0.0] * dim] * len(safe_texts)


# ── Rerank Implementation (LLM based) ────────────────────────────────────────

async def qwen_rerank(
    query: str,
    documents: list[str],
    top_n: int = None,
) -> list[dict]:
    """
    Sử dụng Qwen (qua OpenRouter) để rerank document bằng cách cho điểm từ 0-10.
    Interface tương thích với LightRAG: returns list of {'index': int, 'relevance_score': float}
    """
    from debug_logger import log_rerank_call, log_rerank_response, log_rerank_error
    import time as _time

    if not documents:
        return []
    
    # Giới hạn số lượng docs để tránh vượt quá context (ví dụ top 30)
    docs_to_rerank = documents[:50]
    
    prompt = f"Given a user query and a set of retrieved documents, please estimate the relevance score of each document to the query on a scale of 0 to 10. Output the results as a JSON list of objects, each containing 'index' and 'score'.\n\nQuery: {query}\n\nDocuments:\n"
    for i, doc in enumerate(docs_to_rerank):
        prompt += f"[{i}] {doc[:1000]}\n" # Truncate docs to save tokens
    
    system_prompt = "You are a highly accurate reranking assistant. You respond only with valid JSON."
    
    primary = get_primary_provider()
    log_rerank_call(query, len(docs_to_rerank), top_n, primary.name, primary.rerank_model)
    _t0 = _time.time()
    
    try:
        # Gọi providers tuần tự cho đến khi thành công
        rerank_kwargs = {}
        # Chỉ truyền response_format khi provider hỗ trợ (OpenAI native)
        if primary.name == "openai":
            rerank_kwargs["response_format"] = {"type": "json_object"}
        
        response = await openai_complete_with_fallback(
            model_type="rerank",
            prompt=prompt,
            system_prompt=system_prompt,
            **rerank_kwargs
        )
        
        # Parse JSON output
        import json
        import re
        
        # Clean response if LLM added markdown code blocks
        clean_response = response
        if "```json" in response:
            clean_response = re.search(r"```json\n(.*?)\n```", response, re.DOTALL).group(1)
        elif "```" in response:
            clean_response = re.search(r"```\n(.*?)\n```", response, re.DOTALL).group(1)
            
        data = json.loads(clean_response)
        
        # Expecting {"results": [{"index": 0, "score": 8.5}, ...]} or list directly
        results = data if isinstance(data, list) else data.get("results", [])
        
        # Format for LightRAG
        scored_docs = []
        for r in results:
            idx = int(r.get("index", 0))
            score = float(r.get("score", 0.0)) / 10.0 # Scale to 0-1
            scored_docs.append({"index": idx, "relevance_score": score})
            
        # Sắp xếp theo score giảm dần
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if top_n:
            scored_docs = scored_docs[:top_n]
        
        elapsed = _time.time() - _t0
        log_rerank_response(response, scored_docs, elapsed)
        print(f"✅ Reranked {len(scored_docs)} using Qwen-based LLM reranker")
        return scored_docs
        
    except Exception as e:
        log_rerank_error(e, primary.name)
        print(f"❌ Rerank failed: {e}. Returning original order.")
        # Fallback: trả về thứ tự ban đầu với điểm giảm dần
        return [{"index": i, "relevance_score": 0.9 - (i * 0.01)} for i in range(len(documents))][:top_n]

