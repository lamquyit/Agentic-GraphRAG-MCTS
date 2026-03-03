"""
Centralized configuration for Agentic RAG
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Settings:
    """All configuration in one place."""

    # ── LLM ──────────────────────────────────────────────────
    LLM_API_KEY: str = os.getenv("OPENAI_API_KEY_opr", "")
    LLM_BASE_URL: str = os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ── RAG Base ─────────────────────────────────────────────
    RAG_BASE_PATH: Path = Path(os.getenv("RAG_BASE_PATH", str(Path(__file__).parent.parent / "RAG_base")))

    # ── Agentic Thresholds ───────────────────────────────────
    GROUNDEDNESS_THRESHOLD: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.7"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REGEN_TEMPERATURE: float = float(os.getenv("REGEN_TEMPERATURE", "0.0"))

    # ── Server ───────────────────────────────────────────────
    API_PORT: int = int(os.getenv("API_PORT", "8002"))


settings = Settings()
