"""All settings come from the environment (or .env), so one image runs dev, CI and prod."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _keys(raw):
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: str = field(default_factory=lambda: os.getenv("RAG_DATA_DIR", "indexes/rag_store"))
    embed_model: str = field(default_factory=lambda: os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001"))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("RAG_EMBED_DIM", "768")))
    chat_model: str = field(default_factory=lambda: os.getenv("RAG_CHAT_MODEL", "gemini-3.5-flash-lite"))
    index_factory: str = field(default_factory=lambda: os.getenv("RAG_INDEX_FACTORY", "Flat"))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    chunk_chars: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_CHARS", "1000")))
    bm25_weight: float = field(default_factory=lambda: float(os.getenv("RAG_BM25_WEIGHT", "0.25")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "150")))
    # comma-separated; empty = auth off (local dev only, logged loudly at startup)
    api_keys: frozenset = field(default_factory=lambda: _keys(os.getenv("RAG_API_KEYS", "")))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RAG_RATE_LIMIT", "60")))
    max_question_chars: int = 1000
    max_document_chars: int = 2_000_000
