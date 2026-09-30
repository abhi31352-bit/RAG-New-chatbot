"""Central configuration for the RAG chatbot.

Single source of truth for every tunable (architecture.md S3.2) and for the
GROQ API key. The key is read from .env only and must never be logged,
printed, or surfaced in the UI (architecture.md D3, NFR-4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Config:
    """Runtime configuration. Never holds the API key in a printable form."""

    # LLM
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "qwen/qwen3.8-27b"
    groq_fallback_model: str = "openai/gpt-oss-20b"

    # Paths
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    chroma_collection: str = "hdfc_faq"

    # Embedding (pinned per architecture.md D1)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Chunking (proposal; confirmed in docs/ChunkingStrategy.md)
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_min_chars: int = 120

    # Retrieval
    k_fetch: int = 10
    k_context: int = 10
    # Minimum cosine similarity for the TOP hit to count as a match. 0 disables
    # the check. Measured spread on this corpus: answerable questions score
    # 0.40-0.76, unrelated/gibberish input 0.05-0.20, so 0.25 separates them
    # cleanly. Turning "nothing here is about that" into an empty result set is
    # what lets the answer contract say "not found" instead of inventing one.
    min_score: float = 0.25
    # Relative floor: keep only chunks scoring >= ratio x the top hit. `min_score`
    # above is ABSOLUTE and gates only the top hit, answering "is anything in the
    # corpus about this at all?". This one trims the tail, answering "is this
    # chunk actually about the thing we matched?". Without it, k_context=10 fills
    # its extra slots with Holdings/Fund-management chunks at 0.21-0.46 that
    # clear the absolute floor but answer nothing -- and cost latency, since
    # prompt size is charged against the LLM provider's throughput quota.
    # Measured at 0.75: prompt chars 4898/4158/6340/7270 -> 1023/1029/1017/1219
    # on the four gate questions, with no change to any answer. Set 0 to
    # disable and fall back to plain top-k.
    score_floor_ratio: float = 0.75

    # Sub-directories derived from data_dir
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    chunks_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = _abs(self.data_dir)
        self.chroma_dir = _abs(self.chroma_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.chunks_dir = self.data_dir / "chunks"
        self.logs_dir = REPO_ROOT / "logs"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "") or ""

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.chunks_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def missing_keys(self) -> List[str]:
        """Env vars that still need a value before the app can run."""
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        return missing

    def safe_repr(self) -> str:
        return (
            f"Config(embed_model={self.embed_model!r}, embed_dim={self.embed_dim}, "
            f"chroma_collection={self.chroma_collection!r}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"k_fetch={self.k_fetch}, k_context={self.k_context}, "
            f"data_dir={str(self.data_dir)!r}, "
            f"groq_api_key={'set' if self.groq_api_key else 'NOT SET'})"
        )

    def __repr__(self) -> str:  # never leak the key
        return self.safe_repr()


def _abs(path: Path) -> Path:
    """Resolve relative paths against the repo root, not the cwd."""
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path)


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(
            groq_api_key=os.getenv("GROQ_API_KEY", "") or "",
            groq_model=_env_str("GROQ_MODEL", "qwen/qwen3.8-27b"),
            groq_fallback_model=_env_str(
                "GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"
            ),
            data_dir=_env_str("DATA_DIR", "./data"),
            chroma_dir=_env_str("CHROMA_DIR", "./data/chroma"),
            chroma_collection=_env_str("CHROMA_COLLECTION", "hdfc_faq"),
            embed_model=_env_str(
                "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embed_dim=_env_int("EMBED_DIM", 384),
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            chunk_min_chars=_env_int("CHUNK_MIN_CHARS", 120),
            k_fetch=_env_int("K_FETCH", 10),
            k_context=_env_int("K_CONTEXT", 10),
            min_score=_env_float("MIN_SCORE", 0.25),
            score_floor_ratio=_env_float("SCORE_FLOOR_RATIO", 0.75),
        )
        _CONFIG.ensure_dirs()
    return _CONFIG
