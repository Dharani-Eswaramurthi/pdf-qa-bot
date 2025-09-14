import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment from common locations so `.env` works out-of-the-box
# Priority: project root .env > backend/.env > CWD .env; existing env wins
try:
    root_env = Path(__file__).resolve().parents[2] / ".env"
    backend_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=root_env, override=True)
    load_dotenv(dotenv_path=backend_env, override=True)
    load_dotenv(override=True)
except Exception:
    pass


class Settings:
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data"))
    STORAGE_DIR: str = os.getenv(
        "STORAGE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "storage")
    )
    DEFAULT_PDF_PATH: str = os.getenv("PDF_PATH", os.path.join(DATA_DIR, "draft-oasis-e1-manual-04-28-2024.pdf"))

    # Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: Optional[int] = None  # inferred at runtime

    # Reranker (optional)
    RERANK: bool = os.getenv("RERANK", "true").lower() in {"1", "true", "yes"}
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM (optional; Hugging Face Inference only)
    USE_LLM: bool = os.getenv("USE_LLM", "true").lower() in {"1", "true", "yes"}
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    HF_PROVIDER: str = os.getenv("HF_PROVIDER", "cerebras")
    HF_MODEL: str = os.getenv("HF_MODEL", "openai/gpt-oss-120b")

    # Chunking
    CHUNK_TOKENS: int = int(os.getenv("CHUNK_TOKENS", "350"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "0"))  # 0 means unlimited

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.5"))
    MMR_CANDIDATES: int = int(os.getenv("MMR_CANDIDATES", "24"))
    HIERARCHICAL_RETRIEVAL: bool = os.getenv("HIERARCHICAL_RETRIEVAL", "true").lower() in {"1", "true", "yes"}
    SECTION_TOP_K: int = int(os.getenv("SECTION_TOP_K", "3"))
    USE_HYDE: bool = os.getenv("USE_HYDE", "true").lower() in {"1", "true", "yes"}

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
