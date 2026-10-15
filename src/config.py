"""
MedRAG Configuration
--------------------
Centralised configuration loaded from the .env file.
All modules should import settings from here to ensure consistency.

LLM backend  : Groq Cloud API (fast inference, free tier available).
Vector DB    : MongoDB Atlas with vector search index.
Embeddings   : HuggingFace MiniLM-L6-v2 (local, no API key needed).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


# ─── MongoDB ─────────────────────────────────────────────────────────────────
_raw_uri: str = os.getenv("MONGODB_URI", "")

# Change this block in your config.py to match your implementation:
if _raw_uri and not any(flag in _raw_uri for flag in ["tlsAllowInvalidCertificates", "tlsInsecure"]):
    sep = "&" if "?" in _raw_uri else "?"
    MONGODB_URI: str = f"{_raw_uri}{sep}tlsInsecure=true"
else:
    MONGODB_URI: str = _raw_uri



MONGODB_DB_NAME: str         = os.getenv("MONGODB_DB_NAME",         "medrag")
MONGODB_COLLECTION_NAME: str = os.getenv("MONGODB_COLLECTION_NAME", "medical_chunks")
MONGODB_VECTOR_INDEX: str    = os.getenv("MONGODB_VECTOR_INDEX",    "medrag_vector_index")

# ─── Groq / LLM ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str   = os.getenv("GROQ_MODEL",   "openai/gpt-oss-120b")

# Deprecated alias kept so stale imports don't crash
OLLAMA_MODEL: str = GROQ_MODEL

# ─── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

# ─── Ingestion ───────────────────────────────────────────────────────────────
DOCS_DIR: str     = os.getenv("DOCS_DIR",      "./data/documents")
CHUNK_SIZE: int   = int(os.getenv("CHUNK_SIZE",   "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ─── LLM Generation ─────────────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1   # Low temp for factual medical responses
LLM_MAX_TOKENS: int    = 1024

# ─── Medical Disclaimer ─────────────────────────────────────────────────────
MEDICAL_DISCLAIMER: str = (
    "\n\n---\n⚕️ **Medical Disclaimer**: This information is provided for "
    "educational purposes only and does not constitute professional medical "
    "advice, diagnosis, or treatment. Always consult a qualified healthcare "
    "provider for medical concerns."
)


def validate_config() -> list[str]:
    """Validate that critical configuration values are set. Returns a list of warnings."""
    warnings: list[str] = []
    if not MONGODB_URI or "<user>" in MONGODB_URI:
        warnings.append("MONGODB_URI is not set or still contains placeholder values.")
    if not GROQ_API_KEY or GROQ_API_KEY == "<your-groq-api-key>":
        warnings.append("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
    return warnings


def get_mongo_client():
    """Return a MongoClient with robust SSL handling for Windows compatibility.

    OpenSSL 1.1.1 (bundled with CPython 3.10) has known TLS handshake issues
    with MongoDB Atlas.  We work around this with ``tlsInsecure=True`` which
    disables cert verification and hostname checking while keeping encryption.
    """
    from pymongo import MongoClient

    # Keep tlsInsecure to handle the Windows/Atlas certificate fallback cleanly
    kwargs: dict = {
        "tls": True,
        "tlsInsecure": True,
    }

    try:
        import certifi
        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        pass

    return MongoClient(MONGODB_URI, **kwargs)
