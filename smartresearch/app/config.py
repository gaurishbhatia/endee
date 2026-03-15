"""
Configuration — loads environment variables and exposes typed constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root (smartresearch/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Endee ────────────────────────────────────────────────────────────────────
ENDEE_BASE_URL: str = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")
ENDEE_INDEX_NAME: str = os.getenv("ENDEE_INDEX_NAME", "smartresearch_docs")
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimension

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# ─── Google Gemini ────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "models/gemini-1.5-flash"
