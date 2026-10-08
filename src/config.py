import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # makes API keys in .env visible to the LLM and Langfuse SDKs

# --- Paths ---
DATA_DIR = ROOT / "data"
QDRANT_PATH = str(ROOT / "qdrant_db")  # local on-disk vector DB
LOG_DIR = ROOT / "logs"
COLLECTION = "annual_reports"

# --- Models (all run locally except the LLM) ---
DENSE_MODEL = "BAAI/bge-m3"  
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Embedding resource limits ---
EMBED_BATCH_SIZE = 8  # texts per forward pass; lower if memory is tight
EMBED_MAX_SEQ_LENGTH = 512  # bge-m3 defaults to 8192 tokens, which is very memory-hungry

# --- Chunking / retrieval ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVE_K = 30  # candidates from hybrid search
RERANK_TOP_N = 4  # text files/passages (depends on method) sent to the LLM after reranking

# --- LLM ("google" or "openai") ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
