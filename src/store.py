from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
import os 

from src import config


def _embeddings():
    dense = HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": config.EMBED_BATCH_SIZE},
    )
    dense._client.max_seq_length = config.EMBED_MAX_SEQ_LENGTH
    sparse = FastEmbedSparse(model_name=config.SPARSE_MODEL)
    return dense, sparse


def extract_text_documents(path: str): 
    path = Path(path)
    documents = []
    for file_path in sorted(path.rglob("*.txt")):
        documents.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                },
            )
        )
    return documents 

def build_store(docs: list[Document]):
    """Create (or overwrite) the local Qdrant collection from chunked documents."""
    dense, sparse = _embeddings()
    return QdrantVectorStore.from_documents(
        docs,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        path=config.QDRANT_PATH,
        collection_name=config.COLLECTION,
        force_recreate=True,
        batch_size=config.EMBED_BATCH_SIZE,
    )

def load_store():
    """Open the existing local Qdrant collection."""
    dense, sparse = _embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        path=config.QDRANT_PATH,
        collection_name=config.COLLECTION,
    )
