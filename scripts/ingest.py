from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.store import build_store, extract_text_documents

# %% Load docs (POC: each call transcript is embedded whole, as a single chunk)

def ingest():
    path = config.DATA_DIR / "calls"
            
    documents = extract_text_documents(path)

    # Split into chunks so no single text exceeds the embedding model's memory-safe length
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    documents = splitter.split_documents(documents)
    print(f"{len(documents)} chunks")

    # %% Ingest into the vector store
    build_store(documents)


if __name__ == "__main__":
    ingest()


