from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.store import build_store, extract_text_documents

# %% Load docs (POC: each call transcript is embedded whole, as a single chunk)
path = config.DATA_DIR / "calls"

# path = Path(path)
# documents = []
# for file_path in sorted(path.rglob("*.txt")):
#     documents.append(
#         Document(
#             page_content=file_path.read_text(encoding="utf-8"),
#             metadata={
#                 "source": str(file_path),
#                 "file_name": file_path.name,
#             },
#         )
#     )
        
documents = extract_text_documents(path)

# Split into chunks so no single text exceeds the embedding model's memory-safe length
splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
documents = splitter.split_documents(documents)
print(f"{len(documents)} chunks")

# %% Ingest into the vector store
build_store(documents)

# %% Actually run the RAG pipeline




# %%
