import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CACHE_DIR = Path("data/cache")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

def chunk_documents():
    """Loads serialized pages and chunks them into overlapping text blocks."""
    cache_file = CACHE_DIR / "loaded_docs.pkl"
    if not cache_file.exists():
        print(f"Error: Could not find loaded documents. Run load.py first.")
        return

    with open(cache_file, "rb") as f:
        documents = pickle.load(f)

    print(f"Splitting {len(documents)} document pages...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents=documents)

    with open(CACHE_DIR / "chunked_docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Generated {len(chunks)} text chunks with overlap.")
    return chunks

if __name__ == "__main__":
    chunk_documents()
