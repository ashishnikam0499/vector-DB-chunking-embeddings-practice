import os
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

CACHE_DIR = Path("data/cache")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag_collection")

def get_embedding_model():
    """Initializes Google Gemini if keys exist; defaults back to HuggingFace locally."""
    if os.getenv("GEMINI_API_KEY"):
        print("Initializing cloud-based Google Gemini Embeddings...")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        print("No API Key found. Initializing local HuggingFace Embeddings...")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def generate_deterministic_id(doc) -> str:
    """Generates a stable, reproducible ID string based on file, page, and chunk text."""
    source_file = doc.metadata.get("source", "unknown")
    page_num = doc.metadata.get("page", 0)
    text_content = doc.page_content
    
    # Create a unique tracking fingerprint combining source coordinates and text content
    fingerprint = f"{source_file}_page_{page_num}_{text_content}"
    
    # Hash down to a standard 32-character hexadecimal string
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()

def embed_and_store():
    """Loads chunks, assigns deterministic tracking keys, and upserts into ChromaDB."""
    cache_file = CACHE_DIR / "chunked_docs.pkl"
    if not cache_file.exists():
        print("Error: Could not find chunked text. Run chunk.py first!")
        return

    with open(cache_file, "rb") as f:
        chunks = pickle.load(f)

    embeddings = get_embedding_model()
    
    print(f"Connecting to vector database at {CHROMA_DIR}...")
    
    # 1. Initialize or connect to the persistent vector client
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # 2. Build explicit tracking keys for every chunk in the sequence
    print("Generating deterministic chunk signatures for tracking...")
    chunk_ids = [generate_deterministic_id(chunk) for chunk in chunks]
    
    # 3. Use add_documents with explicit IDs to force a strict upsert operation
    print(f"Syncing {len(chunks)} chunks into ChromaDB (updating modified, ignoring duplicates)...")
    vector_store.add_documents(documents=chunks, ids=chunk_ids)
    
    print("Vector database synchronization complete!")
    return vector_store

if __name__ == "__main__":
    embed_and_store()
