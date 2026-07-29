import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini API key: {gemini_api_key}")

CACHE_DIR = Path("data/cache")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag_collection")

def get_embedding_model():
    """Initializes Google Gemini if key found and defaults back to HuggingFace locally."""

    if os.getenv("GEMINI_API_KEY"):
        print(f"Initializing cloud-based Google Gemini Embediings...")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", # Google's standard embedding model
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        print(f"No API key found. Initializing local HuggingFace Embeddings...")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def embed_and_store():
    """Loads chunks, instanciates embeddings, and seeds ChromaDB."""
    cache_file = CACHE_DIR / "chunked_docs.pkl"

    if not cache_file.exists():
        print("Error: Could not find chunked text. Run chunk.py first.")
        return

    with open(cache_file, "rb") as f:
        chunks = pickle.load(f)

    embeddings = get_embedding_model()

    print(f"Vectorizing {len(chunks)} chunks and saving to {CHROMA_DIR}...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Vector database build successful and persistent!")

    return vector_store


if __name__ == "__main__":
    embed_and_store()
