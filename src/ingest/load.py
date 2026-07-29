import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv()  # Load environment variables from .env file

PDF_DIR = os.getenv("PDF_DATA_DIR", "data/pdfs")  # Default to 'data/pdfs' if not set
CACHE_DIR = Path("data/cache")

def load_all_pdfs():

    """Finds all pdfs, loads text and extract basic folder-level metadata."""

    pdf_path = Path(PDF_DIR)

    if not pdf_path.exists():
        print(f"Creating directory: {pdf_path}")
        pdf_path.mkdir(parents=True, exist_ok=True)
        print(f"Please place your pdfs inside data/pdfs directory and rerun.")
        return []

    pdf_files = list(pdf_path.rglob("*.pdf"))
    all_documents = []

    print(f"Found {len(pdf_files)} pdf files for ingestion.")

    for file_path in pdf_files:

        print(f"Parsing: {file_path}...")
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path.name

            all_documents.extend(docs)

        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "loaded_docs.pkl", "wb") as f:
        pickle.dump(all_documents, f)

    print(f"Successfully loaded {len(all_documents)} total pages.")
    return all_documents

if __name__ == "__main__":
    load_all_pdfs() 
