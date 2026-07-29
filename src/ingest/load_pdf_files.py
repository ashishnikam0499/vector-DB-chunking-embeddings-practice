import os
import csv
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv()

PDF_DIR = os.getenv("PDF_DATA_DIR", "data/pdfs")
CACHE_DIR = Path("data/cache")

def load_manifest(manifest_path: Path) -> dict:
    """Reads the csv manifest and returns a dictionary indexed by filename."""
    manifest_data = {}
    if not manifest_path.exists():
        print(f"⚠️ Warning: Manifest file not found at {manifest_path}")
        return manifest_data

    try:
        with open(manifest_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use filename as the lookup key
                filename = row.get("filename", "").strip()
                if filename:
                    manifest_data[filename] = {
                        "doc_id": row.get("doc_id", "").strip(),
                        "department": row.get("department", "").strip(),
                        "topic": row.get("topic", "").strip(),
                        "doc_type": row.get("doc_type", "").strip()
                    }
        print(f"Loaded metadata descriptions for {len(manifest_data)} files from manifest.")
    except Exception as e:
        print(f"Failed to parse manifest CSV: {e}")
        
    return manifest_data

def load_all_pdfs():
    """Finds all PDFs, attaches manifest metadata, and loads the document objects."""
    pdf_path = Path(PDF_DIR)
    if not pdf_path.exists():
        print(f"Creating directory: {pdf_path}")
        pdf_path.mkdir(parents=True, exist_ok=True)
        return []

    # Look for the manifest inside your data/pdfs folder
    manifest_file = pdf_path / "_manifest.csv"
    metadata_lookup = load_manifest(manifest_file)

    pdf_files = list(pdf_path.rglob("*.pdf"))
    all_documents = []
    
    print(f"Found {len(pdf_files)} PDF file(s) on disk.")
    
    for file_path in pdf_files:
        # Ignore the manifest if it accidentally matches a glob rule
        if file_path.name.startswith("_"):
            continue
            
        print(f"Parsing: {file_path.name}...")
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            
            # Extract manifest metadata row for this specific file
            file_meta = metadata_lookup.get(file_path.name, {})
            
            for doc in docs:
                # Essential layout metadata
                doc.metadata["source"] = file_path.name
                
                # Structural enterprise metadata injected from CSV
                doc.metadata["doc_id"] = file_meta.get("doc_id", "Unknown")
                doc.metadata["department"] = file_meta.get("department", "General")
                doc.metadata["topic"] = file_meta.get("topic", "General Documentation")
                doc.metadata["doc_type"] = file_meta.get("doc_type", "Unknown")
                
            all_documents.extend(docs)
        except Exception as e:
            print(f"Failed to parse {file_path.name}: {e}")
            
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "loaded_docs.pkl", "wb") as f:
        pickle.dump(all_documents, f)
        
    print(f"Successfully loaded {len(all_documents)} pages into internal staging cache.")
    return all_documents

if __name__ == "__main__":
    load_all_pdfs()
