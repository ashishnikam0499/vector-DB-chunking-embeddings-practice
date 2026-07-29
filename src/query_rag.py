import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
COLLECTION_NAME =os.getenv("COLLECTION_NAME", "pdf_rag_collection")

RAG_PROMPT_TEMPLATE = """
You are an intelligent enterprise assistant helping with document analysis.
Answer the user's question using ONLY the provided docuement context below.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

---
DOCUMENT CONTEXT:
{context}
---
USER QUESTION:
{question}

ANSWER:
"""

def initialize_rag_components():
    """Connects to the existing vector DB and sets up Gemini LLM"""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from your environment configuration.")

    # initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # connect to vector store
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # initialize Gemini chat model for precise text generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2 # low temperature ensures factual grounding
    )

    return vector_store, llm

def ask_question(question_text: str, metadata_filter: dict = None):
    """Executes the complete RAG loop with optional filtering."""
    vector_store, llm = initialize_rag_components()

    # configure retriever parameter dynamically
    search_kwargs = {"k": 4}
    if metadata_filter:
        print(f"Applying search filter constraints: {metadata_filter}")
        search_kwargs["filter"] = metadata_filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    print(f"\n Searching vector space for: '{question_text}'...")
    relevant_docs = retriever.invoke(question_text)

    if not relevant_docs:
        print(f"No relevant document chunks matched your query or filter criteria.")
        return

    # extract manifest metadata alongside text contents for deep auditing
    context_blocks = []
    for doc in relevant_docs:
        meta = doc.metadata
        header = f"[ID: {meta.get('doc_id')} | File: {meta.get('source')} | Dept: {meta.get('department')} | Type: {meta.get('doc_type')} | Page: {meta.get('page', 0) + 1}]"
        context_blocks.append(f"{header}\n{doc.page_content}")

    context_content = "\n\n".join(context_blocks)

    # assemble and run the chain via LangChain Expression Language (LCEL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    rag_chain = prompt | llm |  StrOutputParser()

    print("Synthesizing structured answer via Gemini...")
    answer = rag_chain.invoke({"context": context_content, "question": question_text})

    print("\n" + "="*50 + "\nANSWER:\n" + "="*50)
    print(answer)
    print("="*50 + "\n")

    # print manifest citations underneath the answer text block
    print("Verified Manifest Sources Citations:")
    for doc in relevant_docs:
        meta = doc.metadata
        print(f" - [{meta.get('doc_id')}] {meta.get('source')} (Page {meta.get('page', 0) + 1}) -> Topic: {meta.get('topic')}")

if __name__ == "__main__":
    # Test 1: global search across all ingested vector payloads
    print("--- Running Global Search Query ---")
    ask_question("What is the standard procedure for onboarding?")

    # Test 2: specifc search using metadata filter
    print("\n" + "#"*60 + "\n")
    print("--- Running Scoped Metadata Query ---")

    # Pack parameters into an $and conditional sequence to prevent internal API crashes
    safe_chroma_filter = {
        "$and": [
            {"department": "HR"},
            {"doc_type": "Policy"}
        ]
    }

    ask_question(
        question_text="What are the rules regarding policy updates?",
        metadata_filter=safe_chroma_filter
    )
