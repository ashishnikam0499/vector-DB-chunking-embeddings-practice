# vector-DB-chunking-embeddings-practice

Ingestion pipeline that takes a corpus of hundreds of PDFs and turns them
into embedded, queryable vectors in a vector database — using LangChain for
loading and chunking.

## Pipeline

```
data/pdfs/*.pdf
      |
      v
[1] Load & parse       -- extract text from each PDF (LangChain document loader)
      |
      v
[2] Chunk               -- split into overlapping chunks (LangChain text splitter)
      |
      v
[3] Embed                -- embedding model turns each chunk into a vector
      |
      v
[4] Vector DB             -- store chunks + vectors + metadata (source, page, department)
```

## Suggested repo structure

```
pdf-rag-langgraph/
├── data/
│   └── pdfs/                  # source PDFs live here (see note below)
├── src/
│   └── ingest/
│       ├── load.py            # PDF -> LangChain Document objects
│       ├── chunk.py           # Document -> chunks (RecursiveCharacterTextSplitter)
│       └── embed.py           # chunks -> vectors, upsert into vector DB
├── notebooks/
│   └── exploration.ipynb      # ad hoc testing, chunk-size experiments
├── tests/
│   └── test_ingest.py         # sanity checks: chunk counts, embedding shapes
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Suggested stack

| Stage | Options |
|---|---|
| PDF parsing | LangChain's `PyPDFLoader` or `UnstructuredPDFLoader` (better for messier layouts) |
| Chunking | LangChain's `RecursiveCharacterTextSplitter` (start simple, tune chunk size/overlap later) |
| Embeddings | OpenAI/Gemini embeddings API, or a free local model via `sentence-transformers` |
| Vector DB | Chroma (easiest local start), or Qdrant/Weaviate/Pinecone for something closer to production |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # fill in your API keys
python src/ingest/load.py   # parse PDFs into LangChain Documents
python src/ingest/chunk.py  # split into chunks
python src/ingest/embed.py  # embed + load into vector DB
```

## A note on the sample PDFs

`data/pdfs/` isn't included in this repo by default — PDFs are usually large
and shouldn't be committed to git. Add a line like this to `.gitignore`:

```
data/pdfs/*.pdf
```

For this project, use the included **`nimbus_robotics_pdfs.zip`** (200 files)
as your test corpus: a fictional company's internal knowledge base spanning
HR, IT, Security, Finance, Legal, Engineering, Facilities, and Sales, with
overlapping topics and multiple document types (policy / procedure / FAQ /
quick reference) per topic. `_manifest.csv` inside the zip lists every file's
department and topic — useful as a quick ground-truth reference once you get
to testing retrieval later (e.g. does "PTO accrual" pull back the HR PTO
policy doc, not the IT backup policy doc).

## Things worth exploring / mentioning in an interview

- **Chunk size and overlap tradeoffs**: too small loses context, too large
  dilutes relevance in similarity search later. Worth experimenting with a
  couple of configurations and being able to explain the tradeoff, not just
  picking a default.
- **Metadata at ingestion time, not as an afterthought**: tagging each chunk
  with its source file, department, and doc type (all available in
  `_manifest.csv` here) up front makes filtered retrieval possible later —
  this is worth setting up now even before you build the retrieval step.
- **Embedding model choice**: dimension size, cost, and whether it's a
  general-purpose vs. domain-tuned model all affect retrieval quality
  downstream — being able to name this tradeoff is a good signal even at the
  ingestion stage.
- **Idempotent ingestion**: re-running the pipeline shouldn't create
  duplicate vectors for the same chunk — worth deciding early how you key/
  dedupe upserts into the vector DB.
