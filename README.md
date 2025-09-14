**Manual Q&A RAG – FastAPI + React**

- Goal: Accurate, efficient Q&A bot over a long PDF manual with careful chunking and high‑quality retrieval.
- Stack: FastAPI backend, React (Vite) frontend, Sentence‑Transformers embeddings + FAISS, Hugging Face Inference for generation (optional), extractive fallback.

Design Highlights
- Thoughtful chunking:
  - Extract layout‑aware text via PyMuPDF and detect headings by font‑size outliers and numbering (e.g., 2.3.4).
  - Build sections with page spans; create overlapping chunks within each section using a semantic splitter.
- Retrieval quality:
  - all‑MiniLM‑L6‑v2 embeddings with FAISS inner‑product over normalized vectors.
  - Hierarchical retrieval (sections → chunks), MMR selection for diversity, optional cross‑encoder rerank.
  - Optional HyDE (hypothetical answer) to enrich queries for nuanced retrieval.
- Answering:
  - Grounded prompting over retrieved chunks via Hugging Face Inference (provider/model configurable).
  - Extractive stitched answers with citations when LLM is disabled.

Setup
- Prereqs: Python 3.10+, Node 18+
- Backend
  - cd backend
  - Windows: python -m venv .venv; . .venv/Scripts/activate
  - macOS/Linux: python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt
  - Place your manual at ../data/draft-oasis-e1-manual-04-28-2024.pdf (relative to backend/).
  - LLM (Hugging Face Inference): set HF_TOKEN, HF_PROVIDER, HF_MODEL in .env.
  - Start: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
- Frontend
  - cd ../frontend
  - npm install
  - npm run dev

Usage
1. Build the index: POST /ingest (reads data/manual.pdf, builds sections + chunks indexes)
2. Ask questions: POST /query with { question, history }; UI renders answers + citations.

Config (env)
- Paths: DATA_DIR, STORAGE_DIR, PDF_PATH
- Embeddings: EMBEDDING_MODEL
- Chunking: CHUNK_TOKENS, CHUNK_OVERLAP, MAX_CHUNKS
- Retrieval: TOP_K, MMR_LAMBDA, MMR_CANDIDATES, HIERARCHICAL_RETRIEVAL, SECTION_TOP_K, USE_HYDE
- LLM (optional): USE_LLM, HF_TOKEN, HF_PROVIDER, HF_MODEL
Manual Q&A RAG – FastAPI + React

Overview
- Goal: Build a precise, efficient Q&A bot over a long PDF manual with thoughtful data preparation and high‑quality retrieval.
- Stack: FastAPI backend, React (Vite) frontend, Sentence‑Transformers + FAISS for retrieval, Hugging Face Inference (optional) for generation, extractive fallback.

What’s Implemented
- Section‑aware ingestion: PyMuPDF text extraction; heading detection by font‑size outliers + numbering (e.g., “2.3.4”); each section keeps page_start/page_end.
- Smart chunking: Semantic split within sections (paragraphs, bullets, sentence ends) with overlap; chunk metadata persisted.
- Retrieval quality: `all‑MiniLM‑L6‑v2` embeddings + FAISS (inner product on normalized vectors), MMR diversity; hierarchical retrieval (sections → chunks) to focus candidates; optional HyDE query enrichment.
- Conversational answering: Grounded prompt with citations (page range + section); history‑aware follow‑ups; clarifying question on low confidence.
- Indexing UX: Auto‑index on first use; background indexing with progress; 202 responses from `/query` while indexing; frontend overlay with progress bar and disabled input.
- HF Inference only: Uses Hugging Face Inference endpoints if `HF_TOKEN` is set; no OpenAI dependency.

Repository Layout
- backend/
  - app/
    - main.py: FastAPI app (ingest, indexing, query, stats) + query guarding and 202 flow.
    - ingest.py: PDF parsing, heading detection, section building, chunking, JSONL persistence.
    - retriever.py: Embeddings + FAISS indexing/search, hierarchical retrieval, MMR, HyDE support, progress‑aware build.
    - qa.py: Prompting, HyDE text generator, validation/quiz helpers, extractive fallback.
    - indexer.py: Background index builder with status/progress.
    - settings.py: Env‑driven configuration.
    - models.py: Pydantic DTOs (QueryRequest/Response, etc.).
  - requirements.txt
  - .env (your local config)
- frontend/
  - React (Vite) app with chat UI, citations viewer, and indexing progress overlay.

Prerequisites
- Python 3.10+
- Node.js 18+

Quick Start
1) Backend
- cd backend
- python -m venv .venv; . .venv/Scripts/activate   (Windows)
  or python -m venv .venv && source .venv/bin/activate   (macOS/Linux)
- pip install -r requirements.txt
- Place your PDF at `backend/data/<file>.pdf` and set `PDF_PATH` in `backend/.env`.
- Optional (Hugging Face Inference): set `HF_TOKEN`, `HF_PROVIDER`, `HF_MODEL` in `backend/.env`.
- Start API: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

2) Frontend
- cd frontend
- npm install
- npm run dev
- Open http://localhost:5173

Indexing Flow
- On first load (no FAISS index): the app starts indexing automatically and shows an overlay:
  - Parsing and chunking PDF → Loading embedding model (first run download) → Building vectors (chunks/sections) → Finalizing
  - The input is disabled until indexing completes.
- Alternatively, trigger manually:
  - POST http://localhost:8000/ingest
  - Poll GET http://localhost:8000/index/status

API Summary
- POST /ingest → starts background indexing using `PDF_PATH` (returns immediately)
- GET /index/status → { status: idle|indexing|ready|error, progress, message }
- POST /index/start → starts background indexing (optional manual trigger)
- POST /query → { question, top_k?, history? } → 200 answer or 202 {status} if indexing
- GET /stats → counts, model name, has_index flag
- GET /health → { status: ok }

Environment Variables (backend/.env)
- Paths
  - DATA_DIR=./data
  - STORAGE_DIR=./storage
  - PDF_PATH=./data/your-file.pdf
- Embeddings
  - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
- LLM (optional, via HF Inference)
  - USE_LLM=true
  - HF_TOKEN=your_hf_token
  - HF_PROVIDER=cerebras   (example)
  - HF_MODEL=openai/gpt-oss-120b   (example)
- Chunking
  - CHUNK_TOKENS=350
  - CHUNK_OVERLAP=60
  - MAX_CHUNKS=0
- Retrieval
  - TOP_K=5
  - MMR_LAMBDA=0.5
  - MMR_CANDIDATES=24
  - HIERARCHICAL_RETRIEVAL=true
  - SECTION_TOP_K=3
  - USE_HYDE=true
- Server
  - HOST=0.0.0.0
  - PORT=8000

Notes & Assumptions
- First run downloads the embedding model; this can take a few minutes and is shown as “Loading embedding model…”.
- The system grounds answers strictly in the manual; for out‑of‑scope questions it asks for clarification or explains limits.
- If no `HF_TOKEN` is set, the system falls back to extractive answers with citations.

Design Decisions
- Retrieval‑first grounding with hierarchical narrowing + MMR for precision and diversity; HyDE optional to improve recall on nuanced phrasings.
- Section‑aware chunking to preserve semantics and page ranges → better citations and fewer edge truncations.
- Stateless server memory: the client sends message history each request; simpler to scale and debug.
- Lean runtime: removed unused endpoints and reranking to reduce dependencies and speed up first build.

Troubleshooting
- “Stuck” progress: initial model download is the long step; it is now surfaced in the progress message. If it truly doesn’t move, check backend logs and GET /index/status for `last_error`.
- 202 on /query: indexing is still running; the frontend overlay will finish and re‑enable the chat automatically.
- 409/index missing (legacy): ensure `PDF_PATH` points to an existing file and POST /ingest, or just reload the UI to auto‑start indexing.

