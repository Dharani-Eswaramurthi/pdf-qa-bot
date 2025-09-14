**Manual Q&A RAG – FastAPI + React**

- Goal: Accurate, efficient Q&A bot over a long PDF manual with careful chunking and high‑quality retrieval.
- Stack: FastAPI backend, React (Vite) frontend, Sentence‑Transformers embeddings + FAISS, Hugging Face Inference for generation (optional), extractive fallback.

Design Highlights
- Thoughtful chunking:
  - Extract layout‑aware text via PyMuPDF and detect headings by font‑size outliers and numbering (e.g., 2.3.4).
  - Build sections with page spans; create overlapping chunks within each section using a semantic splitter.
- Retrieval quality:
  - `all‑MiniLM‑L6‑v2` embeddings with FAISS inner‑product over normalized vectors.
  - Hierarchical retrieval (sections → chunks), MMR selection for diversity, optional cross‑encoder rerank.
  - Optional HyDE (hypothetical answer) to enrich queries for nuanced retrieval.
- Answering:
  - Grounded prompting over retrieved chunks via Hugging Face Inference (provider/model configurable).
  - Extractive stitched answers with citations when LLM is disabled.

Setup
- Prereqs: Python 3.10+, Node 18+
- Backend
  - `cd backend`
  - Windows: `python -m venv .venv; . .venv/Scripts/activate`
  - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - Place your manual at `../data/manual.pdf` (relative to backend/).
  - LLM (Hugging Face Inference): set `HF_TOKEN`, `HF_PROVIDER`, `HF_MODEL` in `.env`.
  - Start: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Frontend
  - `cd ../frontend`
  - `npm install`
  - `npm run dev`

Usage
1. Build the index: `POST /ingest` (reads `data/manual.pdf`, builds sections + chunks indexes)
2. Ask questions: `POST /query` with `{ question, history }`; UI renders answers + citations.

Config (env)
- Paths: `DATA_DIR`, `STORAGE_DIR`, `PDF_PATH`
- Embeddings: `EMBEDDING_MODEL`
- Chunking: `CHUNK_TOKENS`, `CHUNK_OVERLAP`, `MAX_CHUNKS`
- Retrieval: `TOP_K`, `MMR_LAMBDA`, `MMR_CANDIDATES`, `HIERARCHICAL_RETRIEVAL`, `SECTION_TOP_K`, `USE_HYDE`
- LLM (optional): `USE_LLM`, `HF_TOKEN`, `HF_PROVIDER`, `HF_MODEL`

