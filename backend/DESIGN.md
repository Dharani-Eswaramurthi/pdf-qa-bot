**Manual Q&A RAG – Approach + Runbook**

- Goal: Accurate, efficient Q&A over a long PDF manual with careful data prep and high‑quality retrieval. Designed for correctness, transparency, and responsive UX.
- Stack: FastAPI backend, React (Vite) frontend, Sentence‑Transformers + FAISS, Hugging Face Inference for generation (optional).

**Architecture**
- **Ingestion:** PyMuPDF parses pages with font sizes; heading heuristic forms sections with page spans. A smart splitter creates overlapping chunks within each section.
- **Indexes:**
  - Chunk embeddings index (FAISS, inner product on normalized vectors).
  - Section embeddings index for hierarchical retrieval (focus chunk search inside top sections).
- **Retrieval:** MMR selection for diversity; optional cross‑encoder rerank. Optional HyDE enrichment generates a short hypothetical answer to improve retrieval for nuanced queries.
- **Generation:** Grounded prompting over retrieved chunks via Hugging Face Inference; extractive fallback with citations when LLMs are disabled.
- **Memory + Mode Control:** History‑aware queries, session directives (quiz mode, no‑reveal, avoid tables), clarifying questions on low confidence.

**Data Preparation & Chunking**
- **Heading detection:** Font‑size outliers + numbering pattern (e.g., “2.3.4”) produce logical sections; each section records `page_start`/`page_end`.
- **Semantic splitting:** Prefers natural boundaries (double newlines, bullets, sentence ends) with token‑approx overlap to reduce context loss.
- **Transparency:** Persists `sections.jsonl` and `chunks.jsonl` for inspectability and reproducibility.

**Retrieval Strategy**
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` stored in FAISS with inner product on unit vectors.
- **Hierarchical retrieval:** Search top sections, then restrict chunk candidates to those sections for higher precision.
- **MMR:** Selects diverse top‑k to avoid redundancy; reordering via cross‑encoder reranker is optional.
- **HyDE (optional):** Generate a 2–4 sentence hypothetical answer to steer embedding search towards semantically relevant content.

**Question Understanding**
- **History‑aware search:** Rewrites the search query using the last user turns to handle follow‑ups.
- **Session directives:** Detects “ask me questions one by one and validate”, “do not reveal answers”, “avoid tables / be conversational” and adapts behavior.
- **Low‑confidence guard:** If retrieval confidence is low, asks for clarifying detail (e.g., item code or section) instead of guessing.

**Answering & Citations**
- **LLM prompt:** System persona emphasizes grounded, conversational answers with citations like `[p.start-p.end, Section Title]`. Avoids tables by default unless requested or inherently tabular.
- **Validation mode:** Generates one exam‑style question at a time; the next user message is validated against the manual with citations, respecting “no‑reveal” if requested.
- **Fallback:** If LLM unavailable, returns concise extractive answer stitched from sources with citations.

**Run Instructions**
- Prereqs: Python 3.10+, Node 18+.

- Backend
  - `cd backend`
  - Windows PowerShell: `python -m venv .venv; . .venv/Scripts/activate`
  - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - Place the manual at `../data/manual.pdf` (relative to `backend/`).
  - Optional LLM: set Hugging Face token for chat completions
    - `.env`: `USE_LLM=true`, `HF_TOKEN=...`, `HF_PROVIDER=cerebras`, `HF_MODEL=openai/gpt-oss-120b`
  - Start API: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
  - Build index: `POST http://localhost:8000/ingest` (uses `data/manual.pdf`)

- Frontend
  - `cd ../frontend`
  - `npm install`
  - `npm run dev`
  - Open `http://localhost:5173`

**API Quick Start**
- Health: `GET /health`
- Ingest: `POST /ingest` (reads `data/manual.pdf`, builds both section + chunk indexes)
- Ask: `POST /query` with `{ question: "...", history: [{role, content}, ...] }`
- Search only: `GET /search?q=...`
- Stats: `GET /stats`

**Configuration (env)**
- Paths: `DATA_DIR`, `STORAGE_DIR`, `PDF_PATH`
- Embeddings: `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`)
- Chunking: `CHUNK_TOKENS` (350), `CHUNK_OVERLAP` (60), `MAX_CHUNKS` (0)
- Retrieval: `TOP_K` (5), `MMR_LAMBDA` (0.5), `MMR_CANDIDATES` (24)
- Advanced retrieval: `HIERARCHICAL_RETRIEVAL=true`, `SECTION_TOP_K=3`, `USE_HYDE=true`
- LLM: `USE_LLM=true`, `HF_TOKEN`, `HF_PROVIDER`, `HF_MODEL`

**Testing & Validation**
- After ingest, try:
  - “Summarize the interview protocol for BIMS (bullets).”
  - “Where are coding examples listed?”
  - “Ask me one practice question and validate my answer (don’t reveal).”
- Low‑confidence example: “Tell me about tables unrelated to OASIS” → expects a clarifying question.
- UI renders citations with page ranges; expand source excerpts to inspect grounding.

**Design Decisions**
- Prioritize retrieval precision (hierarchical + MMR + HyDE) to minimize hallucination and improve context fitness.
- Enforce grounded answers with transparent citations and politely decline out‑of‑scope requests.
- Keep state stateless on the server; the client sends conversation history for memory, improving portability and scalability.
- Avoid tables by default for conversational UX; allow explicit opt‑in for inherently tabular data.

**Assumptions**
- The manual is predominantly text‑based and extractable with PyMuPDF; OCR is out‑of‑scope.
- Public Sentence‑Transformers models can be downloaded; if HF auth is misconfigured, anonymous download is used for public models.
- Network available for model download/inference; otherwise, the system falls back to extractive answers without LLM.

**Future Enhancements**
- Add per‑section summarization embeddings and entailment checks for even stronger grounding.
- Add evaluation harness (NDCG/MRR) with a small labeled QA set.
- Support per‑session server memory (session_id) and summarization for very long chats.
