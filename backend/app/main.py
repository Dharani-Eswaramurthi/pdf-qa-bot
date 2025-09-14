import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .settings import settings
from .models import IngestResponse, QueryRequest, QueryResponse, Citation, SearchResponse, StatsResponse
from .ingest import ingest_pdf
from .retriever import VectorStore
from .qa import generate_answer
from .qa import generate_quiz_question, validate_user_answer
from .qa import generate_hypothetical_answer


app = FastAPI(title="Manual Q&A RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(path: Optional[str] = None):
    pdf_path = path

    info = ingest_pdf(pdf_path)
    # Build vector index
    vs = VectorStore()
    vs.build()

    return IngestResponse(
        chunks_indexed=info["chunks"],
        sections_detected=info["sections"],
        pages_processed=info["pages"],
        storage_dir=os.path.abspath(settings.STORAGE_DIR),
    )


def _build_search_query(question: str, history):
    if not history:
        return question
    # Use recent user turns to add context, but keep short (~512 chars)
    user_prev = [m.content for m in history if getattr(m, 'role', getattr(m, 'get', lambda k, d=None: None)('role', '')) == 'user'] if isinstance(history, list) else []
    try:
        # history may be list of Pydantic models
        user_prev = [m.content for m in history if getattr(m, 'role', '') == 'user']
    except Exception:
        # or list of dicts
        user_prev = [m.get('content', '') for m in history if m.get('role') == 'user']
    tail = " ".join(user_prev[-2:] + [question]).strip()
    return tail[:512]


def _derive_directives(history) -> str:
    """Heuristic extraction of session preferences and context directives.
    Looks for patterns like 'do not tell me the answer', 'ask me questions and validate', etc.
    """
    flags = {
        "QUIZ_MODE": False,
        "NO_REVEAL": False,
        "NO_TABLES": False,
    }
    texts = []
    try:
        texts = [getattr(m, 'content', '') for m in (history or [])]
        roles = [getattr(m, 'role', '') for m in (history or [])]
    except Exception:
        texts = [m.get('content', '') for m in (history or [])]
        roles = [m.get('role', '') for m in (history or [])]

    joined = "\n".join([t.lower() for t in texts if t])
    if 'ask me questions' in joined and 'validate' in joined:
        flags["QUIZ_MODE"] = True
    if 'do not tell me the answer' in joined or "don't tell me the answer" in joined:
        flags["NO_REVEAL"] = True
    if 'avoid table' in joined or 'no table' in joined or 'like chatting' in joined or 'chatting' in joined or 'conversational' in joined:
        flags["NO_TABLES"] = True

    parts = [
        "Use conversation context to resolve pronouns and ellipsis.",
        "If ambiguous, ask a short clarifying question before answering.",
        "Ground everything strictly in the provided context chunks with citations.",
    ]
    if flags["QUIZ_MODE"]:
        parts.append(
            "Quiz mode: Ask succinct, standalone questions from the manual as numbered bullets (not tables). When the user answers, validate strictly against the document with citations and provide targeted feedback."
        )
    if flags["NO_REVEAL"]:
        parts.append(
            "Do not reveal answers directly; only evaluate and guide unless explicitly allowed."
        )
    if flags["NO_TABLES"]:
        parts.append(
            "Avoid tables; prefer short paragraphs and bullet lists unless the user explicitly asks for a table or the data is inherently tabular."
        )

    return "\n- ".join(["- "+p for p in parts])


def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}


def _language_pref(text: str) -> Optional[str]:
    t = text.strip().lower()
    if t in {"english", "en", "use english", "english language"}:
        return "English"
    return None


def _last_assistant_question(history) -> Optional[str]:
    last = None
    try:
        for m in reversed(history or []):
            role = getattr(m, 'role', '') if not isinstance(m, dict) else m.get('role')
            content = getattr(m, 'content', '') if not isinstance(m, dict) else m.get('content', '')
            if role == 'assistant' and content:
                cl = content.lower()
                # Only treat as quiz if assistant explicitly asked for an answer/validation
                if ('please type your answer' in cl) or ("i'll validate" in cl) or ("i’ll validate" in cl) or ("please write your answer" in cl):
                    last = content
                    break
    except Exception:
        pass
    return last


def _is_model_query(text: str) -> bool:
    t = text.strip().lower()
    keys = [
        'what model are you', 'which model are you', 'what llm', 'which llm',
        'what embedding model', 'which embedding', 'what reranker', 'which reranker',
        'what are you running', 'model version', 'which ai model'
    ]
    return any(k in t for k in keys)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or settings.TOP_K
    vs = VectorStore()

    # Meta intents: greeting/language
    if _is_greeting(req.question):
        return QueryResponse(answer="Hi! How can I help you with the manual?", citations=[], used_llm=False)
    lang = _language_pref(req.question)
    if lang:
        return QueryResponse(answer=f"Understood. I will continue in {lang}.", citations=[], used_llm=False)
    if _is_model_query(req.question):
        llm = f"Hugging Face Inference • {settings.HF_PROVIDER} • {settings.HF_MODEL}"
        emb = settings.EMBEDDING_MODEL_NAME
        rerank = f"enabled • {settings.RERANK_MODEL_NAME}" if getattr(settings, 'RERANK', False) else "disabled"
        msg = (
            f"I’m Patrick AI. Runtime details:\n"
            f"- Retrieval: sentence-transformers embeddings ({emb}) + FAISS + MMR\n"
            f"- Reranking: {rerank}\n"
            f"- Generator: {llm}\n"
            f"I cite the manual for content answers; meta details like this come from configuration."
        )
        return QueryResponse(answer=msg, citations=[], used_llm=False)

    # Build directives from history
    hist_dicts = None
    if req.history:
        hist_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    directives = _derive_directives(req.history or [])

    # Quiz one-by-one: if last assistant asked a question, treat current input as answer to validate
    last_q = _last_assistant_question(req.history)
    if last_q and "question" in last_q.lower():
        search_q = last_q.strip()
        results = vs.search(search_q, top_k=top_k, mmr_lambda=settings.MMR_LAMBDA, mmr_candidates=settings.MMR_CANDIDATES)
        answer_text = validate_user_answer(req.question, results, directives)
        citations = [
            Citation(
                chunk_id=r["id"],
                section_title=r.get("section_title"),
                page_start=r.get("page_start"),
                page_end=r.get("page_end"),
                score=float(r.get("score", 0.0)),
                text=r.get("text", ""),
            )
            for r in results
        ]
        return QueryResponse(answer=answer_text, citations=citations, used_llm=True, debug={"mode": "validate", "search_q": search_q})

    # If user asked to start practice/quiz or history implies quiz mode + one-by-one, generate one question
    joined = (req.question or "").lower()
    if ("ask me questions" in joined or "quiz" in joined or "practice" in joined) or ("Quiz mode" in directives and "Ask succinct" in directives):
        seed = "interview protocol OR coding instructions OR discharge OR BIMS OR scoring OR assessment OR examples"
        results = vs.search(seed, top_k=top_k, mmr_lambda=settings.MMR_LAMBDA, mmr_candidates=settings.MMR_CANDIDATES)
        qtext = generate_quiz_question(results, hist_dicts, directives) or "Here is a question from the manual: What is the key instruction described?"
        citations = [
            Citation(
                chunk_id=r["id"],
                section_title=r.get("section_title"),
                page_start=r.get("page_start"),
                page_end=r.get("page_end"),
                score=float(r.get("score", 0.0)),
                text=r.get("text", ""),
            )
            for r in results
        ]
        return QueryResponse(answer=qtext, citations=citations, used_llm=True, debug={"mode": "quiz_question"})

    # Default: advanced retrieval + QA over manual
    search_q = _build_search_query(req.question, req.history or [])
    hyde_text = None
    if settings.USE_HYDE:
        try:
            hist_dicts = hist_dicts or []
            hyde_text = generate_hypothetical_answer(req.question, hist_dicts)
        except Exception:
            hyde_text = None
    # Prefer hierarchical search if available
    try:
        results = vs.search_advanced(
            search_q,
            top_k=top_k,
            mmr_lambda=settings.MMR_LAMBDA,
            mmr_candidates=settings.MMR_CANDIDATES,
            hyde=hyde_text,
            use_hierarchical=settings.HIERARCHICAL_RETRIEVAL,
            section_top_k=settings.SECTION_TOP_K,
        )
    except Exception:
        results = vs.search(search_q, top_k=top_k, mmr_lambda=settings.MMR_LAMBDA, mmr_candidates=settings.MMR_CANDIDATES)
    # If retrieval confidence is low, ask a clarifying question instead of guessing
    try:
        top_score = max((float(r.get('score', 0.0)) for r in results), default=0.0)
    except Exception:
        top_score = 0.0
    if top_score < 0.2:
        clarify = (
            "I might need a bit more detail to find the right place in the manual. "
            "Could you mention a section, page, or specific item (e.g., GG0170)?"
        )
        return QueryResponse(answer=clarify, citations=[], used_llm=False, debug={"top_k": top_k, "search_q": search_q, "note": "low_confidence"})
    answer, used_llm = generate_answer(req.question, results, hist_dicts, directives)

    citations = [
        Citation(
            chunk_id=r["id"],
            section_title=r.get("section_title"),
            page_start=r.get("page_start"),
            page_end=r.get("page_end"),
            score=float(r.get("score", 0.0)),
            text=r.get("text", ""),
        )
        for r in results
    ]

    return QueryResponse(
        answer=answer,
        citations=citations,
        used_llm=used_llm,
        debug={"top_k": top_k, "search_q": search_q},
    )


@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 5):
    vs = VectorStore()
    results = vs.search(q, top_k=top_k, mmr_lambda=settings.MMR_LAMBDA, mmr_candidates=settings.MMR_CANDIDATES)
    citations = [
        Citation(
            chunk_id=r["id"],
            section_title=r.get("section_title"),
            page_start=r.get("page_start"),
            page_end=r.get("page_end"),
            score=float(r.get("score", 0.0)),
            text=r.get("text", ""),
        )
        for r in results
    ]
    return SearchResponse(results=citations)


@app.get("/stats", response_model=StatsResponse)
def stats():
    meta_path = os.path.join(settings.STORAGE_DIR, "meta.json")
    chunks_path = os.path.join(settings.STORAGE_DIR, "chunks.jsonl")
    has_index = os.path.isfile(os.path.join(settings.STORAGE_DIR, "index.faiss"))
    pages = sections = chunks = 0
    if os.path.isfile(meta_path):
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        pages = int(m.get("pages", 0))
        sections = int(m.get("sections", 0))
        chunks = int(m.get("chunks", 0))
    return StatsResponse(
        chunks=chunks,
        sections=sections,
        pages=pages,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        rerank_enabled=settings.RERANK,
        has_index=has_index,
        storage_dir=os.path.abspath(settings.STORAGE_DIR),
    )
