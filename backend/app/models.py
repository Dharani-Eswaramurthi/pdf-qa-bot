from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class IngestResponse(BaseModel):
    chunks_indexed: int
    sections_detected: int
    pages_processed: int
    storage_dir: str


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_llm: Optional[bool] = None
    history: Optional[List[ChatMessage]] = None


class Citation(BaseModel):
    chunk_id: str
    section_title: Optional[str]
    page_start: int
    page_end: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_llm: bool
    debug: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[Citation]


class StatsResponse(BaseModel):
    chunks: int
    sections: int
    pages: int
    embedding_model: str
    rerank_enabled: bool
    has_index: bool
    storage_dir: str
