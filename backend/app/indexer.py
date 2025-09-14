import os
import threading
import time
from typing import Optional, Dict, Any

from .settings import settings
from .ingest import ingest_pdf
from .retriever import VectorStore


class Indexer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.status: str = "idle"  # idle | indexing | ready | error
        self.progress: int = 0
        self.message: str = ""
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.last_error: Optional[str] = None

    def _set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _progress(self, pct: int, msg: str):
        self._set(progress=max(0, min(100, int(pct))), message=msg)

    def _run(self, pdf_path: str):
        self._set(status="indexing", progress=1, message="Starting indexing…", started_at=time.time(), ended_at=None, last_error=None)
        try:
            # Ingest PDF -> chunks/sections
            self._progress(10, "Parsing and chunking PDF…")
            info = ingest_pdf(pdf_path)
            self._progress(55, f"Parsed {info['pages']} pages, {info['chunks']} chunks. Loading embedding model…")

            # Build vector store with coarse progress reporting
            vs = VectorStore()
            self._progress(56, "Building vectors…")

            # Approximate progress: chunks encode (~40%), sections encode (~8%), write index (~2%)
            def cb(phase: str, done: int, total: int):
                # Normalize per-phase weighting
                weights = {"chunks": 40.0, "sections": 8.0, "finalize": 2.0}
                base = 55.0
                acc = base
                if phase == "chunks" and total:
                    acc += weights["chunks"] * (done / max(1, total))
                elif phase == "sections" and total:
                    acc += weights["sections"] * (done / max(1, total)) + weights["chunks"]
                elif phase == "finalize":
                    acc = 98.0
                self._progress(int(acc), f"Indexing: {phase} {done}/{total}…")

            vs.build(progress_cb=cb)
            self._progress(100, "Index ready")
            self._set(status="ready", ended_at=time.time())
        except Exception as e:
            self._set(status="error", last_error=str(e), ended_at=time.time())

    def start(self, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        path = pdf_path or settings.DEFAULT_PDF_PATH
        # If already indexing, return status
        with self._lock:
            if self.status == "indexing" and self._thread and self._thread.is_alive():
                return self.status_json()
        # If index exists, mark ready
        if os.path.isfile(os.path.join(settings.STORAGE_DIR, "index.faiss")):
            self._set(status="ready", progress=100, message="Index ready")
            return self.status_json()
        # Start new indexing thread
        t = threading.Thread(target=self._run, args=(path,), daemon=True)
        self._thread = t
        t.start()
        return self.status_json()

    def status_json(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "last_error": self.last_error,
            }


indexer = Indexer()
