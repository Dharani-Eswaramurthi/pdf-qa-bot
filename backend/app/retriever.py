import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from .settings import settings


class VectorStore:
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or settings.STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "index.faiss")
        self.store_path = os.path.join(self.storage_dir, "store.jsonl")
        # Section-level (optional)
        self.s_index_path = os.path.join(self.storage_dir, "index_sections.faiss")
        self.s_store_path = os.path.join(self.storage_dir, "store_sections.jsonl")
        self.meta_path = os.path.join(self.storage_dir, "meta.json")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None  # type: Optional[faiss.Index]
        self.s_index = None  # type: Optional[faiss.Index]
        self._reranker: Optional[CrossEncoder] = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        return X / norms

    def _load_chunks(self) -> List[Dict[str, Any]]:
        raw_path = os.path.join(self.storage_dir, "chunks.jsonl")
        if not os.path.isfile(raw_path):
            raise FileNotFoundError("chunks.jsonl not found; run ingestion first")
        items = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _load_sections(self) -> List[Dict[str, Any]]:
        raw_path = os.path.join(self.storage_dir, "sections.jsonl")
        if not os.path.isfile(raw_path):
            return []
        items = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def build(self) -> Dict[str, Any]:
        chunks = self._load_chunks()
        texts = [c["text"] for c in chunks]
        emb = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        emb = self._normalize(emb.astype("float32"))

        index = faiss.IndexFlatIP(self.dim)
        index.add(emb)
        faiss.write_index(index, self.index_path)

        # Persist store to align IDs to rows
        with open(self.store_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        # Build sections index if sections exist
        sections = self._load_sections()
        if sections:
            s_texts = [s["text"] for s in sections]
            s_emb = self.model.encode(s_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            s_emb = self._normalize(s_emb.astype("float32"))
            s_index = faiss.IndexFlatIP(self.dim)
            s_index.add(s_emb)
            faiss.write_index(s_index, self.s_index_path)
            with open(self.s_store_path, "w", encoding="utf-8") as f:
                for s in sections:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Update meta
        meta = {}
        if os.path.isfile(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.update(
            {
                "embeddings": len(chunks),
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "faiss_index": os.path.abspath(self.index_path),
                "faiss_index_sections": os.path.abspath(self.s_index_path) if sections else None,
                "sections_embedded": len(sections),
            }
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {"embeddings": len(chunks), "dim": self.dim}

    def _ensure_loaded(self):
        if self.index is None:
            if not os.path.isfile(self.index_path):
                raise FileNotFoundError("FAISS index missing; build the index first")
            self.index = faiss.read_index(self.index_path)
        if self.s_index is None and os.path.isfile(self.s_index_path):
            self.s_index = faiss.read_index(self.s_index_path)

    def _iter_store(self):
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _load_store(self) -> List[Dict[str, Any]]:
        return list(self._iter_store())

    def _encode_query(self, texts: List[str]) -> np.ndarray:
        q = self.model.encode(texts, convert_to_numpy=True)
        q = q.astype("float32")
        q = self._normalize(q)
        return q

    def search(self, query: str, top_k: int = 5, mmr_lambda: float = 0.5, mmr_candidates: int = 24) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self.index is not None
        q = self.model.encode([query], convert_to_numpy=True)
        q = q.astype("float32")
        q = self._normalize(q)
        scores, idxs = self.index.search(q, k=min(max(top_k * 4, mmr_candidates), self.index.ntotal))
        scores = scores[0]
        idxs = idxs[0]

        # Load corresponding items
        store = self._load_store()
        candidates = [(int(i), float(s)) for i, s in zip(idxs, scores) if i != -1]

        # MMR selection
        selected: List[int] = []
        selected_scores: List[float] = []
        cand_vecs = self._normalize(self.model.encode([store[i]["text"] for i, _ in candidates], convert_to_numpy=True).astype("float32"))
        q_vec = q[0]
        cand_idxs = [i for i, _ in candidates]

        while len(selected) < min(top_k, len(candidates)) and cand_idxs:
            if not selected:
                # pick best by score
                best = int(np.argmax([s for _, s in candidates]))
                chosen_global_idx = candidates[best][0]
                chosen_pos = best
            else:
                selected_vecs = cand_vecs[[cand_idxs.index(si) for si in selected if si in cand_idxs]] if selected else None
                sim_to_query = cand_vecs @ q_vec
                if selected_vecs is None or selected_vecs.size == 0:
                    diversity = np.zeros(len(cand_idxs))
                else:
                    diversity = cand_vecs @ selected_vecs.T
                    diversity = diversity.max(axis=1)
                mmr = mmr_lambda * sim_to_query + (1 - mmr_lambda) * (1 - diversity)
                chosen_pos = int(np.argmax(mmr))
                chosen_global_idx = cand_idxs[chosen_pos]

            selected.append(chosen_global_idx)
            selected_scores.append(float(candidates[chosen_pos][1]))
            # remove chosen
            del cand_idxs[chosen_pos]
            cand_vecs = np.delete(cand_vecs, chosen_pos, axis=0)
            candidates.pop(chosen_pos)

        # Build preliminary results
        results: List[Dict[str, Any]] = []
        for i, score in zip(selected, selected_scores):
            item = store[i]
            results.append({
                "id": item["id"],
                "text": item["text"],
                "section_title": item.get("section_title"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "score": float(score),
            })

        # Optional: rerank with cross-encoder for better ordering
        if settings.RERANK and results:
            try:
                if self._reranker is None:
                    self._reranker = CrossEncoder(settings.RERANK_MODEL_NAME)
                pairs = [(query, r["text"]) for r in results]
                rerank_scores = self._reranker.predict(pairs)
                for r, s in zip(results, rerank_scores):
                    r["rerank_score"] = float(s)
                results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
            except Exception:
                # If reranker fails (e.g. no model), return original order
                pass

        return results

    def search_advanced(
        self,
        query: str,
        top_k: int = 5,
        mmr_lambda: float = 0.5,
        mmr_candidates: int = 24,
        hyde: Optional[str] = None,
        use_hierarchical: Optional[bool] = None,
        section_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self.index is not None
        use_h = settings.HIERARCHICAL_RETRIEVAL if use_hierarchical is None else use_hierarchical
        sec_k = settings.SECTION_TOP_K if section_top_k is None else section_top_k

        # Build query embedding (with HyDE enrichment if provided)
        if hyde:
            q_vec = self._encode_query([query, hyde]).mean(axis=0, keepdims=True)
        else:
            q_vec = self._encode_query([query])

        scores, idxs = self.index.search(q_vec, k=min(max(top_k * 8, mmr_candidates * 2), self.index.ntotal))
        scores = scores[0]
        idxs = idxs[0]
        store = self._load_store()

        allowed_sections: Optional[Set[str]] = None
        if use_h and self.s_index is not None and os.path.isfile(self.s_store_path):
            s_scores, s_idxs = self.s_index.search(q_vec, k=min(sec_k, self.s_index.ntotal))
            s_idxs = s_idxs[0]
            with open(self.s_store_path, "r", encoding="utf-8") as f:
                sec_store = [json.loads(l) for l in f if l.strip()]
            s_ids = []
            for i in s_idxs:
                if i != -1 and i < len(sec_store):
                    s_ids.append(sec_store[i].get("id"))
            allowed_sections = set([sid for sid in s_ids if sid]) if s_ids else None

        candidates = [(int(i), float(s)) for i, s in zip(idxs, scores) if i != -1]
        if allowed_sections:
            filtered = [(i, s) for (i, s) in candidates if store[i].get("section_id") in allowed_sections]
            if len(filtered) >= max(top_k * 2, mmr_candidates):
                candidates = filtered

        # MMR selection
        selected: List[int] = []
        selected_scores: List[float] = []
        cand_texts = [store[i]["text"] for i, _ in candidates]
        cand_vecs = self._normalize(self.model.encode(cand_texts, convert_to_numpy=True).astype("float32"))
        q_single = q_vec[0]
        cand_idxs = [i for i, _ in candidates]
        while len(selected) < min(top_k, len(candidates)) and cand_idxs:
            if not selected:
                best = int(np.argmax([s for _, s in candidates]))
                chosen_global_idx = candidates[best][0]
                chosen_pos = best
            else:
                selected_vecs = (
                    cand_vecs[[cand_idxs.index(si) for si in selected if si in cand_idxs]] if selected else None
                )
                sim_to_query = cand_vecs @ q_single
                if selected_vecs is None or selected_vecs.size == 0:
                    diversity = np.zeros(len(cand_idxs))
                else:
                    diversity = cand_vecs @ selected_vecs.T
                    diversity = diversity.max(axis=1)
                mmr = mmr_lambda * sim_to_query + (1 - mmr_lambda) * (1 - diversity)
                chosen_pos = int(np.argmax(mmr))
                chosen_global_idx = cand_idxs[chosen_pos]
            selected.append(chosen_global_idx)
            selected_scores.append(float(candidates[chosen_pos][1]))
            del cand_idxs[chosen_pos]
            cand_vecs = np.delete(cand_vecs, chosen_pos, axis=0)
            candidates.pop(chosen_pos)

        results: List[Dict[str, Any]] = []
        for i, score in zip(selected, selected_scores):
            item = store[i]
            results.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "section_title": item.get("section_title"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": float(score),
                }
            )

        if settings.RERANK and results:
            try:
                if self._reranker is None:
                    self._reranker = CrossEncoder(settings.RERANK_MODEL_NAME)
                pairs = [(query, r["text"]) for r in results]
                rerank_scores = self._reranker.predict(pairs)
                for r, s in zip(results, rerank_scores):
                    r["rerank_score"] = float(s)
                results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
            except Exception:
                pass

        return results
