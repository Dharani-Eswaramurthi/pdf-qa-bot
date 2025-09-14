import os
import json
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF

from .settings import settings
from .utils.tokenize import smart_split, rough_token_count


@dataclass
class Section:
    title: str
    level: int
    page_start: int
    page_end: int
    text: str
    id: Optional[str] = None


def _ensure_dirs() -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)


def load_pdf_blocks(pdf_path: str) -> Tuple[int, List[Dict[str, Any]]]:
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    page_blocks: List[Dict[str, Any]] = []
    for i in range(pages):
        page = doc.load_page(i)
        d = page.get_text("dict")
        d["plain_text"] = page.get_text("text")
        page_blocks.append(d)
    return pages, page_blocks


def detect_headings(page_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    # Returns list of (line_text, max_font_size) candidates
    lines: List[Tuple[str, float]] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join([span.get("text", "") for span in line.get("spans", [])]).strip()
            if not text:
                continue
            max_size = max((span.get("size", 0.0) for span in line.get("spans", [])), default=0.0)
            lines.append((text, max_size))
    # Heuristic: top 25% by font size are headings
    if not lines:
        return []
    sizes = sorted([s for _, s in lines])
    thr_idx = max(0, int(0.75 * (len(sizes) - 1)))
    size_thr = sizes[thr_idx]
    candidates = [(t, s) for (t, s) in lines if s >= size_thr and len(t) < 180]
    return candidates


def build_sections(page_blocks: List[Dict[str, Any]]) -> List[Section]:
    sections: List[Section] = []
    cur_title = "Document"
    cur_level = 1
    cur_text_parts: List[str] = []
    cur_start = 1

    for idx, page in enumerate(page_blocks, start=1):
        text_page = page.get("plain_text") or ""

        candidates = detect_headings(page)

        # If we detect strong heading candidates early in page, prefer first
        new_section_started = False
        selected_heading: Optional[str] = None
        if candidates:
            # Pick the first non-empty candidate as heading
            selected_heading = candidates[0][0].strip()
            # Basic level heuristic by numbering pattern
            if selected_heading[:4].strip().lower() in {"toc", "index"}:
                selected_heading = None
            else:
                level = 1
                if selected_heading[:10].strip()[:1].isdigit():
                    dots = selected_heading.split()[0].count(".")
                    level = min(3, dots + 1)
                # commit previous section
                if cur_text_parts:
                    sections.append(
                        Section(
                            title=cur_title,
                            level=cur_level,
                            page_start=cur_start,
                            page_end=idx - 1,
                            text="".join(cur_text_parts).strip(),
                            id=str(uuid.uuid4()),
                        )
                    )
                cur_title = selected_heading
                cur_level = level
                cur_text_parts = []
                cur_start = idx
                new_section_started = True

        cur_text_parts.append(f"\n\n[Page {idx}]\n")
        cur_text_parts.append(text_page)

    # flush last section
    if cur_text_parts:
        sections.append(
            Section(
                title=cur_title,
                level=cur_level,
                page_start=cur_start,
                page_end=len(page_blocks),
                text="".join(cur_text_parts).strip(),
                id=str(uuid.uuid4()),
            )
        )

    return sections


def chunk_sections(
    sections: List[Section],
    chunk_tokens: int,
    overlap_tokens: int,
    max_chunks: int = 0,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for s in sections:
        pieces = smart_split(s.text, chunk_tokens, overlap_tokens)
        for i, txt in enumerate(pieces):
            chunk_id = str(uuid.uuid4())
            chunks.append(
                {
                    "id": chunk_id,
                    "section_title": s.title,
                    "section_id": s.id,
                    "level": s.level,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "chunk_index": i,
                    "text": txt,
                    "approx_tokens": rough_token_count(txt),
                }
            )
            if max_chunks and len(chunks) >= max_chunks:
                return chunks
    return chunks


def ingest_pdf(pdf_path: Optional[str] = None) -> Dict[str, Any]:
    _ensure_dirs()
    path = pdf_path or settings.DEFAULT_PDF_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found at: {path}")

    pages, page_blocks = load_pdf_blocks(path)
    sections = build_sections(page_blocks)
    chunks = chunk_sections(sections, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP, settings.MAX_CHUNKS)

    # Persist raw chunks to JSON for transparency/debug
    chunks_path = os.path.join(settings.STORAGE_DIR, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Persist sections as well
    sections_path = os.path.join(settings.STORAGE_DIR, "sections.jsonl")
    with open(sections_path, "w", encoding="utf-8") as f:
        for s in sections:
            rec = {
                "id": s.id,
                "title": s.title,
                "level": s.level,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "text": s.text,
                "approx_tokens": rough_token_count(s.text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta_path = os.path.join(settings.STORAGE_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf_path": os.path.abspath(path),
                "pages": pages,
                "sections": len(sections),
                "chunks": len(chunks),
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "sections_path": sections_path,
                "chunks_path": chunks_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "pages": pages,
        "sections": len(sections),
        "chunks": len(chunks),
        "chunks_path": chunks_path,
        "sections_path": sections_path,
        "meta_path": meta_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a PDF into chunks")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    args = parser.parse_args()

    info = ingest_pdf(args.pdf)
    print(json.dumps(info, indent=2))
