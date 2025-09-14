import re
from typing import List


_delims = ["\n\n", "\n- ", "\n• ", "\n* ", ". "]


def rough_token_count(text: str) -> int:
    # Approximate tokens: assume ~5 chars per token
    return max(1, int(len(text) / 5))


def smart_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []
    if rough_token_count(text) <= max_tokens:
        return [text.strip()]

    parts = [text]
    for d in _delims:
        new_parts = []
        for p in parts:
            if rough_token_count(p) <= max_tokens:
                new_parts.append(p)
            else:
                new_parts.extend([s for s in p.split(d) if s])
        parts = new_parts

    # If still too big, fall back to hard split by words
    chunks: List[str] = []
    for p in parts:
        words = re.split(r"(\s+)", p)
        cur: List[str] = []
        cur_tokens = 0
        for w in words:
            t = rough_token_count(w)
            if cur_tokens + t > max_tokens and cur:
                chunks.append("".join(cur).strip())
                # overlap by characters approximated from tokens
                overlap_chars = max(0, overlap_tokens * 5)
                if overlap_chars > 0 and chunks[-1]:
                    tail = chunks[-1][-overlap_chars:]
                else:
                    tail = ""
                cur = [tail]
                cur_tokens = rough_token_count(tail)
            cur.append(w)
            cur_tokens += t
        if cur:
            chunks.append("".join(cur).strip())

    return [c for c in chunks if c]

