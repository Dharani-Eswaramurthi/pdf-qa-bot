import os
from typing import List, Dict, Any, Optional

from .settings import settings
from dotenv import load_dotenv

load_dotenv(override=True)


SYS_PROMPT = (
    "You are Patrick AI, a precise, thoughtful Q&A assistant for a specific PDF manual. "
    "Priorities: (1) Ground answers ONLY in provided context; (2) Use session context to interpret pronouns and follow instructions; "
    "(3) Ask a brief clarifying question if the request is ambiguous; (4) Be concise and conversational. "
    "Style: Respond in clean Markdown using short paragraphs and bullet lists. Avoid tables by default. "
    "Only use tables when the user explicitly requests a table or when the data is inherently tabular (e.g., a change log matrix). Do not place questions or narrative content into tables. "
    "When listing itemized changes (and a tabular view is appropriate), a 4-column table is acceptable: Item | Change | Description | Source. "
    "Always include citations as [page_start-page_end, section title] when relevant. If the answer is outside the manual, explain why and offer a next step."
)


def build_prompt(
    question: str,
    contexts: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    directives: Optional[str] = None,
) -> str:
    conv_str = ""
    if history:
        turns = []
        for m in history[-8:]:  # last 8 messages to keep prompt compact
            role = m.get("role", "user").capitalize()
            content = m.get("content", "").strip()
            if not content:
                continue
            turns.append(f"{role}: {content}")
        if turns:
            conv_str = "\n".join(turns)

    ctx_strs = []
    for i, c in enumerate(contexts, start=1):
        title = c.get("section_title") or ""
        pages = f"p.{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") else ""
        ctx_strs.append(f"[Context {i} | {pages} | {title}]\n{c['text']}")
    context_block = "\n\n".join(ctx_strs)
    prompt = []
    if directives:
        prompt.append(f"Session directives:\n{directives.strip()}")
    if conv_str:
        prompt.append(f"Conversation so far:\n{conv_str}")
    prompt.append(f"Context:\n{context_block}")
    prompt.append(f"Question: {question}")
    prompt.append("Answer:")
    return "\n\n".join(prompt)


def extractive_answer(question: str, contexts: List[Dict[str, Any]], directives: Optional[str] = None) -> str:
    # Simple heuristic: return stitched summary-like response with inline citations
    if directives and ('Do not reveal answers' in directives or 'Quiz mode' in directives.lower()):
        # In quiz/evaluation mode, avoid revealing; invite user response
        return (
            "I will evaluate your answer against the manual. Please reply with your answer.\n\n"
            + "Sources:\n"
            + "\n".join(
                f"- p.{c.get('page_start')}-{c.get('page_end')} • {c.get('section_title') or ''}"
                for c in contexts
            )
        )
    parts = []
    for c in contexts:
        pages = f"p.{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") else ""
        title = c.get("section_title") or ""
        excerpt = c["text"].strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600].rsplit(" ", 1)[0] + "…"
        parts.append(f"- {excerpt} [{pages}, {title}]")
    return "\n".join(parts)


def generate_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    directives: Optional[str] = None,
) -> (str, bool):
    used_llm = False
    answer = None
    if settings.USE_LLM and settings.HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=settings.HF_PROVIDER, api_key=settings.HF_TOKEN)
            prompt = build_prompt(question, contexts, history, directives)
            completion = client.chat.completions.create(
                model=settings.HF_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = completion.choices[0].message.get("content") if completion and completion.choices else None
            if content:
                answer = content.strip()
                used_llm = True
        except Exception:
            answer = None
    if not answer:
        answer = extractive_answer(question, contexts, directives)
    return answer, used_llm


def generate_hypothetical_answer(question: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """HyDE: produce a short hypothetical answer to enrich retrieval."""
    try:
        if settings.LLM_PROVIDER.lower() == "hf" and settings.HF_TOKEN:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=settings.HF_PROVIDER, api_key=settings.HF_TOKEN)
            sys = (
                "You are generating a brief hypothetical answer (2-4 sentences) to a question. "
                "Keep it generic and likely given a clinical assessment manual, avoiding hallucinated specifics."
            )
            conv = []
            if history:
                for m in history[-4:]:
                    conv.append(f"{m.get('role','user').capitalize()}: {m.get('content','').strip()}")
            prompt = ("\n".join(conv) + "\n" if conv else "") + f"Question: {question}\nHypothetical answer:"
            resp = client.chat.completions.create(
                model=settings.HF_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = resp.choices[0].message.get("content") if resp and resp.choices else None
            return content.strip() if content else None
    except Exception:
        return None
    return None


def generate_quiz_question(contexts: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None, directives: Optional[str] = None) -> Optional[str]:
    """Ask a single exam-style question from provided contexts. Avoid tables, no answers."""
    sys = (
        "You are Patrick AI. Create ONE exam-style question from the provided context. "
        "Do NOT include the answer. Avoid tables. Keep it concise and unambiguous."
    )
    try:
        if settings.LLM_PROVIDER.lower() == "hf" and settings.HF_TOKEN:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=settings.HF_PROVIDER, api_key=settings.HF_TOKEN)
            ctx_strs = []
            for i, c in enumerate(contexts[:4], start=1):
                title = c.get("section_title") or ""
                pages = f"p.{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") else ""
                ctx_strs.append(f"[Context {i} | {pages} | {title}]\n{c['text']}")
            prompt = (
                (f"Session directives:\n{directives}\n\n" if directives else "")
                + "Context:\n" + "\n\n".join(ctx_strs)
                + "\n\nTask: Write ONE question the user should answer. Do not include the answer."
            )
            completion = client.chat.completions.create(
                model=settings.HF_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = completion.choices[0].message.get("content") if completion and completion.choices else None
            if content:
                q = content.strip()
                if not q.endswith("\n"):
                    q += "\n"
                q += "\nPlease type your answer; I’ll validate it against the manual."
                return q
    except Exception:
        pass
    # Fallback: naive question from first context
    if contexts:
        c = contexts[0]
        title = c.get("section_title") or "the manual"
        pages = f"p.{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") else ""
        return f"According to {title} ({pages}), summarize the key instruction described."
    return "From the manual, describe the key instruction mentioned in the provided section."


def validate_user_answer(user_answer: str, contexts: List[Dict[str, Any]], directives: Optional[str] = None) -> str:
    """Evaluate user's answer against contexts; be concise and cite. Respect no-reveal if requested."""
    no_reveal = False
    if directives and ("Do not reveal answers" in directives or "do not reveal" in directives.lower()):
        no_reveal = True
    sys = (
        "You are Patrick AI. Evaluate the user's answer against the provided context. "
        "Be concise and cite pages/section."
        + (" Do NOT reveal the correct answer; only say Correct/Incorrect with a brief hint and citations." if no_reveal else " Provide a brief correction if incorrect, citing the manual.")
    )
    try:
        if settings.LLM_PROVIDER.lower() == "hf" and settings.HF_TOKEN:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=settings.HF_PROVIDER, api_key=settings.HF_TOKEN)
            ctx_strs = []
            for i, c in enumerate(contexts[:4], start=1):
                title = c.get("section_title") or ""
                pages = f"p.{c.get('page_start')}-{c.get('page_end')}" if c.get("page_start") else ""
                ctx_strs.append(f"[Context {i} | {pages} | {title}]\n{c['text']}")
            prompt = (
                (f"Session directives:\n{directives}\n\n" if directives else "")
                + "Context:\n" + "\n\n".join(ctx_strs)
                + f"\n\nUser answer:\n{user_answer}\n\nTask: Evaluate the user's answer."
            )
            completion = client.chat.completions.create(
                model=settings.HF_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = completion.choices[0].message.get("content") if completion and completion.choices else None
            if content:
                return content.strip()
    except Exception:
        pass
    # Fallback: extractive minimal evaluation
    lines = ["Based on the manual excerpts, your answer will be evaluated when more detail is available."]
    if contexts:
        lines.append("Sources:")
        lines.extend(
            f"- p.{c.get('page_start')}-{c.get('page_end')} • {c.get('section_title') or ''}" for c in contexts[:4]
        )
    return "\n".join(lines)
