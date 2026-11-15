"""Ask Fire-Margin: retrieval-augmented answers grounded ONLY in data/knowledge/chunks.jsonl.

Flow: retrieve (TF-IDF, threshold) -> refuse if nothing grounded -> LLM via src.llm_config.chat
(Meta first, then NVIDIA) -> if every provider fails, return the retrieved passages instead.
Never raises for normal input; the app caches answers with st.cache_data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.knowledge import Hit, Retriever
from src.llm_config import UNAVAILABLE_MESSAGE, LLMResult, chat

REFUSAL = "I don't have NASA evidence for that."
STARTER_QUESTIONS = (
    "Is a Moon base at 34% O2 safe from fire?",
    "Why do droplets burst at high oxygen?",
    "What did Apollo 1 teach us?",
    "Can a fire keep burning if the ventilation is turned off?",
    "Which experiment should NASA run next?",
)
SYSTEM_PROMPT = f"""You are Fire-Margin, a fire-safety assistant for NASA exploration habitats.
Rules (follow all of them):
1. Use ONLY the numbered source passages provided. Do not use outside knowledge.
2. Every factual sentence must end with its citation exactly as given, e.g. [source, p. 12].
3. If the passages do not answer the question, reply exactly: "{REFUSAL}"
4. Distinguish PSI data, literature points (not in PSI) and Fire-Margin analysis when it matters.
5. Be concise (at most 150 words). Never invent numbers, dates or experiments."""


@dataclass
class RagAnswer:
    status: str                      # "answered" | "refused" | "unavailable"
    text: str
    hits: list[Hit] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None

    @property
    def citations(self) -> list[str]:
        return [h.chunk.cite() for h in self.hits]

    def to_dict(self) -> dict:
        return {"status": self.status, "text": self.text, "provider": self.provider, "model": self.model,
                "sources": [{"cite": h.chunk.cite(), "url": h.chunk.url, "score": round(h.score, 3),
                             "text": h.chunk.text} for h in self.hits]}


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict[str, str]]:
    passages = "\n\n".join(f"({i}) {h.chunk.cite()}\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    user = f"Source passages:\n{passages}\n\nQuestion: {question}\nAnswer with citations:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def fallback_text(hits: Sequence[Hit]) -> str:
    lines = [UNAVAILABLE_MESSAGE, ""]
    lines += [f"- {h.chunk.text[:300].rstrip()}… {h.chunk.cite()}" for h in hits]
    return "\n".join(lines)


def answer(question: str, retriever: Retriever,
           chat_fn: Callable[..., LLMResult] = chat) -> RagAnswer:
    """Grounded answer, refusal, or retrieved passages when the AI is unavailable."""
    question = (question or "").strip()
    hits = retriever.search(question) if question else []
    if not hits:
        return RagAnswer("refused", REFUSAL)
    try:
        result = chat_fn(build_messages(question, hits))
    except Exception as exc:  # noqa: BLE001 - contract: never crash the app
        result = LLMResult(False, UNAVAILABLE_MESSAGE, error=type(exc).__name__)
    if not result.ok:
        return RagAnswer("unavailable", fallback_text(hits), hits)
    text = result.text.strip()
    if text.startswith(REFUSAL[:25]):
        return RagAnswer("refused", REFUSAL, hits, result.provider, result.model)
    if "[" not in text:   # model ignored the citation rule: attach the sources it was given
        text += "\n\nSources: " + "; ".join(dict.fromkeys(h.chunk.cite() for h in hits))
    return RagAnswer("answered", text, hits, result.provider, result.model)


def main() -> None:
    from src.knowledge import load_chunks

    r = Retriever(load_chunks())
    for q in (*STARTER_QUESTIONS[:3], "best pizza recipe in Naples"):
        a = answer(q, r)
        print(f"\nQ: {q}\n[{a.status}] {a.text[:700]}")


if __name__ == "__main__":
    main()
